#!/usr/bin/env python
# coding=utf-8

import os
import sys
import json
import logging
import random
import time
import glob
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import gin
from absl import app, flags
import jax
from jax import numpy as jnp
from flax.training import train_state
from flax.training import checkpoints as tcheckpoints
from flax import jax_utils
import optax
from functools import partial
import sentencepiece as spm

# 设置路径
AG4MDIR = '/kaggle/working/ag4masses'
AGLIB = '/kaggle/working/aglib'
AGDIR = f"{AG4MDIR}/alphageometry"
MELIAD_PATH = f"{AGLIB}/meliad"
DATA = f"{AGLIB}/ag_ckpt_vocab"
TESTDIR = f"/kaggle/working/ag4mtest"
DPO_DIR = f"/kaggle/working/dpo_training"

# 确保目录存在
os.makedirs(TESTDIR, exist_ok=True)
os.makedirs(DPO_DIR, exist_ok=True)

# 添加 meliad 到 Python 路径
sys.path.append(MELIAD_PATH)
sys.path.append(f"{MELIAD_PATH}/transformer")

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FLAGS = flags.FLAGS

# 配置参数
flags.DEFINE_string('gin_search_paths', f'{MELIAD_PATH}/transformer/configs', 'Gin配置文件路径')
flags.DEFINE_multi_string('gin_file', ['base_htrans.gin'], 'Gin配置文件列表')
flags.DEFINE_multi_string('gin_param', None, 'Gin参数绑定')
flags.DEFINE_string('ckpt_path', f'{DATA}/checkpoint_10999999', '初始模型检查点路径')
flags.DEFINE_string('vocab_path', f'{DATA}/geometry.757.model', '词汇表文件路径')
flags.DEFINE_string('train_data', f'{DPO_DIR}/train_data.jsonl', '训练数据路径')
flags.DEFINE_string('output_dir', f'{DPO_DIR}/checkpoints', '输出目录')
flags.DEFINE_float('beta', 0.1, 'DPO温度参数')
flags.DEFINE_float('learning_rate', 5e-5, '学习率')
flags.DEFINE_integer('batch_size', 4, '批大小')
flags.DEFINE_integer('max_seq_length', 512, '最大序列长度')
flags.DEFINE_integer('num_epochs', 30, '训练轮数')
flags.DEFINE_integer('steps_per_epoch', 100, '每轮步数')
flags.DEFINE_integer('save_steps', 50, '保存间隔步数')
flags.DEFINE_bool('use_amp', True, '启用混合精度训练')

# 定义TransformerTaskConfig类
@gin.configurable
class TransformerTaskConfig:
    """Transformer任务配置"""
    def __init__(
        self, 
        vocab_size: int, 
        sequence_length: int, 
        batch_size: int, 
        max_sequence_length: int
    ):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length

class CustomTrainState:
    """自定义训练状态结构"""
    def __init__(self, step, params, model_state, optimizer_state):
        self.step = step
        self.params = params
        self.model_state = model_state
        self.optimizer_state = optimizer_state

class DPOTrainer:
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        beta: float = 0.1,
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        max_seq_length: int = 512,
        use_amp: bool = True
    ):
        # 减少JAX输出
        jax.config.update('jax_debug_nans', True)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')
        
        # 配置日志
        self._configure_logging()
        
        # 初始化组件
        self.vocab = self._create_sentencepiece_vocab(vocab_path)
        self._init_gin_config(model_path)
        self.task_config = self._create_task_config()
        self.model = self._init_model()
        self.tstate = self._init_training_state(model_path)
        
        # 配置训练参数
        self.beta = beta
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.global_step = 0
        self.optimizer = self._create_optimizer(learning_rate)
        self.use_amp = use_amp
            
    def _configure_logging(self):
        """配置日志输出级别"""
        logging.getLogger('jax').setLevel(logging.WARNING)
        logging.getLogger('flax').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        
        silenced_modules = [
            'models', 'decoder_stack', 'transformer_layer',
            'attention', 'transformer_base', 'nn_components'
        ]
        
        # 降低特定模块的日志级别
        for module in silenced_modules:
            logging.getLogger(f'transformer.{module}').setLevel(logging.ERROR)
    
    def _create_sentencepiece_vocab(self, model_path: str):
        """创建SentencePiece词汇表"""
        class CustomSPVocabulary:
            def __init__(self, model_path):
                self.sp = spm.SentencePieceProcessor()
                self.sp.Load(model_path)
                self.vocab_size = self.sp.GetPieceSize()
            
            def encode(self, text: str) -> List[int]:
                return self.sp.EncodeAsIds(text)
            
            def decode(self, ids: List[int]) -> str:
                return self.sp.DecodeIds(ids)
        
        return CustomSPVocabulary(model_path)
        
    def _init_gin_config(self, model_path: str):
        """初始化Gin配置"""
        # 配置Gin搜索路径
        gin.add_config_file_search_path(f'{MELIAD_PATH}/transformer/configs')
        
        # 配置要加载的文件
        config_files = [
            f'{MELIAD_PATH}/transformer/configs/base_htrans.gin',
            f'{MELIAD_PATH}/transformer/configs/size/medium_150M.gin',
            f'{MELIAD_PATH}/transformer/configs/options/positions_t5.gin',
            f'{MELIAD_PATH}/transformer/configs/trainer_configuration.gin'
        ]
        
        # 只添加存在的文件
        existing_files = [f for f in config_files if os.path.exists(f)]
        
        # 添加用户指定的文件
        if FLAGS.gin_file:
            for f in FLAGS.gin_file:
                if os.path.exists(f):
                    existing_files.append(f)
        
        # 记录配置
        logger.info(f"加载Gin配置文件: {', '.join(existing_files)}")
        
        # 解析配置
        try:
            gin.parse_config_files_and_bindings(
                existing_files, 
                FLAGS.gin_param
            )
        except Exception as e:
            logger.error(f"Gin配置错误: {str(e)}")
            raise
    
    def _create_task_config(self) -> TransformerTaskConfig:
        """创建任务配置"""
        return TransformerTaskConfig(
            vocab_size=self.vocab.vocab_size,
            sequence_length=512,
            batch_size=FLAGS.batch_size,
            max_sequence_length=512
        )
    
    def _init_model(self):
        """初始化模型"""
        from transformer.models import DecoderOnlyLanguageModel
        
        # 降低初始化期间的日志级别
        orig_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        try:
            model = DecoderOnlyLanguageModel(
                mode="train", 
                task_config=self.task_config
            )
            logger.info("模型初始化成功")
            return model
        finally:
            logging.getLogger().setLevel(orig_level)
    
    def _init_training_state(self, model_path: str) -> CustomTrainState:
        """初始化训练状态"""
        # 初始化模型参数
        rng = jax.random.PRNGKey(0)
        params_rng, dropout_rng = jax.random.split(rng)
        
        fake_input = self.model.get_fake_input()
        variables = self.model.init(
            {'params': params_rng, 'dropout': dropout_rng},
            fake_input
        )
        
        # 分离参数和状态
        params = variables.get('params', {})
        model_state = {k: v for k, v in variables.items() if k != 'params'}
        
        # 创建优化器
        optimizer = optax.adamw(learning_rate=1e-4)
        optimizer_state = optimizer.init(params)
        
        # 创建初始状态
        tstate = CustomTrainState(
            step=0,
            params=params,
            model_state=model_state,
            optimizer_state=optimizer_state
        )
        
        # 加载预训练权重
        if model_path:
            logger.info(f"尝试加载预训练权重: {model_path}")
            try:
                restored_state = tcheckpoints.restore_checkpoint(model_path, None)
                if restored_state:
                    tstate.params = restored_state.get('params', tstate.params)
                    tstate.model_state = restored_state.get('model_state', tstate.model_state)
                    logger.info("预训练权重加载成功")
            except Exception as e:
                logger.warning(f"权重加载失败: {str(e)}")
        
        return tstate
    
    def _create_optimizer(self, learning_rate: float) -> optax.GradientTransformation:
        """创建优化器"""
        return optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=learning_rate)
        )
    
    def _compute_log_probs(self, inputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        """计算对数概率"""
        # 创建模型输入
        input_dict = {
            "targets": inputs,
            "start_of_sequence": jnp.ones((inputs.shape[0]), dtype=jnp.bool_),
        }
        
        # 调用模型
        logits, _ = self.model.apply(
            {'params': self.tstate.params, **self.tstate.model_state},
            input_dict
        )
        
        # 计算log probabilities
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        # 收集目标token的log probs
        target_log_probs = jnp.take_along_axis(
            log_probs, 
            targets[..., None], 
            axis=-1
        ).squeeze(-1)
        
        # 应用mask
        valid_mask = (inputs != 0) & (targets != 0)
        return jnp.sum(target_log_probs * valid_mask, axis=1)
    
    def _dpo_loss(self, chosen_data: Tuple[jnp.ndarray, jnp.ndarray],
                 rejected_data: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """计算DPO损失"""
        chosen_inputs, chosen_targets = chosen_data
        rejected_inputs, rejected_targets = rejected_data
        
        # 计算选择序列的log概率
        chosen_logps = self._compute_log_probs(chosen_inputs, chosen_targets)
        
        # 计算拒绝序列的log概率
        rejected_logps = self._compute_log_probs(rejected_inputs, rejected_targets)
        
        # 计算DPO损失
        log_ratio = self.beta * (chosen_logps - rejected_logps)
        losses = -jax.nn.log_sigmoid(log_ratio)
        return jnp.mean(losses)
    
    def _update_step(self, chosen_data: Tuple[jnp.ndarray, jnp.ndarray],
                   rejected_data: Tuple[jnp.ndarray, jnp.ndarray]):
        """执行单个更新步骤"""
        # 计算损失和梯度
        loss, grads = jax.value_and_grad(self._dpo_loss)(chosen_data, rejected_data)
        
        # 更新参数
        updates, new_opt_state = self.optimizer.update(
            grads, self.tstate.optimizer_state, self.tstate.params
        )
        new_params = optax.apply_updates(self.tstate.params, updates)
        
        # 更新状态
        self.tstate = CustomTrainState(
            step=self.tstate.step + 1,
            params=new_params,
            model_state=self.tstate.model_state,
            optimizer_state=new_opt_state
        )
        
        self.global_step += 1
        return float(loss)
    
    def _prepare_batch(self, examples: List[Dict]) -> Tuple[Dict, Dict]:
        """准备批次数据"""
        # 初始化容器
        batch_data = {
            'chosen_inputs': [],
            'chosen_targets': [],
            'rejected_inputs': [],
            'rejected_targets': []
        }
        
        for ex in examples:
            # 编码选择文本
            chosen_tokens = self.vocab.encode(ex['chosen'])[:self.max_seq_length]
            chosen_inputs = np.zeros(self.max_seq_length, dtype=np.int32)
            chosen_inputs[:len(chosen_tokens)] = chosen_tokens
            
            chosen_targets = np.zeros(self.max_seq_length, dtype=np.int32)
            if len(chosen_tokens) > 0:
                chosen_targets[:len(chosen_tokens)-1] = chosen_tokens[1:]
                chosen_targets[len(chosen_tokens)-1] = self.vocab.sp.PieceToId('<eos>')
            
            # 编码拒绝文本
            rejected_tokens = self.vocab.encode(ex['rejected'])[:self.max_seq_length]
            rejected_inputs = np.zeros(self.max_seq_length, dtype=np.int32)
            rejected_inputs[:len(rejected_tokens)] = rejected_tokens
            
            rejected_targets = np.zeros(self.max_seq_length, dtype=np.int32)
            if len(rejected_tokens) > 0:
                rejected_targets[:len(rejected_tokens)-1] = rejected_tokens[1:]
                rejected_targets[len(rejected_tokens)-1] = self.vocab.sp.PieceToId('<eos>')
            
            batch_data['chosen_inputs'].append(chosen_inputs)
            batch_data['chosen_targets'].append(chosen_targets)
            batch_data['rejected_inputs'].append(rejected_inputs)
            batch_data['rejected_targets'].append(rejected_targets)
        
        # 转换为JAX数组
        return (
            jnp.array(batch_data['chosen_inputs']),
            jnp.array(batch_data['chosen_targets'])
        ), (
            jnp.array(batch_data['rejected_inputs']),
            jnp.array(batch_data['rejected_targets'])
        )
    
    def train(self, train_data: List[Dict], output_dir: str = None) -> str:
        """训练主循环"""
        output_dir = output_dir or FLAGS.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置训练日志
        log_file = os.path.join(output_dir, "dpo_training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        
        try:
            # 编译更新函数
            update_fn = jax.jit(self._update_step)
            
            for epoch in range(FLAGS.num_epochs):
                random.shuffle(train_data)
                epoch_losses = []
                
                with tqdm(total=min(FLAGS.steps_per_epoch, len(train_data)//FLAGS.batch_size), 
                         desc=f"Epoch {epoch+1}") as pbar:
                    for step in range(FLAGS.steps_per_epoch):
                        # 准备批次数据
                        batch_examples = train_data[step*FLAGS.batch_size: (step+1)*FLAGS.batch_size]
                        if not batch_examples:
                            break
                            
                        chosen_data, rejected_data = self._prepare_batch(batch_examples)
                        
                        # 执行更新
                        loss = update_fn(chosen_data, rejected_data)
                        epoch_losses.append(loss)
                        avg_loss = sum(epoch_losses) / len(epoch_losses)
                        
                        pbar.update(1)
                        pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
                        
                        # 定期保存检查点
                        if step > 0 and step % FLAGS.save_steps == 0:
                            self.save_checkpoint(output_dir)
                
                logger.info(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")
            
            # 最终保存
            self.save_checkpoint(output_dir, final=True)
            logger.info(f"训练完成，模型保存在: {output_dir}")
            return output_dir
        except Exception as e:
            logger.exception(f"训练失败: {str(e)}")
            raise
        finally:
            logger.removeHandler(file_handler)

    def save_checkpoint(self, output_dir: str, final: bool = False) -> str:
        """保存检查点"""
        save_name = "final_model" if final else f"checkpoint_{self.global_step}"
        save_path = os.path.join(output_dir, save_name)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存状态
        state_dict = {
            'global_step': self.global_step,
            'params': self.tstate.params,
            'model_state': self.tstate.model_state,
            'optimizer_state': self.tstate.optimizer_state
        }
        
        tcheckpoints.save_checkpoint(
            save_path,
            state_dict,
            step=self.global_step,
            overwrite=True
        )
        
        # 保存元数据
        metadata = {
            'global_step': self.global_step,
            'vocab_path': os.path.abspath(FLAGS.vocab_path),
            'beta': self.beta
        }
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"检查点保存至: {save_path}")
        return save_path

def load_and_validate_data(file_path: str) -> List[Dict]:
    """加载和验证训练数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"训练数据文件不存在: {file_path}")
    
    valid_data = []
    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                ex = json.loads(line)
                if 'prompt' in ex and 'chosen' in ex and 'rejected' in ex:
                    # 确保chosen评分更高
                    chosen_score = ex.get('chosen_score', float('inf'))
                    rejected_score = ex.get('rejected_score', float('-inf'))
                    
                    if chosen_score < rejected_score:
                        ex['chosen'], ex['rejected'] = ex['rejected'], ex['chosen']
                    valid_data.append(ex)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"数据加载完成: 有效样本={len(valid_data)}")
    return valid_data

def main(argv):
    # 解析参数
    if len(argv) > 1:
        raise app.UsageError('未知参数: ' + ' '.join(argv[1:]))
    
    # 验证必要参数
    required_params = ['ckpt_path', 'vocab_path', 'train_data']
    for param in required_params:
        if not getattr(FLAGS, param):
            raise ValueError(f"必须提供--{param}参数")
    
    try:
        # 设置输出目录
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        logger.info(f"输出目录: {FLAGS.output_dir}")
        
        # 初始化训练器
        trainer = DPOTrainer(
            model_path=FLAGS.ckpt_path,
            vocab_path=FLAGS.vocab_path,
            beta=FLAGS.beta,
            learning_rate=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size,
            max_seq_length=FLAGS.max_seq_length,
            use_amp=FLAGS.use_amp
        )
        
        # 加载数据
        train_data = load_and_validate_data(FLAGS.train_data)
        
        # 开始训练
        logger.info("开始DPO训练")
        output_dir = trainer.train(train_data)
        
        logger.info(f"训练完成，模型保存在: {output_dir}")
        return 0
    except Exception as e:
        logger.exception(f"训练过程失败: {str(e)}")
        return 1

if __name__ == '__main__':
    app.run(main)