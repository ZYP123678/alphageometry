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
import flax
from flax.training import train_state
from flax.training import checkpoints as tcheckpoints
from flax import jax_utils
import optax
from functools import partial
import sentencepiece as spm
import types
from flax.core import FrozenDict
import jax.tree_util as tree_util

def load_flax_ckpt(ckpt_path, vocab_size):
    state = tcheckpoints.restore_checkpoint(ckpt_path, None)
    if not state or 'optimizer' not in state or 'target' not in state['optimizer']:
        raise ValueError("无法在checkpoint中找到参数，请检查文件格式！")
    params = state['optimizer']['target']
    # 截取embedding以适配新词表
    if params['decoder']['embed']['embedding'].shape[0] != vocab_size:
        params['decoder']['embed']['embedding'] = params['decoder']['embed']['embedding'][:vocab_size, :]
    return params

def save_flax_ckpt(save_dir, step, params, model_state, opt_state):
    state_dict = {
        'optimizer': {
            'target': params,
            'state': opt_state
        },
        'state': model_state
    }
    tcheckpoints.save_checkpoint(
        ckpt_dir=save_dir,
        target=state_dict,
        step=step,
        overwrite=True
    )

# 应用补丁
# 添加 KeyArray 类型
jax.random.KeyArray = jax.Array
jax.clear_caches()

def recursive_frozendict(d):
    if isinstance(d, dict):
        return FrozenDict({k: recursive_frozendict(v) for k, v in d.items()})
    return d

class OptimizerDef:
    \"\"\"完整的OptimizerDef实现\"\"\"
    def __init__(self, optax_optimizer: optax.GradientTransformation):
        self.optax_optimizer = optax_optimizer
        
    def create(self, target):
        return Optimizer(self.optax_optimizer, target)
    
    def init_state(self, target):
        return self.optax_optimizer.init(target)

class Optimizer:
    \"\"\"完整的Optimizer实现\"\"\"
    def __init__(self, optax_optimizer: optax.GradientTransformation, target):
        self.optimizer_def = OptimizerDef(optax_optimizer)
        self.target = target
        self.state = optax_optimizer.init(target)
        self.step = 0
    
    def apply_gradient(self, grads, **kwargs):
        updates, new_state = self.optimizer_def.optax_optimizer.update(
            grads, self.state, self.target
        )
        new_target = optax.apply_updates(self.target, updates)
        
        new_optimizer = Optimizer(self.optimizer_def.optax_optimizer, new_target)
        new_optimizer.state = new_state
        new_optimizer.step = self.step + 1
        return new_optimizer

def Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    \"\"\"返回OptimizerDef实例\"\"\"
    if weight_decay > 0:
        return OptimizerDef(optax.adamw(
            learning_rate=learning_rate, 
            b1=beta1, 
            b2=beta2, 
            eps=eps,
            weight_decay=weight_decay
        ))
    else:
        return OptimizerDef(optax.adam(
            learning_rate=learning_rate, 
            b1=beta1, 
            b2=beta2, 
            eps=eps
        ))

# 创建 flax.optim 模块
flax.optim = types.ModuleType('optim')
flax.optim.Optimizer = Optimizer
flax.optim.OptimizerDef = OptimizerDef
flax.optim.GradientTransformation = optax.GradientTransformation
flax.optim.Adam = Adam
flax.optim.AdamW = optax.adamw
flax.optim.Momentum = optax.sgd
flax.optim.RMSProp = optax.rmsprop
sys.modules['flax.optim'] = flax.optim

# 修复导入
sys.modules['flax.optim'] = flax.optim

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
flags.DEFINE_string('train_data', f'/kaggle/working/ag4mtest/butterfly_dpo_data.jsonl', '训练数据路径')
flags.DEFINE_string('output_dir', f'{DPO_DIR}/checkpoints', '输出目录')
flags.DEFINE_float('beta', 0.1, 'DPO温度参数')
flags.DEFINE_float('learning_rate', 1e-3, '学习率')
flags.DEFINE_integer('batch_size', 4, '批大小')
flags.DEFINE_integer('max_seq_length', 512, '最大序列长度')
flags.DEFINE_integer('num_epochs', 30, '训练轮数')
flags.DEFINE_integer('steps_per_epoch', 100, '每轮步数')
flags.DEFINE_integer('save_steps', 100, '保存间隔步数')
flags.DEFINE_bool('use_amp', True, '启用混合精度训练')


# 定义TransformerTaskConfig类
@gin.configurable
class TransformerTaskConfig:
    \"\"\"Transformer任务配置\"\"\"
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
    \"\"\"自定义训练状态结构\"\"\"
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
        learning_rate: float = 1e-3,
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
        self.initialized_state = False
        self.model_state = {}
        
        # 创建训练任务
        from transformer import inference_utils
        self.task = inference_utils.training_loop.Trainer(
            get_training_dataset_iterator=lambda: None,
            get_test_dataset_iterator=None,
            pretty_print_input_function=None,
            process_summaries_function=inference_utils.models.process_summaries_function(self.vocab),
            load_dir=model_path,
            workdir='',
            replicate_mode=False
        ).create_training_task('train', self.model, jax.random.PRNGKey(0), {})
        
        # 配置训练参数
        self.beta = beta
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.global_step = 0
        self.optimizer = self._create_optimizer(learning_rate)
        self.use_amp = use_amp
        self.learning_rate = learning_rate
            
    def _configure_logging(self):
        \"\"\"配置日志输出级别\"\"\"
        # 设置基本日志级别为WARNING
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger('jax').setLevel(logging.WARNING)
        logging.getLogger('flax').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.WARNING)
        
        # 配置控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
        
        # 设置环境变量
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置为2以减少TensorFlow输出
        os.environ['JAX_ENABLE_X64'] = '0'
        os.environ['JAX_DISABLE_JIT'] = '0'
        
        # 禁用JAX的调试输出
        jax.config.update('jax_debug_nans', False)
        jax.config.update('jax_log_compiles', False)
    
    def _create_sentencepiece_vocab(self, model_path: str):
        \"\"\"创建SentencePiece词汇表\"\"\"
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
        \"\"\"初始化Gin配置\"\"\"
        # 配置Gin搜索路径
        config_dir = f'{MELIAD_PATH}/transformer/configs'
        logger.info(f"配置文件目录: {config_dir}")
        
        if os.path.exists(config_dir):
            logger.info(f"配置文件目录内容: {os.listdir(config_dir)}")
        
        # 添加 meliad 到 Python 路径
        if MELIAD_PATH not in sys.path:
            sys.path.append(MELIAD_PATH)
        if f"{MELIAD_PATH}/transformer" not in sys.path:
            sys.path.append(f"{MELIAD_PATH}/transformer")
        
        # 导入必要的模块
        from transformer import nn_components
        from transformer import models
        from transformer import transformer_layer
        from transformer import attention
        from transformer import transformer_base
        
        # 打印模块内容以检查可用的组件
        logger.info("nn_components 模块内容:")
        for name in dir(nn_components):
            if not name.startswith('_'):
                logger.info(f"  - {name}")
        
        # 创建 transformer 模块
        transformer = types.ModuleType('transformer')
        transformer.nn_components = nn_components
        transformer.models = models
        transformer.transformer_layer = transformer_layer
        transformer.attention = attention
        transformer.transformer_base = transformer_base
        
        # 注册模块
        gin.configurable(transformer)
        logger.info("注册模块: transformer")
        gin.configurable(transformer.nn_components)
        logger.info("注册模块: transformer.nn_components")
        
        # 注册可配置项（只注册实际存在的组件）
        available_components = {
            'LayerNorm': getattr(nn_components, 'LayerNorm', None),
            'MLP': getattr(nn_components, 'MLP', None),
            'dropout_multiplier_mask': getattr(nn_components, 'dropout_multiplier_mask', None),
            'get_activation_function': getattr(nn_components, 'get_activation_function', None),
            'safe_softmax': getattr(nn_components, 'safe_softmax', None),
            'scalar_initializer': getattr(nn_components, 'scalar_initializer', None),
            'soft_abs': getattr(nn_components, 'soft_abs', None),
            'swish': getattr(nn_components, 'swish', None),
            'tiled_dropout': getattr(nn_components, 'tiled_dropout', None)
        }
        
        for name, component in available_components.items():
            if component is not None:
                logger.info(f"注册组件: {name}")
                gin.configurable(component)
            else:
                logger.warning(f"组件不存在: {name}")
        
        # 注册其他模块中的组件
        try:
            from transformer.transformer_layer import TransformerLayer
            gin.configurable(TransformerLayer)
            logger.info("注册组件: TransformerLayer")
        except ImportError:
            logger.warning("无法导入 TransformerLayer")
        
        
        try:
            from transformer.transformer_base import Config
            gin.configurable(Config)
            logger.info("注册组件: Config")
        except ImportError:
            logger.warning("无法导入 Config")
        
        gin.add_config_file_search_path(config_dir)
        
        # 配置要加载的文件
        config_files = [
            'base_htrans.gin',
            'size/medium_150M.gin',
            'options/positions_t5.gin',
            'trainer_configuration.gin'
        ]
        
        # 检查文件是否存在
        existing_files = []
        for f in config_files:
            full_path = os.path.join(config_dir, f)
            if os.path.exists(full_path):
                existing_files.append(full_path)
                logger.info(f"找到配置文件: {full_path}")

            else:
                logger.warning(f"配置文件不存在: {full_path}")
                # 检查父目录是否存在
                parent_dir = os.path.dirname(full_path)
                if os.path.exists(parent_dir):
                    logger.info(f"父目录 {parent_dir} 存在，内容: {os.listdir(parent_dir)}")
                else:
                    logger.warning(f"父目录 {parent_dir} 不存在")
        
        # 添加用户指定的文件
        if FLAGS.gin_file:
            for f in FLAGS.gin_file:
                full_path = os.path.join(config_dir, f)
                if os.path.exists(full_path):
                    existing_files.append(full_path)
                    logger.info(f"找到用户指定配置文件: {full_path}")
                else:
                    logger.warning(f"用户指定配置文件不存在: {full_path}")
        
        if not existing_files:
            raise ValueError("没有找到任何有效的配置文件")
        
        # 记录配置
        logger.info(f"加载Gin配置文件: {', '.join(existing_files)}")
        
        # 修改 gin 配置以适配新版本
        # 创建一个 lambda 函数作为随机数生成器
        def make_rng():
            return jax.random.PRNGKey(0)
        
        # 注册随机数生成器
        gin.bind_parameter('transformer.nn_components.tiled_dropout.rng_function', make_rng)
        
        # 解析配置
        try:
            # 使用绝对路径解析配置
            gin.parse_config_files_and_bindings(
                existing_files, 
                FLAGS.gin_param,
                skip_unknown=True  # 跳过未知的配置项
            )
        except Exception as e:
            logger.error(f"Gin配置错误: {str(e)}")
            raise
    
    def _create_task_config(self) -> TransformerTaskConfig:
        \"\"\"创建任务配置\"\"\"
        return TransformerTaskConfig(
            vocab_size=self.vocab.vocab_size,
            sequence_length=512,
            batch_size=FLAGS.batch_size,
            max_sequence_length=512
        )
    
    def _init_model(self):
        \"\"\"初始化模型\"\"\"
        from transformer.models import DecoderOnlyLanguageModel
        
        # 降低初始化期间的日志级别
        orig_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        try:
            actual_vocab_size = self.vocab.vocab_size
            original_vocab_size = self.task_config.vocab_size
            self.task_config.vocab_size = actual_vocab_size
            
            print(f"修复词汇表大小: {original_vocab_size} → {actual_vocab_size}")
            model = DecoderOnlyLanguageModel(
                mode="train", 
                task_config=self.task_config
            )

            self.task_config.vocab_size = original_vocab_size
            logger.info("模型初始化成功")
            print(f"  模型模式: {getattr(model, 'mode', 'unknown')}")
            
            logger.info("模型初始化成功（延迟参数初始化）")
            return model
            
        finally:
            logging.getLogger().setLevel(orig_level)
            
    
    def _create_optimizer(self, learning_rate: float) -> optax.GradientTransformation:
        \"\"\"创建优化器\"\"\"
        print(f" 创建优化器，学习率: {learning_rate}")
    
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=learning_rate)
        )
        
        
        return optimizer

    
    def _dpo_loss(self, params: Dict[str, Any], chosen_data: Tuple[jnp.ndarray, jnp.ndarray],
             rejected_data: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        \"\"\"计算DPO损失 - 使用jax.debug.print进行调试\"\"\"
        chosen_inputs, chosen_targets, chosen_loss_mask = chosen_data
        rejected_inputs, rejected_targets, rejected_loss_mask = rejected_data
    
        # 创建临时训练状态
        temp_tstate = self.tstate.replace(params=params)
        
        try:
            # 计算当前策略的对数概率
            policy_chosen_logps = self._compute_log_probs_pure(temp_tstate, chosen_inputs, chosen_targets)
            policy_rejected_logps = self._compute_log_probs_pure(temp_tstate, rejected_inputs, rejected_targets)
            
            if hasattr(self, 'reference_params') and self.reference_params is not None:
                ref_tstate = self.tstate.replace(params=self.reference_params)
                reference_chosen_logps = self._compute_log_probs_pure(ref_tstate, chosen_inputs, chosen_targets)
                reference_rejected_logps = self._compute_log_probs_pure(ref_tstate, rejected_inputs, rejected_targets)
            else:
                # 🔧 如果真的没有参考策略，使用带噪声的版本
                print("警告：没有参考策略，使用噪声版本")
                noise_key1, noise_key2 = jax.random.split(jax.random.PRNGKey(42))
                reference_chosen_logps = policy_chosen_logps + jax.random.normal(noise_key1, policy_chosen_logps.shape) * 0.1
                reference_rejected_logps = policy_rejected_logps + jax.random.normal(noise_key2, policy_rejected_logps.shape) * 0.1
                
            
            # 使用传入的掩码计算平均对数概率
            # 当前策略的平均对数概率
            policy_chosen_sum = jnp.sum(policy_chosen_logps * chosen_loss_mask, axis=-1)
            policy_chosen_lengths = jnp.sum(chosen_loss_mask, axis=-1) + 1e-8
            policy_chosen_mean = policy_chosen_sum / policy_chosen_lengths
            
            policy_rejected_sum = jnp.sum(policy_rejected_logps * rejected_loss_mask, axis=-1)
            policy_rejected_lengths = jnp.sum(rejected_loss_mask, axis=-1) + 1e-8
            policy_rejected_mean = policy_rejected_sum / policy_rejected_lengths
            
            # 参考策略的平均对数概率
            ref_chosen_sum = jnp.sum(reference_chosen_logps * chosen_loss_mask, axis=-1)
            ref_chosen_mean = ref_chosen_sum / policy_chosen_lengths
            
            ref_rejected_sum = jnp.sum(reference_rejected_logps * rejected_loss_mask, axis=-1)
            ref_rejected_mean = ref_rejected_sum / policy_rejected_lengths
            
            # 计算策略优势（标准DPO公式）
            policy_logratios = policy_chosen_mean - policy_rejected_mean
            reference_logratios = ref_chosen_mean - ref_rejected_mean
            logits = policy_logratios - reference_logratios
                    
            embedding_params = params['decoder']['embed']['embedding']
            regularization = 1e-8 * jnp.sum(embedding_params ** 2)  # L2正则化
            logits = logits + regularization
            
            # 应用温度参数
            logits = self.beta * logits
            logits = jnp.clip(logits, -100, 100)
            
            # 标准DPO损失（带标签平滑）
            label_smoothing = 0.1  # 可配置的标签平滑参数
            losses = (
                -jax.nn.log_sigmoid(logits) * (1 - label_smoothing)
                - jax.nn.log_sigmoid(-logits) * label_smoothing
            )
            
            losses = jnp.clip(losses, 0.0, 100)
            
            # 调试信息
            if self.global_step % 10 == 0:
                jax.debug.print("DPO调试: policy_chosen={:.4f}, policy_rejected={:.4f}, ref_chosen={:.4f}, ref_rejected={:.4f}, logits={:.4f}, loss={:.4f}", 
                               jnp.mean(policy_chosen_mean), jnp.mean(policy_rejected_mean),
                               jnp.mean(ref_chosen_mean), jnp.mean(ref_rejected_mean),
                               jnp.mean(logits), jnp.mean(losses))
                # 检查参数差异
                if hasattr(self, 'reference_params'):
                    param_diff = jnp.mean(jnp.abs(params['decoder']['embed']['embedding'] - 
                                                self.reference_params['decoder']['embed']['embedding']))
                    jax.debug.print("参数差异: {:.6f}", param_diff)
                    
                    # 检查logits的原始值（未裁剪）
                    raw_logits = policy_logratios - reference_logratios
                    jax.debug.print("原始logits: {:.4f}, 裁剪后logits: {:.4f}", 
                                   jnp.mean(raw_logits), jnp.mean(logits))
            
            return jnp.mean(losses)
            
        except Exception as e:
            print(f"DPO损失计算失败: {e}")
            return jnp.array(1.0, dtype=jnp.float32)
            
                
    def _compute_log_probs_pure(self, tstate, inputs: dict, targets: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len = targets.shape
    
        try:
            complete_inputs = dict(inputs)
            if 'epoch' not in complete_inputs:
                complete_inputs['epoch'] = jnp.ones((batch_size,), dtype=jnp.int32)
            if 'loss_mask' not in complete_inputs:
                complete_inputs['loss_mask'] = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
    
            # 构建variables字典
            params = tstate.params
            if not isinstance(params, FrozenDict):
                params = FrozenDict(params)
            
            variables = {'params': params}
            
            state_key = f"{batch_size}x{seq_len}"
            
            if not hasattr(self, 'model_states'):
                self.model_states = {}
            
            # 动态初始化状态
            if state_key not in self.model_states:
                if self.global_step % 50 == 0:
                    print(f"为 {state_key} 初始化模型状态...")
                
                original_batch_size = self.task_config.batch_size
                original_seq_length = self.task_config.sequence_length
                
                try:
                    self.task_config.batch_size = batch_size
                    self.task_config.sequence_length = seq_len
                    
                    init_rng = jax.random.PRNGKey(42)
                    init_rngs = {'params': init_rng, 'dropout': init_rng}
                    init_variables = self.model.init(init_rngs, complete_inputs)
                    
                    # 保存
                    self.model_states[state_key] = {k: v for k, v in init_variables.items() if k != 'params'}
                    
                finally:
                    self.task_config.batch_size = original_batch_size
                    self.task_config.sequence_length = original_seq_length
            
            # 添加状态到variables
            if state_key in self.model_states:
                variables.update(self.model_states[state_key])
            
            # 允许状态更新
            result = self.model.apply(
                variables,
                complete_inputs,
                method=self.model.predict_logits,
                rngs={'dropout': jax.random.PRNGKey(0)},
                mutable=['state']  # 允许更新state集合
            )
            
            # 处理返回结果
            if isinstance(result, tuple):
                logits, updated_variables = result
                # 更新状态
                if 'state' in updated_variables:
                    self.model_states[state_key]['state'] = updated_variables['state']
            else:
                logits = result
    
            # 计算log_softmax
            log_probs = jax.nn.log_softmax(logits, axis=-1)
    
            # 获取目标token的log_probs
            batch_indices = jnp.arange(batch_size)[:, None]
            seq_indices = jnp.arange(seq_len)[None, :]
            valid_targets = jnp.clip(targets, 0, logits.shape[-1] - 1)
            target_log_probs = log_probs[batch_indices, seq_indices, valid_targets]
            return target_log_probs
    
        except Exception as e:
            if self.global_step % 50 == 0:
                print(f"计算失败: {e}")
            return jnp.full((batch_size, seq_len), -3.0, dtype=jnp.float32)
    
    def _update_step(self, chosen_data: Tuple[jnp.ndarray, jnp.ndarray],
                rejected_data: Tuple[jnp.ndarray, jnp.ndarray]):
        \"\"\"执行单个更新步骤\"\"\"

        from flax.core import FrozenDict

        def set_in_frozendict(fd, keys, value):
            \"\"\"递归地在FrozenDict中设置值，返回新的FrozenDict。\"\"\"
            if len(keys) == 1:
                d = dict(fd)
                d[keys[0]] = value
                return FrozenDict(d)
            else:
                d = dict(fd)
                d[keys[0]] = set_in_frozendict(d[keys[0]], keys[1:], value)
                return FrozenDict(d)

        def loss_fn(params):
            return self._dpo_loss(params, chosen_data, rejected_data)

        try:

            # 计算梯度
            loss, grads = jax.value_and_grad(loss_fn)(self.tstate.params)

            # 检查梯度
            grad_norm = jax.tree_util.tree_reduce(
                lambda x, y: x + jnp.sum(jnp.square(y)),
                grads,
                initializer=0.0
            )
            grad_norm = jnp.sqrt(grad_norm)

            if self.global_step % 5 == 0:  # 更频繁的检查
                # 检查embedding梯度
                if (
                    'decoder' in grads
                    and 'embed' in grads['decoder']
                    and 'embedding' in grads['decoder']['embed']
                ):
                    embed_grad = grads['decoder']['embed']['embedding']
                    embed_grad_norm = jnp.sqrt(jnp.sum(jnp.square(embed_grad)))
                    embed_grad_max = jnp.max(jnp.abs(embed_grad))
                    embed_grad_mean = jnp.mean(jnp.abs(embed_grad))
                    
                    print(f"Step {self.global_step}: embedding梯度 - norm={embed_grad_norm:.6f}, max={embed_grad_max:.6f}, mean={embed_grad_mean:.6f}")
                    
                    if jax.device_get(embed_grad_norm) < 1e-8:
                        print("  ⚠️ 警告：Embedding梯度为零！")
                        
                        # 检查其他层的梯度
                        total_grad_norm = 0
                        for key, value in grads.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    if isinstance(subvalue, dict):
                                        for subsubkey, subsubvalue in subvalue.items():
                                            if hasattr(subsubvalue, 'shape'):
                                                sub_grad_norm = jnp.sqrt(jnp.sum(jnp.square(subsubvalue)))
                                                total_grad_norm += sub_grad_norm
                                                if sub_grad_norm > 1e-6:
                                                    print(f"    {key}.{subkey}.{subsubkey} 梯度: {sub_grad_norm:.6f}")
                            elif hasattr(value, 'shape'):
                                sub_grad_norm = jnp.sqrt(jnp.sum(jnp.square(value)))
                                total_grad_norm += sub_grad_norm
                                if sub_grad_norm > 1e-6:
                                    print(f"    {key} 梯度: {sub_grad_norm:.6f}")
                        
                        print(f"    总梯度范数: {total_grad_norm:.6f}")
                
                # 检查当前参数与参考参数的差异
                if hasattr(self, 'reference_params'):
                    current_params = self.tstate.params
                    ref_params = self.reference_params
                    
                    if 'decoder' in current_params and 'embed' in current_params['decoder']:
                        current_embed = current_params['decoder']['embed']['embedding']
                        ref_embed = ref_params['decoder']['embed']['embedding']
                        
                        # 计算参数差异
                        param_diff = jnp.mean(jnp.abs(current_embed - ref_embed))
                        param_diff_max = jnp.max(jnp.abs(current_embed - ref_embed))
                        
                        print(f"Step {self.global_step}: 参数差异 - mean={param_diff:.6f}, max={param_diff_max:.6f}")
                        
                        # 如果参数差异很小，可能是训练有问题
                        if jax.device_get(param_diff) < 1e-6:
                            print("  ⚠️ 警告：参数几乎没有变化！")
                        
                        # 检查学习率
                        if hasattr(self.tstate, 'opt_state'):
                            print(f"Step {self.global_step}: 学习率状态存在")

            # 梯度裁剪 - 防止梯度爆炸
            max_grad_norm = 1.0
            if grad_norm > max_grad_norm:
                grads = jax.tree_util.tree_map(
                    lambda g: g * max_grad_norm / grad_norm, grads
                )
                print(f"Step {self.global_step}: 梯度裁剪 - 从 {grad_norm:.4f} 到 {max_grad_norm}")

            # 更新参数（即使梯度很小也更新）
            updates, new_opt_state = self.tstate.tx.update(
                grads, self.tstate.opt_state, self.tstate.params
            )
            new_params = optax.apply_updates(self.tstate.params, updates)
            if not isinstance(new_params, FrozenDict):
                new_params = FrozenDict(new_params)
            # 更新训练状态
            self.tstate = self.tstate.replace(
                step=self.tstate.step + 1,
                params=new_params,
                opt_state=new_opt_state
            )

            self.global_step += 1
            return jax.device_get(loss)

        except Exception as e:
            logger.error(f"更新步骤失败: {str(e)}")
            import traceback
            logger.error(f"完整错误堆栈: {traceback.format_exc()}")
            raise
    

    def _init_training_state(self, model_path: str) -> Any:
        \"\"\"初始化训练状态 - 修复所有配置一致性\"\"\"
        def recursive_frozendict(d):
            if isinstance(d, dict):
                return FrozenDict({k: recursive_frozendict(v) for k, v in d.items()})
            elif isinstance(d, (list, tuple)):
                return type(d)(recursive_frozendict(item) for item in d)
            else:
                return d
                
        try:
            print(" 延迟初始化模型参数...")
            
            #  临时修改task_config以匹配实际词汇表大小
            original_batch_size = self.task_config.batch_size
            original_seq_length = self.task_config.sequence_length
            original_max_seq_length = self.task_config.max_sequence_length
            original_vocab_size = self.task_config.vocab_size
            
            # 使用实际的词汇表大小
            actual_vocab_size = self.vocab.vocab_size
            
            # 使用更小的配置，并确保一致性
            self.task_config.batch_size = 1
            self.task_config.sequence_length = 128  # 使用16
            self.task_config.max_sequence_length = 128  # 使用16
            self.task_config.vocab_size = actual_vocab_size
            
            try:
                # 重新初始化模型以使用新的配置
                from transformer.models import DecoderOnlyLanguageModel
                temp_model = DecoderOnlyLanguageModel(
                    mode="train", 
                    task_config=self.task_config
                )
                
                # 初始化模型参数
                rng = jax.random.PRNGKey(0)
                params_rng, dropout_rng = jax.random.split(rng)
                
                # 获取假输入
                fake_input = temp_model.get_fake_input()  # 使用temp_model而不是self.model
                print(f"  fake_input keys: {fake_input.keys()}")
  
                # 初始化参数
                print("  正在初始化模型参数...")
                variables = temp_model.init(
                    {'params': params_rng, 'dropout': dropout_rng},
                    fake_input
                )
                print(f" 参数初始化成功，keys: {variables.keys()}")
                params = variables.get('params', {})
                params = recursive_frozendict(params)

                model_state = {k: v for k, v in variables.items() if k != 'params'}

                # 更新模型引用
                self.model = temp_model
                
            finally:
                # 恢复原始配置
                self.task_config.batch_size = original_batch_size
                self.task_config.sequence_length = original_seq_length
                self.task_config.max_sequence_length = original_max_seq_length
                self.task_config.vocab_size = original_vocab_size
            
            # 加载预训练权重
            if model_path and os.path.exists(model_path):
                print(f"加载预训练权重: {model_path}")
                try:
                    # 用Flax兼容方式加载参考参数
                    reference_params = load_flax_ckpt(model_path, self.vocab.vocab_size)
                    self.reference_params = recursive_frozendict(reference_params)

                    # 生成current_params（可加扰动）
                    def add_small_perturbation(params):
                        def perturb_array(x):
                            if hasattr(x, 'shape'):
                                noise = jax.random.normal(jax.random.PRNGKey(42), x.shape) * 1e-5
                                return x + noise
                            return x
                        return jax.tree_util.tree_map(perturb_array, params)
                    current_params = add_small_perturbation(reference_params)
                    current_params = recursive_frozendict(current_params)

                    new_optimizer = self._create_optimizer(FLAGS.learning_rate)

                    # 重新创建训练状态
                    from flax.training import train_state
                    tstate = train_state.TrainState.create(
                        apply_fn=self.model.apply,
                        params=current_params,
                        tx=new_optimizer
                    )
                    print("  成功重新创建训练状态")
                        
                except Exception as e:
                    print(f" 权重加载失败，使用随机初始化: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  检查点文件不存在: {model_path}")
                print("  使用随机初始化参数")
            
            print(" 训练状态初始化完成")
    
            
            print(f"参考策略参数数量: {len(jax.tree_leaves(self.reference_params))}")
            print(f"当前策略参数数量: {len(jax.tree_leaves(tstate.params))}")
            print(" 保存参考策略参数")

            
            def params_equal(params1, params2):
                def compare_arrays(x, y):
                    if hasattr(x, 'shape') and hasattr(y, 'shape'):
                        return jnp.array_equal(x, y)
                    return x == y
                
                return tree_util.tree_reduce(
                    lambda acc, val: acc and val,
                    tree_util.tree_map(compare_arrays, params1, params2),
                    initializer=True
                )
            
            # 使用正确的比较
            print(f"深拷贝后是否相同: {params_equal(tstate.params, self.reference_params)}")
            
            self.initialized_state = False
            self.model_state = model_state
            return tstate
            
        except Exception as e:
            print(f" 训练状态初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def _prepare_batch(self, examples: List[Dict]) -> Tuple[Tuple[Dict, jnp.ndarray, jnp.ndarray], Tuple[Dict, jnp.ndarray, jnp.ndarray]]:
        \"\"\"准备批次数据，并为completion-only损失计算创建掩码\"\"\"
        if not examples:
            raise ValueError("批次数据为空")
                    
        # 初始化容器
        batch_data = {
            'chosen_inputs': [],
            'chosen_targets': [],
            'chosen_loss_mask': [],
            'rejected_inputs': [],
            'rejected_targets': [],
            'rejected_loss_mask': [],
        }
        
        for i, ex in enumerate(examples):
            prompt = ex['prompt']
            chosen_text = ex['chosen']
            rejected_text = ex['rejected']
            
            prompt_tokens = self.vocab.encode(prompt)
            prompt_len = len(prompt_tokens)

            # --- 处理 chosen ---
            chosen_completion_tokens = self.vocab.encode(chosen_text)
            chosen_full_tokens = (prompt_tokens + chosen_completion_tokens)[:self.max_seq_length]
            
            chosen_inputs = np.zeros(self.max_seq_length, dtype=np.int32)
            chosen_inputs[:len(chosen_full_tokens)] = chosen_full_tokens
            
            chosen_targets = np.zeros(self.max_seq_length, dtype=np.int32)
            if len(chosen_full_tokens) > 0:
                chosen_targets[:len(chosen_full_tokens)-1] = chosen_full_tokens[1:]
                chosen_targets[len(chosen_full_tokens)-1] = self.vocab.sp.PieceToId('<eos>')

            # 创建损失掩码 (只在 completion 部分计算损失)
            chosen_loss_mask = np.zeros(self.max_seq_length, dtype=np.float32)
            # 损失从 prompt 的最后一个 token 开始计算，因为它预测的是 completion 的第一个 token
            start_index = prompt_len - 1
            end_index = len(chosen_full_tokens) - 1
            if start_index < end_index:
                chosen_loss_mask[start_index:end_index] = 1.0

            # --- 处理 rejected ---
            rejected_completion_tokens = self.vocab.encode(rejected_text)
            rejected_full_tokens = (prompt_tokens + rejected_completion_tokens)[:self.max_seq_length]

            rejected_inputs = np.zeros(self.max_seq_length, dtype=np.int32)
            rejected_inputs[:len(rejected_full_tokens)] = rejected_full_tokens
            
            rejected_targets = np.zeros(self.max_seq_length, dtype=np.int32)
            if len(rejected_full_tokens) > 0:
                rejected_targets[:len(rejected_full_tokens)-1] = rejected_full_tokens[1:]
                rejected_targets[len(rejected_full_tokens)-1] = self.vocab.sp.PieceToId('<eos>')

            # 为 rejected 创建损失掩码
            rejected_loss_mask = np.zeros(self.max_seq_length, dtype=np.float32)
            end_index_rej = len(rejected_full_tokens) - 1
            if start_index < end_index_rej:
                rejected_loss_mask[start_index:end_index_rej] = 1.0
            
            batch_data['chosen_inputs'].append(chosen_inputs)
            batch_data['chosen_targets'].append(chosen_targets)
            batch_data['chosen_loss_mask'].append(chosen_loss_mask)
            batch_data['rejected_inputs'].append(rejected_inputs)
            batch_data['rejected_targets'].append(rejected_targets)
            batch_data['rejected_loss_mask'].append(rejected_loss_mask)
        
        chosen_inputs = jnp.array(batch_data['chosen_inputs'], dtype=jnp.int32)
        chosen_targets = jnp.array(batch_data['chosen_targets'], dtype=jnp.int32)
        chosen_loss_mask = jnp.array(batch_data['chosen_loss_mask'], dtype=jnp.float32)
        rejected_inputs = jnp.array(batch_data['rejected_inputs'], dtype=jnp.int32)
        rejected_targets = jnp.array(batch_data['rejected_targets'], dtype=jnp.int32)
        rejected_loss_mask = jnp.array(batch_data['rejected_loss_mask'], dtype=jnp.float32)
        
        # 确保批次维度正确
        if len(chosen_inputs.shape) != 2 or len(chosen_targets.shape) != 2:
            raise ValueError(f"批次维度错误: chosen_inputs shape={chosen_inputs.shape}, chosen_targets shape={chosen_targets.shape}")
        
        chosen_inputs_dict = {
            "targets": chosen_inputs,
            "start_of_sequence": jnp.ones((chosen_inputs.shape[0],), dtype=jnp.bool_)
        }
        rejected_inputs_dict = {
            "targets": rejected_inputs,
            "start_of_sequence": jnp.ones((rejected_inputs.shape[0],), dtype=jnp.bool_)
        }
        
        return (chosen_inputs_dict, chosen_targets, chosen_loss_mask), (rejected_inputs_dict, rejected_targets, rejected_loss_mask)
    
    
    def train(self, train_data: List[Dict], output_dir: str = None) -> str:
        \"\"\"训练主循环\"\"\"
        output_dir = output_dir or FLAGS.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置训练日志
        log_file = os.path.join(output_dir, "dpo_training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        
        try:
            for epoch in range(FLAGS.num_epochs):
                random.shuffle(train_data)
                epoch_losses = []
                
                with tqdm(total=min(FLAGS.steps_per_epoch, len(train_data)//FLAGS.batch_size), 
                        desc=f"Epoch {epoch+1}") as pbar:
                    for step in range(FLAGS.steps_per_epoch):
                        try:
                            # 准备批次数据
                            batch_examples = train_data[step*FLAGS.batch_size: (step+1)*FLAGS.batch_size]
                            if not batch_examples:
                                break
                            
                            logger.info(f"Step {step}: 准备批次数据，样本数量: {len(batch_examples)}")
                            chosen_data, rejected_data = self._prepare_batch(batch_examples)
                            
                            # logger.info(f"Step {step}: chosen_data类型: {type(chosen_data)}, 长度: {len(chosen_data)}")
                            # logger.info(f"Step {step}: rejected_data类型: {type(rejected_data)}, 长度: {len(rejected_data)}")
                            # logger.info(f"Step {step}: 执行更新步骤")
                            loss = self._update_step(chosen_data, rejected_data)
                            epoch_losses.append(loss)
                            avg_loss = sum(epoch_losses) / len(epoch_losses)
                            
                            pbar.update(1)
                            pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
                            
                            # 定期保存检查点
                            if step > 0 and step % FLAGS.save_steps == 0:
                                self.save_checkpoint(output_dir)
                                
                        except Exception as e:
                            logger.error(f"Step {step} 失败: {str(e)}")
                            import traceback
                            logger.error(f"完整错误堆栈: {traceback.format_exc()}")
                            raise
                
                logger.info(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")
            
            self.save_checkpoint(output_dir, final=True)
            logger.info(f"训练完成，模型保存在: {output_dir}")
            return output_dir
        except Exception as e:
            logger.exception(f"训练失败: {str(e)}")
            raise
        finally:
            logger.removeHandler(file_handler)

    def save_checkpoint(self, output_dir: str, final: bool = False) -> str:
        save_name = "final_model" if final else f"checkpoint_{self.global_step}"
        save_path = os.path.join(output_dir, save_name)
        os.makedirs(save_path, exist_ok=True)
        save_flax_ckpt(
            save_dir=save_path,
            step=self.global_step,
            params=self.tstate.params,
            model_state=self.model_state,
            opt_state=self.tstate.opt_state
        )
        import json
        metadata = {
            'global_step': self.global_step,
            'vocab_path': os.path.abspath(FLAGS.vocab_path),
            'beta': self.beta
        }
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        logger.info(f"检查点保存至: {save_path}")

        # === 新增：导出参数为 npz 和 pkl ===
        import numpy as np
        import pickle

        def flatten_dict(d, parent_key='', sep='/'):
            """递归展开嵌套dict为扁平dict，便于npz保存"""
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        # 导出为 npz
        try:
            flat_params = flatten_dict(self.tstate.params)
            npz_path = os.path.join(save_path, "params.npz")
            np.savez(npz_path, **flat_params)
            logger.info(f"参数已导出为 npz: {npz_path}")
        except Exception as e:
            logger.warning(f"导出 npz 失败: {e}")

        # 导出为 pkl（保留原始嵌套结构）
        try:
            pkl_path = os.path.join(save_path, "params.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(self.tstate.params, f)
            logger.info(f"参数已导出为 pkl: {pkl_path}")
        except Exception as e:
            logger.warning(f"导出 pkl 失败: {e}")

        return save_path

def load_and_validate_data(file_path: str) -> List[Dict]:
    \"\"\"加载和验证训练数据\"\"\"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"训练数据文件不存在: {file_path}")
    
    valid_data = []
    score_issues = 0
    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                ex = json.loads(line)
                if 'prompt' in ex and 'chosen' in ex and 'rejected' in ex:
                    
                    # 打印数据样本
                    # logger.info(f"加载数据样本:")
                    # logger.info(f"  prompt: {ex['prompt'][:100]}...")
                    # logger.info(f"  chosen: {ex['chosen'][:100]}...")
                    # logger.info(f"  rejected: {ex['rejected'][:100]}...")
                    # logger.info(f"  chosen_score: {ex['chosen_score']}")
                    # logger.info(f"  rejected_score: {ex['rejected_score']}")
                    
                    valid_data.append(ex)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"处理数据时出错: {str(e)}")
                continue
    
    logger.info(f"数据加载完成: 有效样本={len(valid_data)}")
    return valid_data

def main(argv):

    \"\"\"主函数入口\"\"\"
    try:
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        logger.info(f"输出目录: {FLAGS.output_dir}")
        
        trainer = DPOTrainer(
            model_path=FLAGS.ckpt_path,
            vocab_path=FLAGS.vocab_path,
            beta=FLAGS.beta,
            learning_rate=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size,
            max_seq_length=FLAGS.max_seq_length,
            use_amp=FLAGS.use_amp
        )
        
        logger.info(f"开始加载训练数据: {FLAGS.train_data}")
        train_data = load_and_validate_data(FLAGS.train_data)
        
        if not train_data:
            raise ValueError("没有有效的训练数据")
            
        logger.info(f"成功加载 {len(train_data)} 条训练数据")
        
        logger.info("开始DPO训练")
        output_dir = trainer.train(train_data)
        
        logger.info(f"训练完成，模型保存在: {output_dir}")
        return 0
    except Exception as e:
        logger.exception(f"训练过程失败: {str(e)}")
        return 1

if __name__ == '__main__':
    app.run(main)