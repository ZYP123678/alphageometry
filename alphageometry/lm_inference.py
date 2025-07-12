# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Wrapper for language modeling inference implemented in Meliad."""
from typing import Any, Dict

import jax
import models  # pylint: disable=unused-import
import t5.data
from transformer import inference_utils
from absl import logging
from flax.core import FrozenDict
import numpy

np = jax.numpy


Trainer = inference_utils.Trainer

MetricsOutput = Dict[str, Any]  # Metrics output by model.


parse_gin_configuration = inference_utils.parse_gin_configuration


class LanguageModelInference:
  """Meliad wrapper for LM inference."""

  def __init__(self, vocab_path: str, load_dir_or_params, mode='beam_search', from_params=False):
    self.vocab = t5.data.SentencePieceVocabulary(vocab_path)

    if from_params:
      # 先正常初始化
      logging.info('================== 加载 PKL 格式参数 (from_params=True) ==================')

      # 1. 初始化一个不加载任何权重的 "空壳" Trainer，以获取模型结构和干净的状态
      def null_iter_fn() -> None:
        return None

      process_summaries_f = inference_utils.models.process_summaries_function(
          self.vocab
      )

      trainer = inference_utils.training_loop.Trainer(
          get_training_dataset_iterator=null_iter_fn,
          get_test_dataset_iterator=None,
          pretty_print_input_function=None,
          process_summaries_function=process_summaries_f,
          load_dir=None,
          workdir='',
          replicate_mode=False,
      )
      self.trainer = trainer

      (tstate, _, imodel, prngs) = trainer.initialize_model()
      self.imodel = imodel
      self.batch_size = imodel.task_config.batch_size
      self.n = imodel.num_heads
      self.h = imodel.head_size
      writers = {}
      self.task = trainer.create_training_task(mode, imodel, prngs, writers)
      inference_utils.training_loop.clear_interstep_callbacks()
      inference_utils.training_loop.register_interstep_callbacks()
      
      # 2. 定义辅助函数，用于处理嵌套字典和参数填充
      def pad_embedding(embedding, target_shape):
          """根据目标形状，填充或截断给定的 embedding 数组。"""
          # 使用原始 numpy 进行形状检查和操作
          if not isinstance(embedding, numpy.ndarray):
              embedding = numpy.array(embedding)
          emb_shape = embedding.shape
          
          logging.info(f"Embedding shape: {emb_shape}, target shape: {target_shape}")
          if emb_shape == target_shape:
              return embedding
          elif len(emb_shape) != 2 or len(target_shape) != 2:
              logging.error(f"Embedding 维度不匹配，无法填充。源: {emb_shape}, 目标: {target_shape}")
              raise ValueError("Embedding 必须是二维的才能填充。")
          elif emb_shape[0] < target_shape[0]:
              # 补齐到目标大小
              pad_width = ((0, target_shape[0] - emb_shape[0]), (0, 0))
              padded = numpy.pad(embedding, pad_width, mode='constant', constant_values=0)
              logging.info(f"Padded embedding from {emb_shape} to {padded.shape}")
              return padded
          else:
              # 截取到目标大小
              truncated = embedding[:target_shape[0], :]
              logging.info(f"Truncated embedding from {emb_shape} to {truncated.shape}")
              return truncated

      def recursive_to_dict(d):
          """递归地将 Flax FrozenDict 转换为 Python 普通字典。"""
          if isinstance(d, (dict, FrozenDict)):
              return {k: recursive_to_dict(v) for k, v in d.items()}
          return d

      def recursive_to_frozendict(d):
          """递归地将 Python 普通字典转换为 Flax FrozenDict。"""
          if isinstance(d, dict):
              return FrozenDict({k: recursive_to_frozendict(v) for k, v in d.items()})
          return d

      # 3. 将传入的 DPO 参数转换为可变字典
      params_dict = recursive_to_dict(load_dir_or_params)

      # 4. 获取目标 embedding 形状并进行填充
      try:
          target_shape = tstate.params['decoder']['embed']['embedding'].shape
          logging.info(f"从 tstate.params 中获取目标 embedding 形状: {target_shape}")
          
          # 检查并填充 embedding
          if 'decoder' in params_dict and 'embed' in params_dict['decoder'] and 'embedding' in params_dict['decoder']['embed']:
              params_dict['decoder']['embed']['embedding'] = pad_embedding(
                  params_dict['decoder']['embed']['embedding'], target_shape)
          else:
              logging.warning("在加载的参数中未找到 'decoder.embed.embedding' 路径，跳过填充。")

      except KeyError:
          logging.error("无法在初始化的 tstate.params 中找到 'decoder.embed.embedding'，无法进行形状适配。")
          raise

      # 5. 将处理过的参数更新回 TrainState
      #    我们只替换 `params`，保持 `opt_state` 等其他部分是干净的、初始化的状态
      self.tstate = tstate.replace(params=recursive_to_frozendict(params_dict))
      logging.info("已成功将 DPO 参数注入到新的 TrainState 中。")

    # ====== 原有的 checkpoint 路径加载逻辑 ======
    def null_iter_fn() -> None:
      return None

    process_summaries_f = inference_utils.models.process_summaries_function(
        self.vocab
    )

    trainer = inference_utils.training_loop.Trainer(
        get_training_dataset_iterator=null_iter_fn,
        get_test_dataset_iterator=None,
        pretty_print_input_function=None,
        process_summaries_function=process_summaries_f,
        load_dir=load_dir_or_params,
        workdir='',  # Don't log or save checkpoints.
        replicate_mode=False,
    )
    self.trainer = trainer

    (tstate, _, imodel, prngs) = trainer.initialize_model()
    self.imodel = imodel
    self.batch_size = imodel.task_config.batch_size
    self.n = imodel.num_heads
    self.h = imodel.head_size
    writers = {}
    self.task = trainer.create_training_task(mode, imodel, prngs, writers)
    inference_utils.training_loop.clear_interstep_callbacks()
    inference_utils.training_loop.register_interstep_callbacks()
    self.tstate = tstate

    eos = [0] * 1024
    for idx in self.encode_list(['.', ';']):
      eos[idx] = 1
    self.eos = np.array(eos, dtype=np.bfloat16)
    self.mask = jax.numpy.ones([1024], dtype=np.bfloat16)

  def decode(self, ids: list[int]) -> str:
    return self.vocab.decode(ids)

  def decode_list(self, tokens: list[int]) -> list[str]:
    return [self.decode([tok]) for tok in tokens]

  def encode(self, inputs_str: str) -> list[int]:
    return self.vocab.encode(inputs_str)

  def encode_list(self, inputs_strs: list[str]) -> list[int]:
    result = [self.vocab.encode(x) for x in inputs_strs]
    assert all([len(x) == 1 for x in result]), [
        self.decode(x) for x in result if len(x) != 1
    ]
    return [x[0] for x in result]

  def call(
      self,
      inputs: np.ndarray,
      dstate: tuple[dict[str, np.ndarray], ...] = None,
      eos: np.ndarray = None,
      mask: np.ndarray = None,
  ) -> MetricsOutput:
    """Call the meliad model."""
    batch_size, length = inputs.shape
    inputs = jax.numpy.pad(inputs, [(0, 0), (0, 1024 - length)])

    if eos is None:
      eos = self.eos
    if mask is None:
      mask = self.mask

    x = {'targets': inputs, 'length': length, 'eos': eos, 'mask': mask}

    if dstate is not None:
      x['start_of_sequence'] = jax.numpy.array([False] * batch_size)
    else:
      dstate = tuple(
          [{  # this dummy value will never be used.
              'current_index': np.array([0] * batch_size, dtype=np.int32),
              'keys': np.zeros(
                  (batch_size, 2048, self.n, self.h), dtype=np.bfloat16
              ),
              'values': np.zeros(
                  (batch_size, 2048, self.n, self.h), dtype=np.bfloat16
              ),
              'recurrent_kvq': None,
              'relative_position_bias': np.zeros(
                  (batch_size, self.n, 1, 1024), dtype=np.bfloat16
              ),
          }]
          * 12
      )
      x['start_of_sequence'] = jax.numpy.array([True] * batch_size)

    x['dstate'] = dstate
    _, metrics_np = self.task.run_step(self.tstate, x, 0)
    return metrics_np

  def beam_decode(
      self,
      inputs: str,
      eos_tokens: np.ndarray = None,
      mask_tokens: np.ndarray = None,
      dstate: dict[str, np.ndarray] = None,
  ) -> MetricsOutput:
    """Beam search."""
    inputs = jax.numpy.array([self.vocab.encode(inputs)] * self.batch_size)

    eos = self.eos
    if eos_tokens is not None:
      eos_ids = self.encode_list(eos_tokens)
      eos = np.array(
          [1 if idx in eos_ids else 0 for idx in range(1024)], dtype=np.bfloat16
      ).reshape((1, 1, 1024))

    mask = self.mask
    if mask_tokens is not None:
      mask_ids = self.encode_list(mask_tokens)
      mask = np.array(
          [0 if idx in mask_ids else 1 for idx in range(1024)],
          dtype=np.bfloat16,
      ).reshape((1, 1, 1024))

    metrics_np = self.call(inputs, dstate=dstate, eos=eos, mask=mask)

    finished_seqs = metrics_np['finished_seqs']
    finished_scores = metrics_np['finished_scores']

    seqs = []
    scores = []
    for seq, score in zip(finished_seqs, finished_scores):
      seq = self.decode(seq[1:])
      seqs.append(seq)
      scores.append(score)

    return {
        'finished_seqs': finished_seqs,
        'finished_scores': finished_scores,
        'seqs_str': seqs,
        'scores': scores,
        'dstate': metrics_np['dstate'],
    }
