# DPO强化学习微调

这个目录包含了用于对AlphaGeometry模型进行DPO（Direct Preference Optimization）强化学习微调的代码。

## 环境要求

- Python 3.8+
- TensorFlow 2.13.0
- JAX 0.4.6
- Flax 0.5.3
- 其他依赖见 `requirements.in`

## 目录结构

```
dpo/
├── dpo_trainer.py    # DPO训练器实现
├── run_dpo.py        # 运行训练的脚本
└── README.md         # 本文档
```

## 使用方法

1. 准备训练数据

训练数据应该是JSONL格式，每行包含以下字段：
```json
{
    "prompt": "问题描述",
    "chosen": "更好的解答",
    "rejected": "较差的解答",
    "chosen_score": 1.0,    // 可选
    "rejected_score": 0.0   // 可选
}
```

2. 设置环境变量

```bash
export AG4MDIR=/path/to/ag4masses
export AGLIB=/path/to/aglib
```

3. 运行训练

```bash
python run_dpo.py \
    --train_data=your_data.jsonl \
    --output_dir=dpo_checkpoints \
    --batch_size=4 \
    --learning_rate=5e-5 \
    --beta=0.1
```

## 主要参数说明

- `train_data`: 训练数据文件路径
- `output_dir`: 输出目录，用于保存检查点
- `batch_size`: 批处理大小
- `learning_rate`: 学习率
- `beta`: DPO温度参数
- `max_seq_length`: 最大序列长度
- `num_epochs`: 训练轮数
- `steps_per_epoch`: 每轮步数
- `save_steps`: 保存检查点的间隔步数
- `use_amp`: 是否使用混合精度训练

## 注意事项

1. 确保模型检查点和词汇表文件路径正确
2. 训练数据应该是JSONL格式
3. 建议使用GPU进行训练
4. 可以通过调整`beta`参数来控制偏好学习的强度
5. 如果遇到内存问题，可以减小`batch_size`或`max_seq_length`

## 输出说明

训练过程中会生成以下文件：

- `dpo_checkpoints/`: 检查点目录
  - `checkpoint_*`: 训练过程中的检查点
  - `final_model/`: 最终模型
  - `dpo_training.log`: 训练日志
  - `dpo_main.log`: 主程序日志

## 故障排除

1. 如果遇到内存不足：
   - 减小`batch_size`
   - 减小`max_seq_length`
   - 禁用混合精度训练（设置`use_amp=False`）

2. 如果训练不稳定：
   - 调整`learning_rate`
   - 调整`beta`参数
   - 增加`batch_size`

3. 如果模型性能不理想：
   - 检查训练数据质量
   - 增加训练轮数
   - 调整`beta`参数 