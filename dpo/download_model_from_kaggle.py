# 在Kaggle笔记本中运行此代码块来打包和下载DPO模型

import os
import subprocess
import zipfile
from datetime import datetime

# DPO模型路径
DPO_MODEL_PATH = "/kaggle/working/dpo_training/dpo_model_20250629_164141/final_model/checkpoint_960"
MODEL_NAME = "dpo_trained_model"

# 创建打包目录
PACKAGE_DIR = f"/kaggle/working/{MODEL_NAME}"
os.makedirs(PACKAGE_DIR, exist_ok=True)

print(f"正在打包DPO模型: {DPO_MODEL_PATH}")
print(f"打包目录: {PACKAGE_DIR}")

# 复制模型文件到打包目录
subprocess.run([
    "cp", "-r", DPO_MODEL_PATH, f"{PACKAGE_DIR}/checkpoint_960"
], check=True)

# 复制词汇表文件（如果需要）
VOCAB_SRC = "/kaggle/working/aglib/ag_ckpt_vocab"
if os.path.exists(VOCAB_SRC):
    subprocess.run([
        "cp", "-r", VOCAB_SRC, f"{PACKAGE_DIR}/vocab"
    ], check=True)

# 创建模型信息文件
model_info = {
    "model_type": "DPO_trained_AlphaGeometry",
    "training_date": "2025-06-29",
    "checkpoint_step": "960",
    "original_model": "AlphaGeometry_150M",
    "training_method": "Direct_Preference_Optimization",
    "usage": "Replace --ckpt_path with this directory path"
}

import json
with open(f"{PACKAGE_DIR}/model_info.json", 'w') as f:
    json.dump(model_info, f, indent=2)

# 创建使用说明
readme_content = f"""
# DPO训练后的AlphaGeometry模型

## 模型信息
- 训练日期: 2025-06-29
- 检查点步数: 960
- 原始模型: AlphaGeometry 150M
- 训练方法: Direct Preference Optimization (DPO)

## 使用方法

### 1. 解压模型文件
```bash
unzip {MODEL_NAME}.zip
```

### 2. 在AlphaGeometry中使用
将原来的模型路径:
```bash
--ckpt_path=/path/to/original/ag_ckpt_vocab
```

替换为:
```bash
--ckpt_path=/path/to/{MODEL_NAME}/checkpoint_960
```

### 3. 完整命令示例
```bash
python -m alphageometry \\
  --problems_file=problems.txt \\
  --problem_name=imo-2024-q4 \\
  --mode=alphageometry \\
  --ckpt_path=/path/to/{MODEL_NAME}/checkpoint_960 \\
  --vocab_path=/path/to/vocab/geometry.757.model \\
  --gin_search_paths=/path/to/meliad/transformer/configs,/path/to/alphageometry \\
  --gin_file=base_htrans.gin \\
  --gin_file=size/medium_150M.gin \\
  --gin_file=options/positions_t5.gin \\
  --gin_file=options/lr_cosine_decay.gin \\
  --gin_file=options/seq_1024_nocache.gin \\
  --gin_file=geometry_150M_generate.gin \\
  --out_file=result.out
```

## 文件结构
```
{MODEL_NAME}/
├── checkpoint_960/          # 模型权重文件
│   ├── array_metadatas/
│   ├── d/
│   ├── _CHECKPOINT_METADATA
│   ├── manifest.ocdbt
│   ├── _METADATA
│   ├── ocdbt.process_0/
│   └── _sharding
├── vocab/                   # 词汇表文件（可选）
│   ├── geometry.757.vocab
│   ├── geometry.757.model
│   └── checkpoint_10999999
└── model_info.json         # 模型信息
```

## 注意事项
1. 确保词汇表文件路径正确
2. 保持其他gin配置文件不变
3. 此模型经过DPO强化学习优化，在特定任务上可能有更好的性能
"""

with open(f"{PACKAGE_DIR}/README.md", 'w') as f:
    f.write(readme_content)

# 创建ZIP文件
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_filename = f"/kaggle/working/{MODEL_NAME}_{timestamp}.zip"

print(f"正在创建ZIP文件: {zip_filename}")

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(PACKAGE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, PACKAGE_DIR)
            zipf.write(file_path, arcname)

# 显示文件信息
zip_size = os.path.getsize(zip_filename) / (1024 * 1024)  # MB
print(f"ZIP文件创建完成: {zip_filename}")
print(f"文件大小: {zip_size:.2f} MB")

# 列出ZIP文件内容
print("\nZIP文件内容:")
with zipfile.ZipFile(zip_filename, 'r') as zipf:
    for info in zipf.infolist():
        print(f"  {info.filename}")

print(f"\n✅ 模型打包完成！")
print(f"📦 下载文件: {zip_filename}")
print(f"📋 使用说明已包含在ZIP文件的README.md中") 