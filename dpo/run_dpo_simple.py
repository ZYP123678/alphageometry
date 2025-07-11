#!/usr/bin/env python3
"""
简化的DPO模型运行脚本 - 解决模块导入问题
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent
    
    # 设置路径
    alphageometry_path = project_root / "alphageometry"
    
    # 检查alphageometry路径是否存在
    if not alphageometry_path.exists():
        print(f"❌ 错误: alphageometry路径不存在: {alphageometry_path}")
        print("请确保您在正确的目录中运行此脚本")
        return 1
    
    # 设置环境变量
    os.environ["PYTHONPATH"] = str(alphageometry_path)
    
    # DPO模型路径（请根据实际情况修改）
    dpo_model_path = "/path/to/your/dpo_model/checkpoint_960"  # 请修改为实际路径
    vocab_path = "/path/to/your/vocab/geometry.757.model"      # 请修改为实际路径
    problem_file = str(alphageometry_path / "data/ag4m_problems.txt")
    
    # 检查必要文件是否存在
    if not os.path.exists(dpo_model_path):
        print(f"❌ DPO模型路径不存在: {dpo_model_path}")
        print("请修改脚本中的dpo_model_path变量")
        return 1
    
    if not os.path.exists(vocab_path):
        print(f"❌ 词汇表路径不存在: {vocab_path}")
        print("请修改脚本中的vocab_path变量")
        return 1
    
    # 输出文件
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, "imo-2024-q4_dpo.out")
    
    print("🚀 开始运行DPO模型...")
    print(f"📁 alphageometry路径: {alphageometry_path}")
    print(f"🤖 DPO模型路径: {dpo_model_path}")
    print(f"📝 输出文件: {outfile}")
    
    # 构建命令
    cmd = [
        sys.executable, "-m", "alphageometry",
        "--alsologtostderr",
        f"--problems_file={problem_file}",
        "--problem_name=imo-2024-q4",
        "--mode=alphageometry",
        f"--defs_file={alphageometry_path}/defs.txt",
        f"--rules_file={alphageometry_path}/rules.txt",
        "--beam_size=64",
        "--search_depth=16",
        f"--ckpt_path={dpo_model_path}",
        f"--vocab_path={vocab_path}",
        f"--gin_search_paths={alphageometry_path}",
        "--gin_file=base_htrans.gin",
        "--gin_file=size/medium_150M.gin",
        "--gin_file=options/positions_t5.gin",
        "--gin_file=options/lr_cosine_decay.gin",
        "--gin_file=options/seq_1024_nocache.gin",
        "--gin_file=geometry_150M_generate.gin",
        "--gin_param=DecoderOnlyLanguageModelGenerate.output_token_losses=True",
        "--gin_param=TransformerTaskConfig.batch_size=16",
        "--gin_param=TransformerTaskConfig.sequence_length=128",
        "--gin_param=Trainer.restore_state_variables=False",
        f"--out_file={outfile}",
        "--n_workers=2"
    ]
    
    print("\n📋 运行命令:")
    print(" ".join(cmd))
    print("\n" + "="*50)
    
    # 运行命令
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        print("✅ 运行完成!")
        print(f"返回码: {result.returncode}")
        
        if result.stdout:
            print("\n📤 标准输出:")
            print(result.stdout)
        
        if result.stderr:
            print("\n⚠️ 错误输出:")
            print(result.stderr)
        
        if os.path.exists(outfile):
            print(f"\n📄 结果文件: {outfile}")
            with open(outfile, 'r') as f:
                print(f.read())
        
        return 0 if result.returncode == 0 else 1
        
    except subprocess.TimeoutExpired:
        print("⏰ 运行超时（1小时）")
        return 1
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        return 1

if __name__ == "__main__":
    print("🔧 DPO模型运行脚本")
    print("=" * 50)
    print("⚠️  请先修改脚本中的路径变量:")
    print("   - dpo_model_path: DPO模型路径")
    print("   - vocab_path: 词汇表路径")
    print("=" * 50)
    
    sys.exit(main()) 