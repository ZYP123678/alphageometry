#!/usr/bin/env python3
"""
在本地使用DPO训练后的AlphaGeometry模型
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """设置环境变量"""
    # 根据实际情况修改这些路径
    env_vars = {
        "BATCH_SIZE": "16",
        "BEAM_SIZE": "64", 
        "DEPTH": "16",
        "NWORKERS": "2",
        "CUDA_VISIBLE_DEVICES": "0"  # 根据您的GPU数量调整
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars

def run_alphageometry_with_dpo(
    dpo_model_path,
    vocab_path,
    meliad_path,
    alphageometry_path,
    problem_file,
    problem_name="imo-2024-q4",
    output_dir="./outputs",
    mode="alphageometry"
):
    """
    使用DPO训练后的模型运行AlphaGeometry
    
    Args:
        dpo_model_path: DPO模型路径 (checkpoint_960目录)
        vocab_path: 词汇表文件路径
        meliad_path: meliad库路径
        alphageometry_path: alphageometry源码路径
        problem_file: 问题文件路径
        problem_name: 问题名称
        output_dir: 输出目录
        mode: 运行模式 (alphageometry 或 ddar)
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件
    outfile = os.path.join(output_dir, f"{problem_name}_dpo.out")
    errfile = os.path.join(output_dir, f"{problem_name}_dpo.log")
    
    # 设置Python路径
    python_path = f"{alphageometry_path}:{meliad_path}"
    if "PYTHONPATH" in os.environ:
        os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{python_path}"
    else:
        os.environ["PYTHONPATH"] = python_path
    
    print(f"使用DPO模型: {dpo_model_path}")
    print(f"问题: {problem_name}")
    print(f"输出文件: {outfile}")
    print(f"日志文件: {errfile}")
    
    # 构建命令
    cmd = [
        sys.executable, "-m", "alphageometry",
        "--alsologtostderr",
        f"--problems_file={problem_file}",
        f"--problem_name={problem_name}",
        f"--mode={mode}",
        f"--defs_file={alphageometry_path}/defs.txt",
        f"--rules_file={alphageometry_path}/rules.txt",
        f"--beam_size={os.environ['BEAM_SIZE']}",
        f"--search_depth={os.environ['DEPTH']}",
        f"--ckpt_path={dpo_model_path}",
        f"--vocab_path={vocab_path}",
        f"--gin_search_paths={meliad_path}/transformer/configs,{alphageometry_path}",
        "--gin_file=base_htrans.gin",
        "--gin_file=size/medium_150M.gin",
        "--gin_file=options/positions_t5.gin",
        "--gin_file=options/lr_cosine_decay.gin",
        "--gin_file=options/seq_1024_nocache.gin",
        "--gin_file=geometry_150M_generate.gin",
        "--gin_param=DecoderOnlyLanguageModelGenerate.output_token_losses=True",
        f"--gin_param=TransformerTaskConfig.batch_size={os.environ['BATCH_SIZE']}",
        "--gin_param=TransformerTaskConfig.sequence_length=128",
        "--gin_param=Trainer.restore_state_variables=False",
        f"--out_file={outfile}",
        f"--n_workers={os.environ['NWORKERS']}"
    ]
    
    print("==========================================")
    print("开始使用DPO模型求解几何问题")
    print("==========================================")
    
    # 运行命令
    try:
        with open(errfile, 'w') as log_file:
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600  # 1小时超时
            )
            log_file.write(result.stdout)
        
        print("==========================================")
        print("DPO模型求解完成")
        print(f"结果保存在: {outfile}")
        print(f"日志保存在: {errfile}")
        print(f"返回码: {result.returncode}")
        print("==========================================")
        
        # 显示结果
        if os.path.exists(outfile):
            print("\n求解结果:")
            with open(outfile, 'r') as f:
                print(f.read())
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("求解超时（1小时）")
        return False
    except Exception as e:
        print(f"运行出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="使用DPO训练后的AlphaGeometry模型")
    parser.add_argument("--dpo_model", required=True, help="DPO模型路径 (checkpoint_960目录)")
    parser.add_argument("--vocab_path", required=True, help="词汇表文件路径")
    parser.add_argument("--meliad_path", required=True, help="meliad库路径")
    parser.add_argument("--alphageometry_path", required=True, help="alphageometry源码路径")
    parser.add_argument("--problem_file", required=True, help="问题文件路径")
    parser.add_argument("--problem_name", default="imo-2024-q4", help="问题名称")
    parser.add_argument("--output_dir", default="./outputs", help="输出目录")
    parser.add_argument("--mode", default="alphageometry", choices=["alphageometry", "ddar"], help="运行模式")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 验证路径
    if not os.path.exists(args.dpo_model):
        print(f"错误: DPO模型路径不存在: {args.dpo_model}")
        return 1
    
    if not os.path.exists(args.vocab_path):
        print(f"错误: 词汇表路径不存在: {args.vocab_path}")
        return 1
    
    if not os.path.exists(args.problem_file):
        print(f"错误: 问题文件不存在: {args.problem_file}")
        return 1
    
    # 运行
    success = run_alphageometry_with_dpo(
        dpo_model_path=args.dpo_model,
        vocab_path=args.vocab_path,
        meliad_path=args.meliad_path,
        alphageometry_path=args.alphageometry_path,
        problem_file=args.problem_file,
        problem_name=args.problem_name,
        output_dir=args.output_dir,
        mode=args.mode
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 