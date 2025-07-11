# 在Jupyter中运行DPO模型的完整代码块
# 复制此代码到Jupyter notebook中运行

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """设置Jupyter环境"""
    print("🔧 设置Jupyter环境...")
    
    # 获取当前工作目录
    current_dir = Path.cwd()
    print(f"📁 当前工作目录: {current_dir}")
    
    # 查找alphageometry模块
    possible_paths = [
        current_dir / "alphageometry",
        current_dir.parent / "alphageometry", 
        current_dir / "alphageometry" / "alphageometry",
    ]
    
    alphageometry_path = None
    for path in possible_paths:
        if path.exists():
            alphageometry_path = path
            print(f"✅ 找到alphageometry模块: {path}")
            break
    
    if alphageometry_path is None:
        print("❌ 未找到alphageometry模块")
        return None
    
    # 设置Python路径
    if str(alphageometry_path) not in sys.path:
        sys.path.insert(0, str(alphageometry_path))
    
    os.environ["PYTHONPATH"] = str(alphageometry_path)
    
    return alphageometry_path

def run_dpo_model(
    dpo_model_path,
    vocab_path,
    problem_name="imo-2024-q4",
    output_dir="./outputs"
):
    """运行DPO模型"""
    
    # 设置环境
    alphageometry_path = setup_environment()
    if alphageometry_path is None:
        print("❌ 环境设置失败")
        return False
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件
    outfile = os.path.join(output_dir, f"{problem_name}_dpo.out")
    errfile = os.path.join(output_dir, f"{problem_name}_dpo.log")
    
    # 问题文件路径
    problem_file = str(alphageometry_path / "data/ag4m_problems.txt")
    
    print(f"🤖 DPO模型路径: {dpo_model_path}")
    print(f"📝 词汇表路径: {vocab_path}")
    print(f"📄 问题文件: {problem_file}")
    print(f"📤 输出文件: {outfile}")
    
    # 构建命令
    cmd = [
        sys.executable, "-m", "alphageometry",
        "--alsologtostderr",
        f"--problems_file={problem_file}",
        f"--problem_name={problem_name}",
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
    
    print("\n🚀 开始运行DPO模型...")
    print("=" * 50)
    
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
        
        print("✅ 运行完成!")
        print(f"返回码: {result.returncode}")
        
        if result.stdout:
            print("\n📤 输出日志:")
            print(result.stdout[-1000:])  # 显示最后1000个字符
        
        if os.path.exists(outfile):
            print(f"\n📄 结果文件: {outfile}")
            with open(outfile, 'r') as f:
                result_content = f.read()
                print(result_content)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ 运行超时（1小时）")
        return False
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        return False

# 使用示例 - 请修改以下路径
if __name__ == "__main__":
    # 请根据实际情况修改这些路径
    DPO_MODEL_PATH = "/path/to/your/dpo_model/checkpoint_960"  # 修改为实际路径
    VOCAB_PATH = "/path/to/your/vocab/geometry.757.model"      # 修改为实际路径
    
    print("🎯 DPO模型运行脚本")
    print("=" * 50)
    print("⚠️  请先修改以下路径变量:")
    print(f"   DPO_MODEL_PATH = '{DPO_MODEL_PATH}'")
    print(f"   VOCAB_PATH = '{VOCAB_PATH}'")
    print("=" * 50)
    
    # 检查路径是否存在
    if not os.path.exists(DPO_MODEL_PATH):
        print(f"❌ DPO模型路径不存在: {DPO_MODEL_PATH}")
    elif not os.path.exists(VOCAB_PATH):
        print(f"❌ 词汇表路径不存在: {VOCAB_PATH}")
    else:
        # 运行模型
        success = run_dpo_model(DPO_MODEL_PATH, VOCAB_PATH)
        if success:
            print("\n🎉 DPO模型运行成功!")
        else:
            print("\n❌ DPO模型运行失败") 