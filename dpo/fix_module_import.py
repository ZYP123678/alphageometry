#!/usr/bin/env python3
"""
修复alphageometry模块导入问题
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_python_path():
    """设置正确的Python路径"""
    
    # 检查是否在Jupyter/IPython环境中
    try:
        # 尝试获取当前脚本路径
        current_dir = Path(__file__).parent.absolute()
    except NameError:
        # 在Jupyter环境中，使用当前工作目录
        current_dir = Path.cwd()
        print("📝 检测到Jupyter环境，使用当前工作目录")
    
    # 项目根目录（包含alphageometry文件夹的目录）
    project_root = current_dir.parent
    
    # alphageometry模块路径
    alphageometry_path = project_root / "alphageometry"
    
    # meliad路径（如果存在）
    meliad_path = project_root / "meliad"
    
    # 检查路径是否存在
    if not alphageometry_path.exists():
        print(f"错误: alphageometry路径不存在: {alphageometry_path}")
        print(f"当前目录: {current_dir}")
        print(f"项目根目录: {project_root}")
        return False
    
    # 设置PYTHONPATH
    python_paths = [str(alphageometry_path)]
    
    if meliad_path.exists():
        python_paths.append(str(meliad_path))
    
    # 添加到sys.path
    for path in python_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # 设置环境变量
    os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)
    
    print(f"✅ Python路径设置完成:")
    print(f"   alphageometry: {alphageometry_path}")
    if meliad_path.exists():
        print(f"   meliad: {meliad_path}")
    
    return True

def test_import():
    """测试模块导入"""
    try:
        import alphageometry
        print("✅ alphageometry模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ alphageometry模块导入失败: {e}")
        return False

def run_alphageometry_with_dpo(
    dpo_model_path,
    vocab_path,
    problem_file,
    problem_name="imo-2024-q4",
    output_dir="./outputs",
    mode="alphageometry"
):
    """使用DPO模型运行AlphaGeometry"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件
    outfile = os.path.join(output_dir, f"{problem_name}_dpo.out")
    errfile = os.path.join(output_dir, f"{problem_name}_dpo.log")
    
    # 获取alphageometry模块路径
    try:
        current_dir = Path(__file__).parent.absolute()
    except NameError:
        current_dir = Path.cwd()
    
    alphageometry_path = current_dir.parent / "alphageometry"
    
    print(f"使用DPO模型: {dpo_model_path}")
    print(f"问题: {problem_name}")
    print(f"输出文件: {outfile}")
    
    # 构建命令
    cmd = [
        sys.executable, "-m", "alphageometry",
        "--alsologtostderr",
        f"--problems_file={problem_file}",
        f"--problem_name={problem_name}",
        f"--mode={mode}",
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
    """主函数"""
    print("🔧 修复alphageometry模块导入问题")
    print("=" * 50)
    
    # 设置Python路径
    if not setup_python_path():
        return 1
    
    # 测试导入
    if not test_import():
        print("\n💡 可能的解决方案:")
        print("1. 确保您在正确的目录中运行脚本")
        print("2. 检查alphageometry文件夹是否存在")
        print("3. 确保所有依赖已安装")
        print("4. 尝试使用绝对路径")
        return 1
    
    print("\n✅ 模块导入问题已修复！")
    print("现在可以正常使用alphageometry模块了")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 