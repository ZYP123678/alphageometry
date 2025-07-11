# 在Kaggle笔记本中使用DPO训练模型的代码块

import os
import subprocess

# 设置路径
DPO_MODEL_PATH = "/kaggle/working/dpo_training/dpo_model_20250629_164141/final_model/checkpoint_960"
TESTDIR = "/kaggle/working/ag4mtest"
AG4MDIR = "/kaggle/working/ag4masses"
AGLIB = "/kaggle/working/aglib"
AGDIR = f"{AG4MDIR}/alphageometry"

# 确保输出目录存在
os.makedirs(TESTDIR, exist_ok=True)

# 设置环境变量
os.environ.update({
    "TESTDIR": TESTDIR,
    "AG4MDIR": AG4MDIR,
    "AGLIB": AGLIB,
    "BATCH_SIZE": "16",
    "BEAM_SIZE": "64", 
    "DEPTH": "16",
    "NWORKERS": "2",
    "CUDA_VISIBLE_DEVICES": "0,1"
})

# 问题设置
PROB_FILE = f"{AG4MDIR}/data/ag4m_problems.txt"
PROB = "imo-2024-q4"
MODEL = "alphageometry"

# 输出文件
OUTFILE = f"{TESTDIR}/{PROB}_dpo.out"
ERRFILE = f"{TESTDIR}/{PROB}_dpo.log"

print(f"使用DPO训练模型: {DPO_MODEL_PATH}")
print(f"问题: {PROB}")
print(f"输出文件: {OUTFILE}")

# 构建命令
cmd = [
    "python", "-m", "alphageometry",
    "--alsologtostderr",
    f"--problems_file={PROB_FILE}",
    f"--problem_name={PROB}",
    f"--mode={MODEL}",
    f"--defs_file={AGDIR}/defs.txt",
    f"--rules_file={AGDIR}/rules.txt",
    f"--beam_size={os.environ['BEAM_SIZE']}",
    f"--search_depth={os.environ['DEPTH']}",
    f"--ckpt_path={DPO_MODEL_PATH}",
    f"--vocab_path={AGLIB}/ag_ckpt_vocab/geometry.757.model",
    f"--gin_search_paths={AGLIB}/meliad/transformer/configs,{AGDIR}",
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
    f"--out_file={OUTFILE}",
    f"--n_workers={os.environ['NWORKERS']}"
]

print("==========================================")
print("开始使用DPO模型求解几何问题")
print("==========================================")

# 运行命令
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
    
    # 保存日志
    with open(ERRFILE, 'w') as f:
        f.write(result.stdout)
        f.write(result.stderr)
    
    print("==========================================")
    print("DPO模型求解完成")
    print(f"结果保存在: {OUTFILE}")
    print(f"日志保存在: {ERRFILE}")
    print(f"返回码: {result.returncode}")
    print("==========================================")
    
    # 显示结果
    if os.path.exists(OUTFILE):
        print("\n求解结果:")
        with open(OUTFILE, 'r') as f:
            print(f.read())
    
except subprocess.TimeoutExpired:
    print("求解超时（1小时）")
except Exception as e:
    print(f"运行出错: {e}") 