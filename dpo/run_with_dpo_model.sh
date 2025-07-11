#!/bin/bash
set -e
set -x

# 使用DPO训练后的模型路径
DPO_MODEL_PATH="/kaggle/working/dpo_training/dpo_model_20250629_164141/final_model/checkpoint_960"

# 环境变量设置
export TESTDIR="/kaggle/working/ag4mtest"
export AG4MDIR="/kaggle/working/ag4masses"
export AGLIB="/kaggle/working/aglib"
export AGDIR="$AG4MDIR/alphageometry"
export PYTHONPATH="$PYTHONPATH:$AGDIR:$AGLIB"

# 模型参数
export BATCH_SIZE="16"
export BEAM_SIZE="64"
export DEPTH="16"
export NWORKERS="2"
export CUDA_VISIBLE_DEVICES="0,1"

# 问题设置
export PROB_FILE="$AG4MDIR/data/ag4m_problems.txt"
export PROB="imo-2024-q4"
export MODEL="alphageometry"

# 输出文件
OUTFILE="$TESTDIR/${PROB}_dpo.out"
ERRFILE="$TESTDIR/${PROB}_dpo.log"

# 确保输出目录存在
mkdir -p "$TESTDIR"

# 重定向输出
exec >"$ERRFILE" 2>&1

echo "使用DPO训练模型: $DPO_MODEL_PATH"
echo "问题: $PROB"
echo "输出文件: $OUTFILE"

# DDAR参数
DDAR_ARGS=(
  --defs_file="$AGDIR/defs.txt"
  --rules_file="$AGDIR/rules.txt"
)

# 搜索参数
SEARCH_ARGS=(
  --beam_size="$BEAM_SIZE"
  --search_depth="$DEPTH"
)

# 语言模型参数 - 使用DPO模型
LM_ARGS=(
  --ckpt_path="$DPO_MODEL_PATH"
  --vocab_path="$AGLIB/ag_ckpt_vocab/geometry.757.model"
  --gin_search_paths="$AGLIB/meliad/transformer/configs,$AGDIR"
  --gin_file=base_htrans.gin
  --gin_file=size/medium_150M.gin
  --gin_file=options/positions_t5.gin
  --gin_file=options/lr_cosine_decay.gin
  --gin_file=options/seq_1024_nocache.gin
  --gin_file=geometry_150M_generate.gin
  --gin_param=DecoderOnlyLanguageModelGenerate.output_token_losses=True
  --gin_param=TransformerTaskConfig.batch_size="$BATCH_SIZE"
  --gin_param=TransformerTaskConfig.sequence_length=128
  --gin_param=Trainer.restore_state_variables=False
)

echo "=========================================="
echo "开始使用DPO模型求解几何问题"
echo "=========================================="

python -m alphageometry \
  --alsologtostderr \
  --problems_file="$PROB_FILE" \
  --problem_name="$PROB" \
  --mode="$MODEL" \
  "${DDAR_ARGS[@]}" \
  "${SEARCH_ARGS[@]}" \
  "${LM_ARGS[@]}" \
  --out_file="$OUTFILE" \
  --n_workers="$NWORKERS"

echo "=========================================="
echo "DPO模型求解完成"
echo "结果保存在: $OUTFILE"
echo "日志保存在: $ERRFILE"
echo "==========================================" 