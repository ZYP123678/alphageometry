# !/bin/bash
set -e
set -x

# Directory where output files go
TESTDIR=$HOME/ag4mtest
# Directory containing AG4Masses source files
AG4MDIR=$HOME/ag4masses
# Directory containing external libraries including ag_ckpt_vocab and meliad
AGLIB=$HOME/aglib

AGDIR=$AG4MDIR/alphageometry
export PYTHONPATH=$PYTHONPATH:$AGDIR:$AGLIB

PROB_FILE=$AG4MDIR/data/ag4m_problems.txt
PROB=square_angle15
# alphageometry | ddar
MODEL=alphageometry

# stdout, solution is written here
OUTFILE=$TESTDIR/${PROB}.out
# stderr, a lot of information, error message, log etc.
ERRFILE=$TESTDIR/${PROB}.log

# stdout and stderr are written to both ERRFILF and console
exec > >(tee $ERRFILE) 2>&1

echo PROB=$PROB
echo PROB_FILE=$PROB_FILE
echo MODEL=$MODEL

echo OUTFILE=$OUTFILE
echo ERRFILE=$ERRFILE

# BATCH_SIZE: number of outputs for each LM query
# BEAM_SIZE: size of the breadth-first search queue
# DEPTH: search depth (number of auxilary points to add)
# NWORKERS: number of parallel run worker processes. Rule of thumb: on a 128G machine with 16 logical CPUs,
#   use NWORKERS=8, BATCH_SIZE=24.
# 
# Memory usage is affected by BATCH_SIZE, NWORKER and complexity of the problem.
# Larger NWORKER and BATCH_SIZE tends to cause out of memory issue

BATCH_SIZE=32
BEAM_SIZE=32
DEPTH=16
NWORKERS=1

#The results in Google's paper can be obtained by setting BATCH_SIZE=32, BEAM_SIZE=512, DEPTH=16

DATA=$AGLIB/ag_ckpt_vocab
MELIAD_PATH=$AGLIB/meliad
export PYTHONPATH=$PYTHONPATH:$MELIAD_PATH

COLLECT_RL_DATA=true  # 设置为false禁用RL数据收集
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RL_DATA_DIR=$TESTDIR/rl_data_${TIMESTAMP}

DDAR_ARGS=(
  --defs_file=$AGDIR/defs.txt \
  --rules_file=$AGDIR/rules.txt \
);

SEARCH_ARGS=(
  --beam_size=$BEAM_SIZE
  --search_depth=$DEPTH
)

RL_ARGS=(
  --collect_rl_data=$COLLECT_RL_DATA
  --rl_data_dir=$RL_DATA_DIR
)

LM_ARGS=(
  --ckpt_path=$DATA \
  --vocab_path=$DATA/geometry.757.model \
  --gin_search_paths=$MELIAD_PATH/transformer/configs,$AGDIR \
  --gin_file=base_htrans.gin \
  --gin_file=size/medium_150M.gin \
  --gin_file=options/positions_t5.gin \
  --gin_file=options/lr_cosine_decay.gin \
  --gin_file=options/seq_1024_nocache.gin \
  --gin_file=geometry_150M_generate.gin \
  --gin_param=DecoderOnlyLanguageModelGenerate.output_token_losses=True \
  --gin_param=TransformerTaskConfig.batch_size=$BATCH_SIZE \
  --gin_param=TransformerTaskConfig.sequence_length=128 \
  --gin_param=Trainer.restore_state_variables=False
);

if [ "$COLLECT_RL_DATA" = "false" ]; then
  RL_ARGS=()
  echo "RL数据收集已禁用"
else
  # 确保RL数据目录存在
  mkdir -p $RL_DATA_DIR
  echo "RL数据将保存到: $RL_DATA_DIR"
fi

true "=========================================="


python -m alphageometry \
--alsologtostderr \
--problems_file=$PROB_FILE \
--problem_name=$PROB \
--mode=$MODEL \
"${DDAR_ARGS[@]}" \
"${SEARCH_ARGS[@]}" \
"${RL_ARGS[@]}" \
"${LM_ARGS[@]}" \
--out_file=$OUTFILE \
--n_workers=$NWORKERS 2>&1


# 如果启用了RL数据收集，显示收集结果
if [ "$COLLECT_RL_DATA" = "true" ]; then
  echo "=========================================="
  echo "RL数据收集完成，保存路径: $RL_DATA_DIR"
  if [ -d "$RL_DATA_DIR" ]; then
    ls -l $RL_DATA_DIR
    COUNT=$(cat $RL_DATA_DIR/*.jsonl 2>/dev/null | wc -l || echo "0")
    echo "共收集了 $COUNT 条样本"
  else
    echo "警告: RL数据目录未创建"
  fi
fi