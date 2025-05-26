## after running train.sh, use lm-evaluation-harness provided by LLM-Pruner repo

#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICE=3
MODEL="hf-causal-experimental"
PYTHON_SCRIPT="LLM-Pruner/lm-evaluation-harness/main.py"
BASE_CHECKPOINT="/disk/dingding/LLM_P/pytorch_model.bin"
CONFIG_PRETRAINED="meta-llama/Llama-3.2-1B"
TASKS="arc_easy,wikitext"
DEVICE="cuda:0"
NO_CACHE="--no_cache"

# Function to evaluate a dataset group
run_evaluation() {
  local DATASET_NAME=$1
  local PARQUET_DIR=$2

  echo ">>> Starting evaluation for dataset: $DATASET_NAME"
  
  PARQUET_FILES=(
      "${PARQUET_DIR}/10%_5176.parquet"
      "${PARQUET_DIR}/20%_10352.parquet"
      "${PARQUET_DIR}/30%_15528.parquet"
      "${PARQUET_DIR}/40%_20704.parquet"
      "${PARQUET_DIR}/50%_25880.parquet"
      "${PARQUET_DIR}/60%_31056.parquet"
      "${PARQUET_DIR}/70%_36232.parquet"
      "${PARQUET_DIR}/80%_41408.parquet"
      "${PARQUET_DIR}/90%_46584.parquet"
  )

  for pf in "${PARQUET_FILES[@]}"; do
    name=$(basename "$pf" .parquet)
    id=${name//%/_}
    output_path="/disk/dingding/algo_models/${DATASET_NAME}/${id}.json"
    peft_path="/disk/dingding/algo_models/${DATASET_NAME}/${id}"

    echo "=== Evaluating ${name} → id=${id}"

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 $PYTHON_SCRIPT \
      --model $MODEL \
      --model_args checkpoint=$BASE_CHECKPOINT,peft=$peft_path,config_pretrained=$CONFIG_PRETRAINED \
      --tasks $TASKS \
      --device $DEVICE $NO_CACHE \
      --output_path $output_path

    echo
  done
}

# Run evaluations for each dataset group
run_evaluation "pbc_random" "pbc_random"
run_evaluation "sg_facility" "sg_facility"
run_evaluation "sg_norms" "sg_norms"
