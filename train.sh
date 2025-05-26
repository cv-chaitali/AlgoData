## git clone https://github.com/horseee/LLM-Pruner.git and tune with normal alpaca template 

#!/bin/bash

# Shared configuration
CUDA_DEVICE=3
PYTHON_SCRIPT="/disk/dingding/LLM_P/LLM-Pruner/all_tuning.py"
PRUNE_MODEL="/disk/dingding/LLM_P/pytorch_model.bin"
DATA_PATH="selen-kim/selected-dataset"
LORA_R=32
ONLY_TRAIN_FLAG="--only_train"

# Function to run tuning on a dataset group
run_tuning() {
    local DATASET_NAME=$1
    local BASE_OUTPUT_DIR="/disk/dingding/algo_models/${DATASET_NAME}"
    local PREFIX="${DATASET_NAME}"

    # Define parquet files for this dataset
    local PARQUET_FILES=(
        "${PREFIX}/10%_5176.parquet"
        "${PREFIX}/20%_10352.parquet"
        "${PREFIX}/30%_15528.parquet"
        "${PREFIX}/40%_20704.parquet"
        "${PREFIX}/50%_25880.parquet"
        "${PREFIX}/60%_31056.parquet"
        "${PREFIX}/70%_36232.parquet"
        "${PREFIX}/80%_41408.parquet"
        "${PREFIX}/90%_46584.parquet"
    )

    # Run tuning for each parquet file
    for file in "${PARQUET_FILES[@]}"; do
        filename_base=$(basename "$file" .parquet)
        filename_base="${filename_base//%/_}" 
        output_dir="${BASE_OUTPUT_DIR}/${filename_base}"

        echo "Running command for: ${file}"
        echo "Output directory: ${output_dir}"

        CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python3 ${PYTHON_SCRIPT} \
            --prune_model "${PRUNE_MODEL}" \
            --data_path "${DATA_PATH}" \
            --output_dir "${output_dir}" \
            --file "${file}" \
            --lora_r "${LORA_R}" \
            ${ONLY_TRAIN_FLAG}
        echo "----------------------------------------------------"
    done
}

# Run for each dataset group
run_tuning "craig"
run_tuning "sg_facility"
run_tuning "pbc_random"
