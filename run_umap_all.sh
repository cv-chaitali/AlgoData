#!/bin/bash

export LD_LIBRARY_PATH=/home/selen/anaconda3/envs/algo/lib:$LD_LIBRARY_PATH

EMBEDDING_PATH="./embeddings_checkpoint_alpaca.npy"
INDEX_DIR="recovered_indices"
OUTPUT_DIR="umap_plot"

mkdir -p "$OUTPUT_DIR"

declare -A METHODS=(
  ["craig"]="CRAIG"
  ["pbc_random"]="PBC"
  ["sg_facility"]="SG_FACILITY"
  ["sg_norms"]="SG_NORMS"
)

PERCENT_SIZES=(
  "10 5176"
  "20 10352"
  "30 15528"
  "40 20704"
  "50 25880"
  "60 31056"
  "70 36232"
  "80 41408"
  "90 46584"
)

for METHOD in "${!METHODS[@]}"; do
  for entry in "${PERCENT_SIZES[@]}"; do
    PERCENT=$(echo $entry | cut -d ' ' -f1)
    SIZE=$(echo $entry | cut -d ' ' -f2)

    INDEX_PATH="$INDEX_DIR/${METHOD}_selected_${SIZE}_indices.json"
    OUTPUT_PATH="$OUTPUT_DIR/${METHOD}_umap_${PERCENT}.png"
    LABEL="${METHODS[$METHOD]} ${PERCENT}%"

    if [[ -f "$INDEX_PATH" ]]; then
      echo ">> Generating UMAP for $LABEL using $INDEX_PATH"
      python visualize_umap.py \
        --embedding_path "$EMBEDDING_PATH" \
        --subset_path "$INDEX_PATH" \
        --label "$LABEL" \
        --output_path "$OUTPUT_PATH"
    else
      echo " Missing file: $INDEX_PATH"
    fi
  done
done
