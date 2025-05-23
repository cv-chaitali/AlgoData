#!/bin/bash

EMBEDDING_CHECKPOINT="embeddings_checkpoint_alpaca.npy"

# 전체 임베딩 수 확인
TOTAL_SAMPLES=$(python -c "import numpy as np; print(np.load('$EMBEDDING_CHECKPOINT').shape[0])")
echo "Total samples in embedding: $TOTAL_SAMPLES"

# 비율: 10% ~ 90%
for PERCENT in $(seq 10 10 90); do
  SUBSET_SIZE=$((TOTAL_SAMPLES * PERCENT / 100))
  echo ""
  echo "Running subset selection for ${PERCENT}% of data (${SUBSET_SIZE} samples)"
  TIMING_FILE="execution_times_${SUBSET_SIZE}.txt"

  echo "Algorithm Execution Times" > "$TIMING_FILE"
  echo "--------------------------" >> "$TIMING_FILE"
  echo "Date: $(date)" >> "$TIMING_FILE"
  echo "Subset size: $SUBSET_SIZE" >> "$TIMING_FILE"
  echo "--------------------------" >> "$TIMING_FILE"

  #############################
  # Method 1: CRAIG
  #############################
  echo "Running CRAIG..."
  echo "Method: CRAIG" >> "$TIMING_FILE"
  { time -p python dswa.py \
      --embedding_checkpoint "$EMBEDDING_CHECKPOINT" \
      --subset_size "$SUBSET_SIZE" \
      --method craig \
      --output_json "craig_selected_${SUBSET_SIZE}.json" ; } >> "$TIMING_FILE" 2>> "$TIMING_FILE"
  echo "--------------------------" >> "$TIMING_FILE"
  echo "CRAIG method finished."

  #############################
  # Method 2: Stochastic Greedy - Facility Location
  #############################
  echo "Running Stochastic Greedy (Facility Location)..."
  echo "Method: Stochastic Greedy (Facility Location)" >> "$TIMING_FILE"
  { time -p python dswa.py \
      --embedding_checkpoint "$EMBEDDING_CHECKPOINT" \
      --subset_size "$SUBSET_SIZE" \
      --method stochastic_greedy \
      --sg_sample_size_Rt 150 \
      --sg_objective_type facility_location \
      --output_json "sg_facility_selected_${SUBSET_SIZE}.json" ; } >> "$TIMING_FILE" 2>> "$TIMING_FILE"
  echo "--------------------------" >> "$TIMING_FILE"
  echo "Stochastic Greedy (Facility Location) method finished."

  #############################
  # Method 3: Stochastic Greedy - Sum of Squared Norms
  #############################
  echo "Running Stochastic Greedy (Sum of Squared Norms)..."
  echo "Method: Stochastic Greedy (Sum of Squared Norms)" >> "$TIMING_FILE"
  { time -p python dswa.py \
      --embedding_checkpoint "$EMBEDDING_CHECKPOINT" \
      --subset_size "$SUBSET_SIZE" \
      --method stochastic_greedy \
      --sg_sample_size_Rt 150 \
      --sg_objective_type sum_sq_norms \
      --output_json "sg_norms_selected_${SUBSET_SIZE}.json" ; } >> "$TIMING_FILE" 2>> "$TIMING_FILE"
  echo "--------------------------" >> "$TIMING_FILE"
  echo "Stochastic Greedy (Sum of Squared Norms) method finished."

  #############################
  # Method 4: PBC
  #############################
  echo "Running PBC (Placeholder)..."
  echo "Method: PBC (Placeholder)" >> "$TIMING_FILE"
  { time -p python dswa.py \
      --embedding_checkpoint "$EMBEDDING_CHECKPOINT" \
      --subset_size "$SUBSET_SIZE" \
      --method pbc \
      --output_json "pbc_random_selected_${SUBSET_SIZE}.json" ; } >> "$TIMING_FILE" 2>> "$TIMING_FILE"
  echo "--------------------------" >> "$TIMING_FILE"
  echo "PBC (Placeholder) method finished."

  echo "All methods completed for SUBSET_SIZE=$SUBSET_SIZE"
done

echo "All subset selection experiments completed."
