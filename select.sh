#!/bin/bash

SUBSET_SIZE=12000
EMBEDDING_CHECKPOINT="embeddings_checkpoint_alpaca.npy"
TIMING_FILE="execution_times_${SUBSET_SIZE}.txt"

echo "Algorithm Execution Times" > "$TIMING_FILE"
echo "--------------------------" >> "$TIMING_FILE"
echo "Date: $(date)" >> "$TIMING_FILE"
echo "--------------------------" >> "$TIMING_FILE"

echo "Running CRAIG..."
echo "Method: CRAIG" >> "$TIMING_FILE"
{ time -p python dswa.py \
    --embedding_checkpoint "$EMBEDDING_CHECKPOINT" \
    --subset_size "$SUBSET_SIZE" \
    --method craig \
    --output_json "craig_selected_${SUBSET_SIZE}.json" ; } 2>> "$TIMING_FILE"
echo "--------------------------" >> "$TIMING_FILE"
echo "CRAIG method finished."

echo "Running Stochastic Greedy (Facility Location)..."
echo "Method: Stochastic Greedy (Facility Location)" >> "$TIMING_FILE"
{ time -p python dswa.py \
    --embedding_checkpoint "$EMBEDDING_CHECKPOINT" \
    --subset_size "$SUBSET_SIZE" \
    --method stochastic_greedy \
    --sg_sample_size_Rt 150 \
    --sg_objective_type facility_location \
    --output_json "sg_facility_selected_${SUBSET_SIZE}.json" ; } 2>> "$TIMING_FILE"
echo "--------------------------" >> "$TIMING_FILE"
echo "Stochastic Greedy (Facility Location) method finished."

echo "Running Stochastic Greedy (Sum of Squared Norms)..."
echo "Method: Stochastic Greedy (Sum of Squared Norms)" >> "$TIMING_FILE"
{ time -p python dswa.py \
    --embedding_checkpoint "$EMBEDDING_CHECKPOINT" \
    --subset_size "$SUBSET_SIZE" \
    --method stochastic_greedy \
    --sg_objective_type sum_sq_norms \
    --output_json "sg_norms_selected_${SUBSET_SIZE}.json" ; } 2>> "$TIMING_FILE"
echo "--------------------------" >> "$TIMING_FILE"
echo "Stochastic Greedy (Sum of Squared Norms) method finished."

echo "Running PBC (Placeholder)..."
echo "Method: PBC (Placeholder)" >> "$TIMING_FILE"
{ time -p python dswa.py \
    --embedding_checkpoint "$EMBEDDING_CHECKPOINT" \
    --subset_size "$SUBSET_SIZE" \
    --method pbc \
    --output_json "pbc_random_selected_${SUBSET_SIZE}.json" ; } 2>> "$TIMING_FILE"
echo "--------------------------" >> "$TIMING_FILE"
echo "PBC (Placeholder) method finished."

echo "All methods processed. Timings recorded in $TIMING_FILE"

