import os
import json
from datasets import load_dataset

# 설정
SELECTION_METHODS = ["craig", "pbc_random", "sg_facility", "sg_norms"]
SIZES = [5176, 10352, 15528, 20704, 25880, 31056, 36232, 41408, 46584]
DATASET_NAME = "yahma/alpaca-cleaned"
DATASET_SPLIT = "train"
INPUT_DIR = "dataset"
OUTPUT_DIR = "recovered_indices"
OUTPUT_SUFFIX = "_indices.json"

# 1. 전체 데이터셋 로드 (한 번만)
print(f"Loading dataset {DATASET_NAME}...")
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
dataset_list = list(dataset)
print(f"Loaded dataset of size: {len(dataset_list)}")

# 2. 루프 돌면서 처리
for method in SELECTION_METHODS:
    for size in SIZES:
        input_json = f"{INPUT_DIR}/{method}_selected_{size}.json"
        output_json = f"{OUTPUT_DIR}/{method}_selected_{size}{OUTPUT_SUFFIX}"

        if not os.path.exists(input_json):
            print(f"Skipping {input_json}: file not found.")
            continue

        print(f"\nProcessing: {input_json}")

        with open(input_json) as f:
            selected_data = json.load(f)

        recovered_indices = []
        for sel in selected_data:
            for idx, full in enumerate(dataset_list):
                if all(sel.get(k, "") == full.get(k, "") for k in ["instruction", "input", "output"]):
                    recovered_indices.append(idx)
                    break

        print(f"Matched {len(recovered_indices)} / {len(selected_data)} samples")

        with open(output_json, "w") as f:
            json.dump(recovered_indices, f)
        print(f"Saved indices to {output_json}")
