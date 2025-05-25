from datasets import Dataset
from pathlib import Path
import re

from huggingface_hub import HfApi

HF_REPO = "selen-kim/selected-dataset"
json_dir = Path(".")

# 전체 임베딩 수 (10% 계산용)
TOTAL_SAMPLES = 51760  

# 정규식으로 메서드명과 샘플 수 추출
pattern = re.compile(r"(craig|sg_facility|sg_norms|pbc_random)_selected_(\d+)\.json")

# Hugging Face 업로드용 API
api = HfApi()

# 처리 루프
for file in sorted(json_dir.glob("*_selected_*.json")):
    match = pattern.match(file.name)
    if not match:
        continue

    method, count_str = match.groups()
    count = int(count_str)
    percent = (count * 100) // TOTAL_SAMPLES

    try:
        ds = Dataset.from_json(str(file))

        # 저장할 parquet 파일 경로 설정
        parquet_filename = f"{percent}%_{count}.parquet"
        parquet_path = Path(f"{method}/{parquet_filename}")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        # parquet 저장
        ds.to_parquet(parquet_path)
        print(f"Saved to {parquet_path}")

        # 업로드
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"{method}/{parquet_filename}",
            repo_id=HF_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded {parquet_path} to {HF_REPO}")
    except Exception as e:
        print(f"Error processing {file.name}: {e}")
