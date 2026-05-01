# scripts/download_hf_dataset.py
import pandas as pd
import os

# Using the URL identified during research
url = "hf://datasets/dhyann2815/india-crop-yield-prediction/data/train-00000-of-00001.parquet"
print(f"Downloading dataset from Hugging Face...")

try:
    # Environment has pyarrow, which pandas uses for parquet
    df = pd.read_parquet(url)

    # Save to data/raw
    output_path = os.path.join("data", "raw", "india_crop_yield.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path} with {len(df)} rows.")
except Exception as e:
    print(f"Error downloading or saving dataset: {e}")
    print("Trying alternative method: using huggingface_hub if available...")
    try:
        from huggingface_hub import hf_hub_download
        file_path = hf_hub_download(repo_id="dhyann2815/india-crop-yield-prediction", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(file_path)
        output_path = os.path.join("data", "raw", "india_crop_yield.csv")
        df.to_csv(output_path, index=False)
        print(f"Dataset successfully saved to {output_path} with {len(df)} rows.")
    except Exception as e2:
        print(f"Alternative method failed: {e2}")
