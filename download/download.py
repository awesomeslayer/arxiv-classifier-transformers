import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

PROXY_URL = os.getenv("PROXY_URL", "")
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
HF_TOKEN = os.getenv("HF_TOKEN")

if PROXY_URL:
    os.environ["http_proxy"] = PROXY_URL
    os.environ["https_proxy"] = PROXY_URL
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

def download_all_arxiv():
    local_dir = Path(__file__).resolve().parent.parent / "arxiv_data"
    local_dir.mkdir(exist_ok=True)
    
    domains = ["cs", "math", "physics"] #add more if needed
    
    print(f"Syncing to: {local_dir}")
    
    for domain in domains:
        print(f"\n--- Syncing domain: {domain} ---")
        try:
            snapshot_download(
                repo_id="permutans/arxiv-papers-by-subject",
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=f"data/{domain}.*/**/*.parquet", 
                max_workers=8,
                resume_download=True
            )
        except Exception as e:
            print(f"Error syncing {domain}: {e}")
            continue

if __name__ == "__main__":
    download_all_arxiv()