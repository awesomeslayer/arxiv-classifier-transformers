import os
import sys
import argparse
import random
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.data_loader import get_main_category

def analyze_data(max_samples: int):
    print("=== ArXiv Data Analysis ===")
    print(f"Analyzing up to {max_samples} samples...")

    data_dir = os.path.join(project_root, "arxiv_data")
    parquet_files = glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True)

    if not parquet_files:
        raise FileNotFoundError(
            "No .parquet files found in 'arxiv_data/'. "
            "Please run 'python download/download.py' first."
        )

    random.seed(42)
    random.shuffle(parquet_files)
    
    dataset = load_dataset("parquet", data_files=parquet_files, split="train")

    data = []
    for i, row in enumerate(dataset):
        if i >= max_samples:
            break
        
        title = row.get('title')
        categories = row.get('subjects') or row.get('primary_subject')
        
        if title and categories:
            label = get_main_category(categories)
            data.append({"label": label})
            
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} initial samples.")

    counts = df['label'].value_counts()
    valid_labels = counts[counts >= 50].index
    df_filtered = df[df['label'].isin(valid_labels)].copy() 
    
    print(f"Samples after filtering rare (<50) classes: {len(df_filtered)}")
    num_classes = df_filtered['label'].nunique()
    print(f"Total unique classes: {num_classes}")

    class_counts = df_filtered['label'].value_counts()
    print("\n--- Top 10 Most Common Classes ---")
    print(class_counts.head(10))
    print("\n--- Top 10 Least Common Classes ---")
    print(class_counts.tail(10))

    output_dir = os.path.join(project_root, "outputs/plots")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))
    
    top_n = 25
    sns.barplot(y=class_counts.index[:top_n], x=class_counts.values[:top_n], palette="viridis")
    
    plt.title(f'Distribution of Top {top_n} ArXiv Categories (from {max_samples} samples)', fontsize=16)
    plt.xlabel('Number of Articles', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'data_distribution.png')
    plt.savefig(output_path, dpi=300)
    print(f"\nData distribution plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ArXiv dataset distribution.")
    parser.add_argument(
        "--samples", 
        type=int, 
        default=300000, 
        help="Number of samples to load and analyze, same as in training."
    )
    args = parser.parse_args()
    
    analyze_data(max_samples=args.samples)