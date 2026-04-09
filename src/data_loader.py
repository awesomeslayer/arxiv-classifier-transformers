import os
import glob
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def get_main_category(cat_string):
    if not cat_string:
        return "unknown"
    return cat_string.split(',')[0].strip()

def load_and_prepare_data(model_name: str, max_samples: int = 300000):
    data_dir = "arxiv_data"

    parquet_files = glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True)
    
    if parquet_files:
        print(f"Found {len(parquet_files)} local parquet files. Loading...")
        dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    else:
        print("Loading dataset from HuggingFace hub (streaming)...")
        dataset = load_dataset("permutans/arxiv-papers-by-subject", streaming=True, split="train")
    
    data = []
    print(f"Collecting up to {max_samples} samples...")
    for i, row in enumerate(dataset):
        if i >= max_samples:
            break
        if row.get('title') and row.get('categories'):
            title = row.get('title', '').strip()
            abstract = row.get('abstract', '')
            abstract = abstract.strip() if abstract is not None else ''
            
            text = f"{title}. {abstract}" if abstract else title
            label = get_main_category(row['categories'])
            data.append({"text": text, "label": label})
            
    df = pd.DataFrame(data)
    
    counts = df['label'].value_counts()
    valid_labels = counts[counts >= 50].index
    df = df[df['label'].isin(valid_labels)]
    
    unique_labels = sorted(df['label'].unique().tolist())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    df['label'] = df['label'].map(label2id)
    
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
    print("Tokenizing data...")
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    
    return tokenized_datasets, label2id, id2label, tokenizer