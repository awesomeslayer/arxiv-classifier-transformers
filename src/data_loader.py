import os
import glob
import random
import pandas as pd
from datasets import load_dataset, Dataset, Features, Value, ClassLabel
from transformers import AutoTokenizer

import re

def get_main_category(cat_string):
    if not cat_string:
        return "unknown"
    
    first_cat = str(cat_string).split(',')[0].strip()
    
    if '(' in first_cat and '.' in first_cat:
        main_domain = first_cat.split('.')[0].strip() + ")"
        return main_domain
        
    return first_cat.split('.')[0].strip()

def load_and_prepare_data(model_name: str, max_samples: int = 300000):
    data_dir = "arxiv_data" 
    
    parquet_files = glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True)
    
    if parquet_files:
        print(f"Found {len(parquet_files)} local parquet files. Shuffling and loading...")
        random.seed(42)
        random.shuffle(parquet_files)
        
        dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    else:
        print("Loading dataset from HuggingFace hub (streaming)...")
        dataset = load_dataset("permutans/arxiv-papers-by-subject", streaming=True, split="train")
    
    data = []
    print(f"Collecting up to {max_samples} samples...")
    
    for i, row in enumerate(dataset):            
        if i >= max_samples:
            break
            
        title = row.get('title')
        categories = row.get('subjects') or row.get('primary_subject')
        
        if title and categories:
            title = title.strip()
            abstract = row.get('abstract', '')
            abstract = abstract.strip() if abstract is not None else ''
            
            text = f"{title}. {abstract}" if abstract else title
            label = get_main_category(categories)
            data.append({"text": text, "label": label})
            
    df = pd.DataFrame(data)
    
    if df.empty:
        raise ValueError("DataFrame is empty! Model didn't find 'title' and 'subjects' columns in the data.")
    
    counts = df['label'].value_counts()
    valid_labels = counts[counts >= 50].index
    df = df[df['label'].isin(valid_labels)]
    
    unique_labels = sorted(df['label'].unique().tolist())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    df['label'] = df['label'].map(label2id)
    
    features = Features({
        'text': Value('string'),
        'label': ClassLabel(names=unique_labels)
    })
    
    df = df.reset_index(drop=True)
    
    hf_dataset = Dataset.from_pandas(df, features=features, preserve_index=False)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
    print("Tokenizing data...")
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    
    return tokenized_datasets, label2id, id2label, tokenizer