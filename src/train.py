import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

PROXY_URL = os.getenv("PROXY_URL")
if PROXY_URL:
    os.environ["http_proxy"] = PROXY_URL
    os.environ["https_proxy"] = PROXY_URL
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://huggingface.co")

from transformers import TrainingArguments, Trainer
from data_loader import load_and_prepare_data
from utils import compute_metrics, plot_metrics
from model import get_model

def main():
    parser = argparse.ArgumentParser(description="Train ArXiv Paper Classifier")
    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased", help="HuggingFace model ID")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--samples", type=int, default=100000, help="Number of samples to load")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    print("=== ArXiv Classification Pipeline ===")
    print(f"Model: {args.model_name}")
    print(f"Target Samples: {args.samples}")

    tokenized_datasets, label2id, id2label, tokenizer = load_and_prepare_data(args.model_name, max_samples=args.samples)
    num_labels = len(label2id)
    print(f"Configured {num_labels} classes.")

    model = get_model(args.model_name, num_labels, id2label, label2id)

    output_dir = "outputs/model"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1", 
        bf16=True,                  
        optim="adamw_torch_fused",  
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training phase...")
    trainer.train()

    print("Evaluating on test split...")
    eval_results = trainer.evaluate()
    print(f"Test Metrics: {eval_results}")

    final_model_path = "outputs/best_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Best model saved to {final_model_path}")

    plot_metrics(trainer.state.log_history)

if __name__ == "__main__":
    main()