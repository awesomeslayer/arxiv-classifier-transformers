import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    return {"accuracy": acc, "f1": f1}

def plot_metrics(trainer_logs, output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    train_loss, eval_loss, eval_acc, steps = [], [], [], []
    
    for log in trainer_logs:
        if "loss" in log:
            train_loss.append(log["loss"])
            steps.append(log["step"])
        if "eval_loss" in log:
            eval_loss.append(log["eval_loss"])
            eval_acc.append(log["eval_accuracy"])

    eval_steps = [log["step"] for log in trainer_logs if "eval_loss" in log]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(steps, train_loss, label='Training Loss', color='tab:blue', linewidth=2)
    if eval_loss:
        ax1.plot(eval_steps, eval_loss, label='Validation Loss', color='tab:orange', linewidth=2, marker='o')
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()

    if eval_acc:
        ax2.plot(eval_steps, eval_acc, label='Validation Accuracy', color='tab:green', linewidth=2, marker='s')
        ax2.set_title('Validation Accuracy over time', fontsize=14)
        ax2.set_xlabel('Steps', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(output_path, dpi=300)
    print(f"Metrics plot saved to {output_path}")