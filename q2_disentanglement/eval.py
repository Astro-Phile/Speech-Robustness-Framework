### BLOCK: IMPORTS ###
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import the architectures and dataset from your train.py script
from train import BaselineEncoder, DisentangledEncoder, LibriSpeechSpeakerEnvDataset

### BLOCK: EVALUATION PIPELINE ###
def evaluate():
    dataset_path = os.path.join("LibriSpeech_Dataset", "LibriSpeech", "train-clean-100")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("Loading Evaluation Dataset...")
    # Using 10 samples per speaker for a quick evaluation set
    dataset = LibriSpeechSpeakerEnvDataset(dataset_path, num_speakers=5, samples_per_speaker=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    print("Loading Models from Checkpoints...")
    model_base = BaselineEncoder(num_speakers=5)
    model_base.load_state_dict(torch.load('q2/configs/baseline_ckpt.pth'))
    model_base.eval()

    model_dis = DisentangledEncoder(num_speakers=5)
    model_dis.load_state_dict(torch.load('q2/configs/disentangled_ckpt.pth'))
    model_dis.eval()

    # Metrics trackers
    metrics = {
        'base': {'clean_correct': 0, 'noisy_correct': 0},
        'dis': {'clean_correct': 0, 'noisy_correct': 0},
        'total_clean': 0,
        'total_noisy': 0
    }

    print("Running Inference...\n")
    with torch.no_grad():
        for audio, spk_idx, env_idx in dataloader:
            
            # Count environments
            clean_mask = (env_idx == 0)
            noisy_mask = (env_idx == 1)
            metrics['total_clean'] += clean_mask.sum().item()
            metrics['total_noisy'] += noisy_mask.sum().item()

            # Baseline Inference
            base_out = model_base(audio)
            base_preds = torch.argmax(base_out, dim=1)
            metrics['base']['clean_correct'] += (base_preds[clean_mask] == spk_idx[clean_mask]).sum().item()
            metrics['base']['noisy_correct'] += (base_preds[noisy_mask] == spk_idx[noisy_mask]).sum().item()

            # Disentangled Inference
            dis_spk_out, _ = model_dis(audio)
            dis_preds = torch.argmax(dis_spk_out, dim=1)
            metrics['dis']['clean_correct'] += (dis_preds[clean_mask] == spk_idx[clean_mask]).sum().item()
            metrics['dis']['noisy_correct'] += (dis_preds[noisy_mask] == spk_idx[noisy_mask]).sum().item()

    # Calculate Accuracies
    calc_acc = lambda correct, total: (correct / total * 100) if total > 0 else 0.0

    base_clean_acc = calc_acc(metrics['base']['clean_correct'], metrics['total_clean'])
    base_noisy_acc = calc_acc(metrics['base']['noisy_correct'], metrics['total_noisy'])
    
    dis_clean_acc = calc_acc(metrics['dis']['clean_correct'], metrics['total_clean'])
    dis_noisy_acc = calc_acc(metrics['dis']['noisy_correct'], metrics['total_noisy'])

    # Console Output
    print("-" * 50)
    print("EVALUATION RESULTS: BASELINE vs DISENTANGLED")
    print("-" * 50)
    print(f"Baseline     -> Clean Acc: {base_clean_acc:.2f}% | Noisy Acc: {base_noisy_acc:.2f}%")
    print(f"Disentangled -> Clean Acc: {dis_clean_acc:.2f}% | Noisy Acc: {dis_noisy_acc:.2f}%")
    print("-" * 50)

    # Save Results to Text File
    with open('q2/results/eval_metrics.txt', 'w') as f:
        f.write("Baseline Performance:\n")
        f.write(f"Clean Environment Accuracy: {base_clean_acc:.2f}%\n")
        f.write(f"Noisy Environment Accuracy: {base_noisy_acc:.2f}%\n\n")
        f.write("Disentangled Model Performance:\n")
        f.write(f"Clean Environment Accuracy: {dis_clean_acc:.2f}%\n")
        f.write(f"Noisy Environment Accuracy: {dis_noisy_acc:.2f}%\n")

    ### BLOCK: VISUALIZATION ###
    labels = ['Clean Environment', 'Noisy Environment']
    base_scores = [base_clean_acc, base_noisy_acc]
    dis_scores = [dis_clean_acc, dis_noisy_acc]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, base_scores, width, label='Baseline Model', color='salmon')
    ax.bar(x + width/2, dis_scores, width, label='Disentangled Model', color='royalblue')

    ax.set_ylabel('Speaker Recognition Accuracy (%)')
    ax.set_title('Robustness Comparison: Clean vs Noisy Audio')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim([0, 105])
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('q2/results/robustness_comparison_bar.png')
    print("Saved evaluation plot to -> q2/results/robustness_comparison_bar.png")

if __name__ == "__main__":
    evaluate()