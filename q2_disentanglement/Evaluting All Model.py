### BLOCK: IMPORTS ###
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os

### BLOCK: DATASET ###
class LibriSpeechSpeakerEnvDataset(Dataset):
    def __init__(self, root_dir, num_speakers=5, samples_per_speaker=10, fixed_length=48000):
        self.data_paths = []
        self.labels = [] 
        self.fixed_length = fixed_length 
        
        all_speaker_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        selected_speakers = all_speaker_folders[:num_speakers]
        self.speaker_to_idx = {spk: i for i, spk in enumerate(selected_speakers)}

        for spk in selected_speakers:
            spk_dir = os.path.join(root_dir, spk)
            count = 0
            for root, dirs, files in os.walk(spk_dir):
                for file in files:
                    if file.endswith('.flac') and count < samples_per_speaker:
                        self.data_paths.append(os.path.join(root, file))
                        env_idx = count % 2 
                        self.labels.append((self.speaker_to_idx[spk], env_idx))
                        count += 1

    def __len__(self): return len(self.data_paths)

    def __getitem__(self, idx):
        path = self.data_paths[idx]
        spk_idx, env_idx = self.labels[idx]
        
        waveform, sr = torchaudio.load(path)
        if sr != 16000: waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
            
        if waveform.shape[1] > self.fixed_length: waveform = waveform[:, :self.fixed_length]
        elif waveform.shape[1] < self.fixed_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.fixed_length - waveform.shape[1]))
            
        if env_idx == 1: waveform = waveform + torch.randn_like(waveform) * 0.05
        return waveform, torch.tensor(spk_idx, dtype=torch.long), torch.tensor(env_idx, dtype=torch.long)

### BLOCK: ALL ARCHITECTURES ###
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0): return GradientReversal.apply(x, alpha)

# 1. Baseline
class BaselineEncoder(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=4), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, stride=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1) 
        )
        self.speaker_classifier = nn.Linear(32, num_speakers)

    def forward(self, x):
        features = self.feature_extractor(x).squeeze(-1)
        return self.speaker_classifier(features)

# 2. Paper's Disentangled
class DisentangledEncoder(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=4), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, stride=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.speaker_classifier = nn.Linear(32, num_speakers)
        self.env_classifier = nn.Linear(32, 2)

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x).squeeze(-1)
        spk_out = self.speaker_classifier(features)
        env_out = self.env_classifier(grad_reverse(features, alpha))
        return spk_out, env_out

# 3. Your Improved Attention Model
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(), nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(x * attn_weights, dim=1), attn_weights

class AttentionDisentangledEncoder(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=4), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, stride=2), nn.ReLU()
        )
        self.temporal_attention = TemporalAttention(hidden_size=32)
        self.speaker_classifier = nn.Linear(32, num_speakers)
        self.env_classifier = nn.Linear(32, 2)

    def forward(self, x, alpha=1.0):
        raw_features = self.feature_extractor(x)
        attended_features, _ = self.temporal_attention(raw_features)
        spk_out = self.speaker_classifier(attended_features)
        env_out = self.env_classifier(grad_reverse(attended_features, alpha))
        return spk_out, env_out

### BLOCK: THE GRAND EVALUATION ###
def evaluate_all():
    dataset_path = os.path.join("LibriSpeech_Dataset", "LibriSpeech", "train-clean-100")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("Loading Evaluation Dataset...")
    dataset = LibriSpeechSpeakerEnvDataset(dataset_path, num_speakers=5, samples_per_speaker=15)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    print("Loading All Models from Checkpoints...")
    
    # Init Models
    model_base = BaselineEncoder(num_speakers=5)
    model_dis = DisentangledEncoder(num_speakers=5)
    model_att = AttentionDisentangledEncoder(num_speakers=5)
    
    # Load Weights
    try:
        model_base.load_state_dict(torch.load('q2/configs/baseline_ckpt.pth'))
        model_dis.load_state_dict(torch.load('q2/configs/disentangled_ckpt.pth'))
        model_att.load_state_dict(torch.load('q2/configs/attention_disentangled_ckpt.pth'))
    except FileNotFoundError as e:
        print(f"Error loading checkpoints: {e}")
        print("Make sure you have run BOTH train.py and train_improved.py first!")
        return

    # Set to Eval mode
    model_base.eval()
    model_dis.eval()
    model_att.eval()

    metrics = {
        'base': {'clean_correct': 0, 'noisy_correct': 0},
        'dis': {'clean_correct': 0, 'noisy_correct': 0},
        'att': {'clean_correct': 0, 'noisy_correct': 0},
        'total_clean': 0, 'total_noisy': 0
    }

    print("Running Inference...\n")
    with torch.no_grad():
        for audio, spk_idx, env_idx in dataloader:
            clean_mask = (env_idx == 0)
            noisy_mask = (env_idx == 1)
            metrics['total_clean'] += clean_mask.sum().item()
            metrics['total_noisy'] += noisy_mask.sum().item()

            # Baseline
            base_preds = torch.argmax(model_base(audio), dim=1)
            metrics['base']['clean_correct'] += (base_preds[clean_mask] == spk_idx[clean_mask]).sum().item()
            metrics['base']['noisy_correct'] += (base_preds[noisy_mask] == spk_idx[noisy_mask]).sum().item()

            # Disentangled
            dis_preds = torch.argmax(model_dis(audio)[0], dim=1)
            metrics['dis']['clean_correct'] += (dis_preds[clean_mask] == spk_idx[clean_mask]).sum().item()
            metrics['dis']['noisy_correct'] += (dis_preds[noisy_mask] == spk_idx[noisy_mask]).sum().item()
            
            # Attention-Disentangled
            att_preds = torch.argmax(model_att(audio)[0], dim=1)
            metrics['att']['clean_correct'] += (att_preds[clean_mask] == spk_idx[clean_mask]).sum().item()
            metrics['att']['noisy_correct'] += (att_preds[noisy_mask] == spk_idx[noisy_mask]).sum().item()

    calc_acc = lambda correct, total: (correct / total * 100) if total > 0 else 0.0

    # Calculate final scores
    scores = {
        'base_clean': calc_acc(metrics['base']['clean_correct'], metrics['total_clean']),
        'base_noisy': calc_acc(metrics['base']['noisy_correct'], metrics['total_noisy']),
        'dis_clean': calc_acc(metrics['dis']['clean_correct'], metrics['total_clean']),
        'dis_noisy': calc_acc(metrics['dis']['noisy_correct'], metrics['total_noisy']),
        'att_clean': calc_acc(metrics['att']['clean_correct'], metrics['total_clean']),
        'att_noisy': calc_acc(metrics['att']['noisy_correct'], metrics['total_noisy']),
    }

    print("=" * 60)
    print("FINAL EVALUATION RESULTS: 3-WAY ARCHITECTURE COMPARISON")
    print("=" * 60)
    print(f"1. Baseline CNN         -> Clean: {scores['base_clean']:.2f}% | Noisy: {scores['base_noisy']:.2f}%")
    print(f"2. Paper's Disentangled -> Clean: {scores['dis_clean']:.2f}% | Noisy: {scores['dis_noisy']:.2f}%")
    print(f"3. Improved Attention   -> Clean: {scores['att_clean']:.2f}% | Noisy: {scores['att_noisy']:.2f}%")
    print("=" * 60)

    # Save Results to Text File
    with open('q2/results/final_3way_metrics.txt', 'w') as f:
        f.write("=== FINAL ARCHITECTURE COMPARISON ===\n\n")
        f.write(f"Baseline CNN:\n  Clean: {scores['base_clean']:.2f}%\n  Noisy: {scores['base_noisy']:.2f}%\n\n")
        f.write(f"Paper Disentangled:\n  Clean: {scores['dis_clean']:.2f}%\n  Noisy: {scores['dis_noisy']:.2f}%\n\n")
        f.write(f"Improved Attention:\n  Clean: {scores['att_clean']:.2f}%\n  Noisy: {scores['att_noisy']:.2f}%\n")

    ### BLOCK: 3-WAY VISUALIZATION ###
    labels = ['Clean Environment', 'Noisy Environment']
    base_bars = [scores['base_clean'], scores['base_noisy']]
    dis_bars = [scores['dis_clean'], scores['dis_noisy']]
    att_bars = [scores['att_clean'], scores['att_noisy']]

    x = np.arange(len(labels))
    width = 0.25 # Thinner bars to fit 3 per group

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Offset the bars so they group together nicely
    ax.bar(x - width, base_bars, width, label='1. Baseline CNN', color='lightcoral')
    ax.bar(x, dis_bars, width, label="2. Paper's Disentangled", color='royalblue')
    ax.bar(x + width, att_bars, width, label='3. Improved Attention', color='mediumseagreen')

    ax.set_ylabel('Speaker Recognition Accuracy (%)', fontsize=12)
    ax.set_title('Architecture Robustness Comparison: Clean vs Noisy Data', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 110]) # Extra space for the legend
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('q2/results/final_3way_comparison_bar.png', dpi=300)
    print("Saved ultimate comparison plot to -> q2/results/final_3way_comparison_bar.png")

if __name__ == "__main__":
    evaluate_all()