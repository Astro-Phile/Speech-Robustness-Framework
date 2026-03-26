### BLOCK: IMPORTS ###
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os

### BLOCK: SETUP & FOLDERS ###
os.makedirs("q2/results", exist_ok=True)
os.makedirs("q2/configs", exist_ok=True)

### BLOCK: LIBRISPEECH DATASET ###
class LibriSpeechSpeakerEnvDataset(Dataset):
    """
    Loads actual LibriSpeech audio, maps Speaker IDs, and simulates 
    two domains (Clean vs. Noisy) for Disentangled Representation Learning.
    """
    def __init__(self, root_dir, num_speakers=5, samples_per_speaker=40, fixed_length=48000):
        self.data_paths = []
        self.labels = [] # Stores tuples of (speaker_idx, env_idx)
        self.fixed_length = fixed_length # 3 seconds of audio at 16kHz
        
        print(f"Scanning '{root_dir}' for speakers...")
        # LibriSpeech structure: train-clean-100/SPEAKER_ID/CHAPTER_ID/file.flac
        all_speaker_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        selected_speakers = all_speaker_folders[:num_speakers]
        
        self.speaker_to_idx = {spk: i for i, spk in enumerate(selected_speakers)}
        self.num_speakers = num_speakers

        for spk in selected_speakers:
            spk_dir = os.path.join(root_dir, spk)
            count = 0
            for root, dirs, files in os.walk(spk_dir):
                for file in files:
                    if file.endswith('.flac') and count < samples_per_speaker:
                        self.data_paths.append(os.path.join(root, file))
                        # Even count = Clean (Env 0), Odd count = Noisy (Env 1)
                        env_idx = count % 2 
                        self.labels.append((self.speaker_to_idx[spk], env_idx))
                        count += 1

        print(f"Loaded {len(self.data_paths)} audio files across {num_speakers} speakers.")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        path = self.data_paths[idx]
        spk_idx, env_idx = self.labels[idx]
        
        # Load audio using torchaudio
        waveform, sr = torchaudio.load(path)
        
        # Resample to 16kHz if necessary
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Pad or Crop to exactly `fixed_length` (48,000 samples = 3 seconds)
        if waveform.shape[1] > self.fixed_length:
            waveform = waveform[:, :self.fixed_length]
        elif waveform.shape[1] < self.fixed_length:
            pad_amount = self.fixed_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        # Apply the "Environment" challenge
        if env_idx == 1:
            # Inject white noise to simulate a noisy environment
            noise = torch.randn_like(waveform) * 0.05
            waveform = waveform + noise

        return waveform, torch.tensor(spk_idx, dtype=torch.long), torch.tensor(env_idx, dtype=torch.long)

### BLOCK: ARCHITECTURE ###
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

class BaselineEncoder(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) 
        )
        self.speaker_classifier = nn.Linear(32, num_speakers)

    def forward(self, x):
        features = self.feature_extractor(x).squeeze(-1)
        spk_out = self.speaker_classifier(features)
        return spk_out

class DisentangledEncoder(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.speaker_classifier = nn.Linear(32, num_speakers)
        self.env_classifier = nn.Linear(32, 2)

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x).squeeze(-1)
        spk_out = self.speaker_classifier(features)

        rev_features = grad_reverse(features, alpha)
        env_out = self.env_classifier(rev_features)

        return spk_out, env_out

### BLOCK: TRAINING ###
def train():
    dataset_path = os.path.join("LibriSpeech_Dataset", "LibriSpeech", "train-clean-100")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    dataset = LibriSpeechSpeakerEnvDataset(dataset_path, num_speakers=5, samples_per_speaker=40)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model_base = BaselineEncoder(num_speakers=5)
    model_disentangled = DisentangledEncoder(num_speakers=5)
    
    opt_base = optim.Adam(model_base.parameters(), lr=0.001)
    opt_dis = optim.Adam(model_disentangled.parameters(), lr=0.001)
    
    # Schedulers to reduce learning rate when the loss plateaus
    scheduler_base = optim.lr_scheduler.ReduceLROnPlateau(opt_base, mode='min', factor=0.5, patience=3)
    scheduler_dis = optim.lr_scheduler.ReduceLROnPlateau(opt_dis, mode='min', factor=0.5, patience=3)
    
    criterion = nn.CrossEntropyLoss()

    epochs = 100 # High ceiling, we will rely on early stopping
    patience = 8 # Number of epochs to wait for improvement before stopping
    best_dis_loss = float('inf')
    epochs_no_improve = 0

    base_spk_losses = []
    dis_spk_losses = []

    print("\nStarting Smart Training Loop...")
    for epoch in range(epochs):
        base_epoch_loss = 0
        dis_epoch_loss = 0
        
        for audio, spk_idx, env_idx in dataloader:
            # Train Baseline
            opt_base.zero_grad()
            out_base = model_base(audio)
            loss_base = criterion(out_base, spk_idx)
            loss_base.backward()
            opt_base.step()
            base_epoch_loss += loss_base.item()

            # Train Disentangled
            opt_dis.zero_grad()
            spk_out, env_out = model_disentangled(audio)
            loss_spk = criterion(spk_out, spk_idx)
            loss_env = criterion(env_out, env_idx)
            
            # The disentanglement penalty
            loss_dis = loss_spk + (0.5 * loss_env) 
            loss_dis.backward()
            opt_dis.step()
            dis_epoch_loss += loss_spk.item() 

        current_base_loss = base_epoch_loss / len(dataloader)
        current_dis_loss = dis_epoch_loss / len(dataloader)
        
        base_spk_losses.append(current_base_loss)
        dis_spk_losses.append(current_dis_loss)
        
        # Step the schedulers
        scheduler_base.step(current_base_loss)
        scheduler_dis.step(current_dis_loss)

        print(f"Epoch {epoch+1:02d} | Baseline Spk Loss: {current_base_loss:.4f} | Disentangled Spk Loss: {current_dis_loss:.4f}")

        # Early Stopping Logic (tracking the disentangled model)
        if current_dis_loss < best_dis_loss:
            best_dis_loss = current_dis_loss
            epochs_no_improve = 0
            # Save the *best* models exactly when they peak
            torch.save(model_base.state_dict(), 'q2/configs/baseline_ckpt.pth')
            torch.save(model_disentangled.state_dict(), 'q2/configs/disentangled_ckpt.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n--- Early stopping triggered at Epoch {epoch+1}! ---")
                print("Model has converged and stopped improving. Peak weights have been saved.")
                break

    # Plot Comparison using however many epochs it actually ran
    plt.figure(figsize=(10, 6))
    plt.plot(base_spk_losses, label='Baseline', color='red', linestyle='--')
    plt.plot(dis_spk_losses, label='Disentangled', color='blue')
    plt.title('Training Loss (Early Stopping Enabled)')
    plt.xlabel('Epochs')
    plt.ylabel('Speaker Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('q2/results/training_comparison.png')
    print("\nSaved comparison plot to -> q2/results/training_comparison.png")
    print("Saved best model checkpoints to -> q2/configs/")

if __name__ == "__main__":
    train()