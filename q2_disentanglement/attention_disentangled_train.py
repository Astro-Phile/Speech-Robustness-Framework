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
    def __init__(self, root_dir, num_speakers=5, samples_per_speaker=40, fixed_length=48000):
        self.data_paths = []
        self.labels = [] 
        self.fixed_length = fixed_length 
        
        print(f"Scanning '{root_dir}' for speakers...")
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

### BLOCK: NEW ARCHITECTURE (TEMPORAL ATTENTION) ###
class TemporalAttention(nn.Module):
    """
    Replaces standard Average Pooling. Learns to weigh important 
    speech frames heavily and ignore pure noise/silence frames.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # x shape from CNN: (batch_size, channels, time_steps)
        # Transpose to: (batch_size, time_steps, channels) for Linear layer
        x = x.transpose(1, 2)
        
        # Calculate attention weights for each time step
        attn_weights = torch.softmax(self.attention(x), dim=1)
        
        # Multiply features by weights and sum across time
        weighted_features = torch.sum(x * attn_weights, dim=1)
        
        return weighted_features, attn_weights

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0): return GradientReversal.apply(x, alpha)

class AttentionDisentangledEncoder(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        # Notice we removed the AdaptiveAvgPool1d from the end
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=4), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, stride=2), nn.ReLU()
        )
        
        # Insert our custom improvement here
        self.temporal_attention = TemporalAttention(hidden_size=32)
        
        self.speaker_classifier = nn.Linear(32, num_speakers)
        self.env_classifier = nn.Linear(32, 2)

    def forward(self, x, alpha=1.0):
        # 1. Extract raw sequential features
        raw_features = self.feature_extractor(x)
        
        # 2. Apply Attention Mechanism (This is the core improvement)
        attended_features, attn_weights = self.temporal_attention(raw_features)
        
        # 3. Classify Speaker
        spk_out = self.speaker_classifier(attended_features)

        # 4. Reverse Gradients and Classify Environment
        rev_features = grad_reverse(attended_features, alpha)
        env_out = self.env_classifier(rev_features)

        return spk_out, env_out

### BLOCK: TRAINING LOOP ###
def train_improved_model():
    dataset_path = os.path.join("LibriSpeech_Dataset", "LibriSpeech", "train-clean-100")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("Initializing Data and Advanced Attention Model...")
    dataset = LibriSpeechSpeakerEnvDataset(dataset_path, num_speakers=5, samples_per_speaker=40)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize our newly improved model
    model = AttentionDisentangledEncoder(num_speakers=5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    epochs = 100 
    patience = 8 
    best_loss = float('inf')
    epochs_no_improve = 0
    training_losses = []

    print("\nStarting Training for Attention-Disentangled Model...")
    for epoch in range(epochs):
        epoch_loss = 0
        
        for audio, spk_idx, env_idx in dataloader:
            optimizer.zero_grad()
            
            spk_out, env_out = model(audio)
            
            loss_spk = criterion(spk_out, spk_idx)
            loss_env = criterion(env_out, env_idx)
            
            # Minimize speaker loss, maximize environment confusion
            loss = loss_spk + (0.5 * loss_env) 
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss_spk.item() 

        current_loss = epoch_loss / len(dataloader)
        training_losses.append(current_loss)
        scheduler.step(current_loss)

        print(f"Epoch {epoch+1:02d} | Speaker Loss: {current_loss:.4f}")

        if current_loss < best_loss:
            best_loss = current_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'q2/configs/attention_disentangled_ckpt.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n--- Early stopping triggered at Epoch {epoch+1}! ---")
                break

    # Save visualization of the improved training
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Attention-Disentangled Model', color='purple')
    plt.title('Training Loss: Improved Attention Mechanism')
    plt.xlabel('Epochs')
    plt.ylabel('Speaker Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('q2/results/improved_training_loss.png')
    
    print("\nSaved improved checkpoint to -> q2/configs/attention_disentangled_ckpt.pth")
    print("Saved loss plot to -> q2/results/improved_training_loss.png")

if __name__ == "__main__":
    train_improved_model()
