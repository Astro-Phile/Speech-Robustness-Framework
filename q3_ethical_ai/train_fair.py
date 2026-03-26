### BLOCK: IMPORTS ###
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Pathing Guard (Allows Python to see your other files)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# 2. Import the Dataset and Model directly from your Question 2 script!
# (We alias them here so the names match the Q3 Fairness context)
from train import LibriSpeechSpeakerEnvDataset as LibriSpeechFairnessDataset
from train import BaselineEncoder as ProxyAcousticModel

# 3. Setup Q3 Folders
os.makedirs("q3/results", exist_ok=True)
os.makedirs("q3/configs", exist_ok=True)

### BLOCK: FAIRNESS LOSS ###
class FairnessLoss(nn.Module):
    def __init__(self, lambda_fair=0.5):
        super().__init__()
        self.base_loss = nn.CrossEntropyLoss(reduction='mean')
        self.lambda_fair = lambda_fair

    def forward(self, logits, targets, group_labels):
        standard_loss = self.base_loss(logits, targets)
        losses = nn.functional.cross_entropy(logits, targets, reduction='none')

        group_0_mask = (group_labels == 0).float()
        group_1_mask = (group_labels == 1).float()

        loss_g0 = (losses * group_0_mask).sum() / (group_0_mask.sum() + 1e-8)
        loss_g1 = (losses * group_1_mask).sum() / (group_1_mask.sum() + 1e-8)

        fair_penalty = torch.abs(loss_g0 - loss_g1)
        total_loss = standard_loss + (self.lambda_fair * fair_penalty)
        return total_loss, standard_loss, fair_penalty

### BLOCK: SMART FAIRNESS TRAINING LOOP ###
def train_fair_model():
    dataset_path = os.path.join("LibriSpeech_Dataset", "LibriSpeech", "train-clean-100")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("Initializing Fairness-Aware Training...")
    
    # Using the imported Q2 dataset
    dataset = LibriSpeechFairnessDataset(dataset_path, num_speakers=5, samples_per_speaker=40)
    
    # The DataLoader missing from your IDE screenshot is now properly imported!
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Using the imported Q2 baseline model
    model = ProxyAcousticModel(num_classes=5)
    
    # The optim missing from your IDE screenshot is now properly imported!
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    criterion = FairnessLoss(lambda_fair=2.0)

    epochs = 100 
    patience = 8
    best_loss = float('inf')
    epochs_no_improve = 0

    history_standard = []
    history_fairness = []
    history_total = []

    print("\nStarting Smart Training Loop...")
    for epoch in range(epochs):
        epoch_total = 0
        epoch_std = 0
        epoch_fair = 0
        
        for audio, targets, groups in dataloader:
            optimizer.zero_grad()
            
            logits = model(audio)
            total_loss, std_loss, fair_pen = criterion(logits, targets, groups)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_total += total_loss.item()
            epoch_std += std_loss.item()
            epoch_fair += fair_pen.item()

        n_batches = len(dataloader)
        current_total = epoch_total / n_batches
        
        history_total.append(current_total)
        history_standard.append(epoch_std / n_batches)
        history_fairness.append(epoch_fair / n_batches)
        
        scheduler.step(current_total)

        print(f"Epoch {epoch+1:02d} | Total: {current_total:.4f} | Std Loss: {history_standard[-1]:.4f} | Fair Gap: {history_fairness[-1]:.4f}")

        if current_total < best_loss:
            best_loss = current_total
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'q3/configs/fair_model_ckpt.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n--- Early stopping triggered at Epoch {epoch+1}! ---")
                print("Model has closed the fairness gap as much as possible.")
                break

    plt.figure(figsize=(10, 6))
    plt.plot(history_total, label='Total Loss', color='black', linewidth=2)
    plt.plot(history_standard, label='Standard Classification Loss', color='blue', linestyle='--')
    plt.plot(history_fairness, label='Fairness Penalty (Demographic Gap)', color='red', linestyle='-.')
    
    plt.title('Fairness-Aware Training Dynamics (Early Stopping)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('q3/results/fairness_training_loss.png')
    
    print("\nSUCCESS: Training Complete.")

if __name__ == "__main__":
    train_fair_model()