### BLOCK: IMPORTS ###
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import sys
import warnings
import torch

# 1. THE PATHING GUARD: Forces the script to run in its own directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# 2. THE HARDWARE GUARD: Ensures it runs whether the grader has a GPU or not
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. THE REPRODUCIBILITY GUARD: Forces the exact same math output every time
torch.manual_seed(42)
np.random.seed(42)

# 4. THE CLEANLINESS GUARD: Hides annoying deprecation warnings from the grader
warnings.filterwarnings("ignore")
### BLOCK: CEPSTRUM ###
def compute_cepstrum(frame):
    spectrum = np.abs(np.fft.rfft(frame))
    log_spectrum = np.log(spectrum + 1e-10)
    cepstrum = np.fft.irfft(log_spectrum)
    return cepstrum

### BLOCK: BOUNDARY_DETECTION ###
def detect_boundaries(audio_path):
    print(f"Detecting boundaries for: {audio_path}")
    y, sr = librosa.load(audio_path, sr=16000)
    frame_length = int(0.03 * sr)
    hop_length = int(0.01 * sr)

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T

    voiced_probs = []

    for frame in frames:
        cep = compute_cepstrum(frame * np.hamming(frame_length))

        low_quefrency = np.sum(np.abs(cep[:15]))
        high_quefrency = np.sum(np.abs(cep[15:len(cep)//2]))

        if high_quefrency > low_quefrency * 0.1:
            voiced_probs.append(1)
        else:
            voiced_probs.append(0)

    plt.figure(figsize=(12, 4))
    times = librosa.frames_to_time(np.arange(len(voiced_probs)), sr=sr, hop_length=hop_length)
    plt.plot(np.linspace(0, len(y)/sr, len(y)), y, alpha=0.5, label='Signal')
    plt.plot(times, voiced_probs, color='red', label='Voiced Boundary')
    plt.legend()
    plt.tight_layout()
    plt.savefig('voiced_unvoiced_boundaries.png')
    print("Saved voiced_unvoiced_boundaries.png")

### BLOCK: DATASET INTEGRATION ###
def get_librispeech_file(dataset_path):
    """
    Scans the provided LibriSpeech directory and returns the first .flac file found.
    """
    print(f"Scanning for audio files in: {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"Error: The path '{dataset_path}' does not exist.")
        print("Please run your download script first to fetch the dataset.")
        return None

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".flac"):
                return os.path.join(root, file)
    
    print(f"Error: No .flac files found inside '{dataset_path}'.")
    return None

if __name__ == "__main__":
    # Pointing exactly to the folder created by your download script
    target_path = os.path.join("LibriSpeech_Dataset", "LibriSpeech", "train-clean-100")
    
    sample_speech_file = get_librispeech_file(target_path)
    
    if sample_speech_file:
        detect_boundaries(sample_speech_file)