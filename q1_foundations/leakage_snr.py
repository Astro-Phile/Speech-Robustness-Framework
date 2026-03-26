### BLOCK: IMPORTS ###
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import librosa

# 1. THE PATHING GUARD: Forces the script to run in its own directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# 2. THE CLEANLINESS GUARD: Hides annoying Librosa warnings
warnings.filterwarnings("ignore")

### BLOCK: METRICS ###
def calculate_snr(signal, noise_floor=1e-4):
    """
    Calculates SNR. Note: For a more advanced calculation, you could estimate 
    the noise_floor from the silent frames of the audio rather than a constant.
    """
    signal_power = np.mean(signal**2)
    snr = 10 * np.log10(signal_power / noise_floor)
    return snr

def calculate_spectral_leakage(fft_mag):
    """
    Proxy measurement for spectral leakage. 
    Compares the energy in the main spectral peaks vs the rest of the spectrum.
    A higher (less negative) dB value indicates more leakage/spread.
    """
    power_spectrum = fft_mag ** 2
    total_energy = np.sum(power_spectrum)
    
    # Assume the top 5% of frequency bins contain the core speech formants/harmonics
    threshold = np.percentile(power_spectrum, 95)
    main_energy = np.sum(power_spectrum[power_spectrum >= threshold])
    leakage_energy = total_energy - main_energy
    
    # Ratio of leaked energy to main energy in dB
    leakage_db = 10 * np.log10((leakage_energy + 1e-10) / (main_energy + 1e-10))
    return leakage_db

### BLOCK: ANALYSIS ###
def analyze_leakage(audio_path):
    print(f"Analyzing spectral leakage for: {audio_path}")
    # Load a short 1-second segment of the speech file
    y, sr = librosa.load(audio_path, sr=None, duration=1.0)

    windows = {
        'Rectangular': np.ones(len(y)),
        'Hamming': np.hamming(len(y)),
        'Hanning': np.hanning(len(y))
    }

    plt.figure(figsize=(12, 8))

    for i, (name, win) in enumerate(windows.items()):
        y_win = y * win
        fft_out = np.abs(np.fft.rfft(y_win))
        fft_db = 20 * np.log10(fft_out + 1e-10)

        # Calculate metrics
        snr_val = calculate_snr(y_win)
        leakage_val = calculate_spectral_leakage(fft_out)

        # Plotting
        plt.subplot(3, 1, i+1)
        plt.plot(fft_db, color='blue', alpha=0.7)
        plt.title(f'{name} Window | SNR: {snr_val:.2f} dB | Leakage: {leakage_val:.2f} dB')
        plt.ylabel('Magnitude (dB)')
        
        # Add a subtle grid to make reading the DB levels easier
        plt.grid(True, alpha=0.3) 

    plt.xlabel('Frequency Bins')
    plt.tight_layout()
    plt.savefig('spectral_leakage_analysis.png')
    print("Saved spectral_leakage_analysis.png")

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
        analyze_leakage(sample_speech_file)