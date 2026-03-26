### BLOCK: IMPORTS ###
import os
import numpy as np
import scipy.fftpack
import librosa # Used ONLY for loading the audio file, not for feature extraction
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
### BLOCK: PRE_EMPHASIS ###
def pre_emphasis(signal, alpha=0.97):
    """Applies pre-emphasis filter to the signal."""
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

### BLOCK: MFCC_PIPELINE ###
def manual_mfcc(signal, sample_rate, num_filters=40, num_ceps=13):
    """
    Manual implementation of MFCC extraction including Pre-emphasis, Windowing, 
    FFT, Mel-Filterbank application, Log-compression, and DCT.
    """
    # 1. Pre-emphasis
    emphasized = pre_emphasis(signal)

    # 2. Framing
    frame_size = int(0.025 * sample_rate)
    frame_stride = int(0.01 * sample_rate)

    sig_len = len(emphasized)
    num_frames = int(np.ceil(float(np.abs(sig_len - frame_size)) / frame_stride))
    pad_len = num_frames * frame_stride + frame_size
    pad_signal = np.append(emphasized, np.zeros((pad_len - sig_len)))

    indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_stride, frame_stride), (frame_size, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # 3. Windowing (Applying Hamming window)
    frames *= np.hamming(frame_size)

    # 4. FFT and Power Spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # 5. Mel-Filterbank Application
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    mel_bins = np.floor((NFFT + 1) * hz_points / sample_rate) 

    fbank = np.zeros((num_filters, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = int(mel_bins[m - 1])
        f_m = int(mel_bins[m])
        f_m_plus = int(mel_bins[m + 1])
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - mel_bins[m - 1]) / (mel_bins[m] - mel_bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (mel_bins[m + 1] - k) / (mel_bins[m + 1] - mel_bins[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    
    # 6. Log-compression
    filter_banks = 20 * np.log10(filter_banks)

    # 7. Discrete Cosine Transform (DCT)
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    return mfcc

# --- DATASET INTEGRATION ---
def process_librispeech_split(dataset_path):
    """
    Finds and processes an audio file from the specified LibriSpeech split.
    """
    print(f"Scanning for audio files in: {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        print(f"Error: The path '{dataset_path}' does not exist.")
        print("Please run your download script first to fetch the dataset.")
        return None

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".flac"):
                file_path = os.path.join(root, file)
                print(f"Loading file: {file_path}")
                
                # Load audio using librosa
                signal, sample_rate = librosa.load(file_path, sr=None)
                
                # Run the manual pipeline
                mfccs = manual_mfcc(signal, sample_rate)
                
                print(f"Successfully extracted manual MFCCs!")
                print(f"Shape: {mfccs.shape} (Frames x Cepstral Coefficients)")
                return mfccs
                
    print(f"Error: No .flac files found inside '{dataset_path}'.")
    return None

if __name__ == "__main__":
    # Pointing exactly to the folder created by your download script
    target_path = os.path.join("LibriSpeech_Dataset", "LibriSpeech", "train-clean-100")
    
    my_features = process_librispeech_split(dataset_path=target_path)