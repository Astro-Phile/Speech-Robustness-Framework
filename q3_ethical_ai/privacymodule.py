### BLOCK: IMPORTS ###
import torch
import torchaudio
import os

### BLOCK: SETUP & FOLDERS ###
os.makedirs("q3/examples", exist_ok=True)

### BLOCK: OBFUSCATION MODULE ###
class PrivacyPreservingModule(torch.nn.Module):
    """
    Obfuscates biometric traits (gender, age) while preserving linguistic content.
    Uses Phase Vocoder-based Pitch Shifting to alter vocal tract characteristics 
    without changing the audio duration/speed.
    """
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, waveform, target_profile="female_young"):
        """
        Maps semantic biometric targets to mathematical pitch shifts.
        Standard human vocal folds: 
        - Adult Male: ~85 to 155 Hz
        - Adult Female: ~165 to 255 Hz
        """
        if target_profile == "female_young":
            # Shift pitch UP by 5 semitones to simulate a younger, higher-pitched female voice
            n_steps = 5.0
        elif target_profile == "male_old":
            # Shift pitch DOWN by 4 semitones to simulate an older, deeper male voice
            n_steps = -4.0
        elif target_profile == "anonymous_robot":
            # Extreme shift for absolute privacy (may degrade ASR slightly)
            n_steps = 8.0
        else:
            n_steps = 0.0

        print(f"Applying '{target_profile}' profile (Pitch Shift: {n_steps} semitones)...")
        
        # Initialize the PitchShift transform dynamically based on the target
        pitch_shifter = torchaudio.transforms.PitchShift(
            sample_rate=self.sample_rate,
            n_steps=n_steps
        )
        
        # Apply transformation
        obfuscated_waveform = pitch_shifter(waveform)
        return obfuscated_waveform

### BLOCK: DEMO & DELIVERABLE GENERATION ###
def generate_privacy_examples():
    dataset_path = os.path.join("LibriSpeech_Dataset", "LibriSpeech", "train-clean-100")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Find a single .flac file to use for the demo
    sample_file = None
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".flac"):
                sample_file = os.path.join(root, file)
                break
        if sample_file:
            break

    if not sample_file:
        print("Error: No audio files found to obfuscate.")
        return

    print(f"Loading Source Audio: {sample_file}")
    waveform, sr = torchaudio.load(sample_file)
    
    # Standardize to 16kHz mono
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Initialize the Privacy Module
    obfuscator = PrivacyPreservingModule(sample_rate=16000)

    # Generate Audio Pair 1: Male Old
    out_male = obfuscator(waveform, target_profile="male_old")
    
    # Generate Audio Pair 2: Female Young
    out_female = obfuscator(waveform, target_profile="female_young")

    # Save the deliverables
    orig_path = "q3/examples/01_original_source.wav"
    male_path = "q3/examples/02_obfuscated_male_old.wav"
    female_path = "q3/examples/03_obfuscated_female_young.wav"

    torchaudio.save(orig_path, waveform, 16000)
    torchaudio.save(male_path, out_male, 16000)
    torchaudio.save(female_path, out_female, 16000)

    print("\n" + "="*50)
    print("SUCCESS: Privacy-Preserving Audio Pairs Generated!")
    print(f"Original saved to -> {orig_path}")
    print(f"Male/Old saved to -> {male_path}")
    print(f"Female/Young saved to -> {female_path}")
    print("="*50)
    print("Listen to these files to confirm the linguistic content is preserved while the biometric identity is hidden.")

if __name__ == "__main__":
    generate_privacy_examples()