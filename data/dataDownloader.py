import os
import urllib.request
import tarfile

def fetch_librispeech_train100():
    url = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
    tar_filename = "train-clean-100.tar.gz"
    extract_folder = "LibriSpeech_Dataset"

    # 1. Download the file if we don't already have it
    if not os.path.exists(tar_filename):
        print(f"Downloading {tar_filename}... (Warning: It's about 6.3 GB, grab a coffee!)")
        urllib.request.urlretrieve(url, tar_filename)
        print("Download complete!")
    else:
        print(f"Found {tar_filename} locally, skipping download.")

    # 2. Extract the file if we haven't already
    extracted_path = os.path.join(extract_folder, "LibriSpeech", "train-clean-100")
    if not os.path.exists(extracted_path):
        print(f"Extracting to '{extract_folder}'...")
        os.makedirs(extract_folder, exist_ok=True)
        with tarfile.open(tar_filename, "r:gz") as tar:
            tar.extractall(path=extract_folder)
        print("Extraction complete! Your dataset is ready.")
    else:
        print(f"Dataset already extracted at '{extracted_path}'.")

if __name__ == "__main__":
    fetch_librispeech_train100()