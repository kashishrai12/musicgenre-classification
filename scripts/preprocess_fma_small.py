import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

def preprocess_audio(audio_path, sr=22050, duration=30):
    """
    Preprocess a single audio file:
    - Convert to mono
    - Trim to 30 seconds
    - Normalize
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
        
        # Ensure exact length (pad or trim)
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
            
        # Normalize
        y = librosa.util.normalize(y)
        
        return y
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def preprocess_fma(data_dir, output_dir, sr=22050, duration=30):
    """
    Process the entire fma_small dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(data_dir):
        for file in tqdm(files):
            if file.endswith('.mp3'):
                audio_path = os.path.join(root, file)
                y = preprocess_audio(audio_path, sr=sr, duration=duration)
                if y is not None:
                    output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.npy")
                    np.save(output_path, y)

if __name__ == "__main__":
    data_dir = r"D:\research_project\dataset2\fma_small"
    output_dir = r"D:\research_project\dataset2\fma_small_preprocessed"
    preprocess_fma(data_dir, output_dir)