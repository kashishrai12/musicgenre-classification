import os
import numpy as np
import librosa
from pathlib import Path

def preprocess_audio(audio_path, sr=16000, duration=10):
    """
    Preprocess a single audio file
    - Resample to 16kHz
    - Convert to mono
    - Trim to 10 seconds
    - Normalize
    """
    # Load audio with resampling
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

def process_gtzan_dataset(data_dir, output_dir):
    """
    Process the entire GTZAN dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize lists to store data and labels
    X = []
    y = []
    genres = []
    
    # Process each genre folder
    for genre_folder in Path(data_dir).glob("*"):
        if genre_folder.is_dir():
            genre = genre_folder.name
            print(f"Processing {genre}...")
            
            # Process each audio file in the genre folder
            for audio_file in genre_folder.glob("*.wav"):
                processed_audio = preprocess_audio(str(audio_file))
                if processed_audio is not None:
                    X.append(processed_audio)
                    y.append(genres.index(genre) if genre in genres else len(genres))
                    if genre not in genres:
                        genres.append(genre)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Save preprocessed data
    np.savez(
        os.path.join(output_dir, "train.npz"),
        X=X,
        y=y,
        genres=genres
    )
    
    print(f"Preprocessed data saved with shape: {X.shape}")
    return X, y, genres

if __name__ == "__main__":
    
    DATA_DIR = r"D:\research_project\dataset\genres" 
    OUTPUT_DIR = r"D:\research_project\preprocessed"
    
    X, y, genres = process_gtzan_dataset(DATA_DIR, OUTPUT_DIR)
