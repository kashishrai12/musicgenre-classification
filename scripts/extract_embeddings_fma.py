import torch
import numpy as np
import os
import sys
import nnAudio.features
import torchaudio
from pathlib import Path
import pandas as pd

# Add BYOL-A to Python path - adjust this to your project structure
project_root = Path(__file__).parent.parent  # Gets D:/research_project
byol_a_path = project_root / 'byol-a' / 'v2'
if str(byol_a_path) not in sys.path:
    sys.path.append(str(byol_a_path))

# Set audio backend before importing BYOL-A modules
if 'soundfile' in torchaudio.list_audio_backends():
    torchaudio.set_audio_backend('soundfile')

# Now import BYOL-A modules
from byol_a2.common import load_yaml_config
from byol_a2.models import AudioNTT2022, load_pretrained_weights

def extract_features(audio_data, model, to_melspec, device, batch_size=32):
    """Extract BYOL-A embeddings from preprocessed audio data"""
    features = []
    stats = [-9.660292, 4.7219563]  # From BYOL-A paper for audio normalization
    
    with torch.no_grad():
        for i in range(0, len(audio_data), batch_size):
            # Get batch
            batch = torch.from_numpy(audio_data[i:i+batch_size]).float().to(device)
            
            # Convert to log-mel spectrogram
            lms = (to_melspec(batch) + torch.finfo(torch.float).eps).log()
            
            # Normalize
            lms = (lms - stats[0]) / stats[1]
            
            # Reshape to match expected input shape [batch_size, channels=1, n_mels, time]
            if len(lms.shape) == 3:
                lms = lms.unsqueeze(1)  # Add channel dimension
            
            # Extract features
            batch_features = model(lms)
            features.append(batch_features.cpu().numpy())
            
            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"Processed {i + batch_size}/{len(audio_data)} samples")
                print(f"Input shape: {lms.shape}, Output shape: {batch_features.shape}")
    
    return np.concatenate(features)

def extract_and_save_features(split_csv, preprocessed_dir, output_npz, model, to_melspec, device):
    # Load the split CSV file
    split_df = pd.read_csv(split_csv, index_col=0, header=[0, 1])
    
    features = []
    filenames = []
    missing_files = []
    
    for track_id in split_df.index:
        track_id_str = str(track_id).zfill(6)
        preprocessed_path = os.path.join(preprocessed_dir, f"{track_id_str}.npy")
        
        if os.path.exists(preprocessed_path):
            try:
                audio_data = np.load(preprocessed_path)
                file_features = extract_features(audio_data[np.newaxis, :], model, to_melspec, device)
                features.append(file_features)
                filenames.append(f"{track_id_str}.npy")
            except Exception as e:
                print(f"Error processing {preprocessed_path}: {e}")
        else:
            missing_files.append(preprocessed_path)
            print(f"Warning: {preprocessed_path} not found")
    
    if features:
        features = np.concatenate(features)
        filenames = np.array(filenames)
        
        # Save features and filenames to NPZ file
        np.savez(output_npz, features=features, filenames=filenames)
        print(f"Features saved to {output_npz}")
    else:
        print(f"No features extracted for {split_csv}")
    
    if missing_files:
        print(f"Missing files: {missing_files}")

def main():
    try:
        # Load config
        cfg = load_yaml_config(os.path.join(byol_a_path, 'config_v2.yaml'))
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load preprocessed data directory
        preprocessed_dir = r"D:\research_project\dataset2\fma_small_preprocessed"
        
        # Load BYOL-A model
        print("Loading BYOL-A model...")
        model = AudioNTT2022(n_mels=cfg['n_mels'], d=cfg['feature_d']).to(device)
        weights_path = os.path.join(byol_a_path, 'AudioNTT2022-BYOLA-64x96d2048.pth')
        load_pretrained_weights(model, weights_path)
        model.eval()
        
        # Print model's expected input shape
        print("Model's first conv layer weight shape:", 
              next(model.parameters()).shape)
        
        # Initialize mel spectrogram converter
        to_melspec = nnAudio.features.MelSpectrogram(
            sr=cfg['sample_rate'],
            n_fft=cfg['n_fft'],
            win_length=cfg['win_length'],
            hop_length=cfg['hop_length'],
            n_mels=cfg['n_mels'],
            fmin=cfg['f_min'],
            fmax=cfg['f_max'],
            center=True,
            power=2,
            verbose=False,
        ).to(device)
        
        # Extract and save features for train and test splits
        train_csv = r"D:\research_project\dataset2\train_split.csv"
        test_csv = r"D:\research_project\dataset2\test_split.csv"
        train_output_npz = r"D:\research_project\dataset2\byola_features_train.npz"
        test_output_npz = r"D:\research_project\dataset2\byola_features_test.npz"
        
        extract_and_save_features(train_csv, preprocessed_dir, train_output_npz, model, to_melspec, device)
        extract_and_save_features(test_csv, preprocessed_dir, test_output_npz, model, to_melspec, device)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        # Print available audio backends
        print(f"Available audio backends: {torchaudio.list_audio_backends()}")
        raise

if __name__ == "__main__":
    main()