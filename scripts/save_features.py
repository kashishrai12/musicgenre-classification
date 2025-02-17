import torch
import numpy as np
import os
import sys
import nnAudio.features
import torchaudio
from pathlib import Path

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

def main():
    try:
        # Load config
        cfg = load_yaml_config(os.path.join(byol_a_path, 'config_v2.yaml'))
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load preprocessed data
        data_path = r"D:\research_project\preprocessed\train.npz"
        print(f"Loading data from {data_path}")
        data = np.load(data_path)
        X, y = data["X"], data["y"]
        print(f"Loaded data with shape: {X.shape}")
        
        # Initialize mel spectrogram converter
        to_melspec = nnAudio.features.MelSpectrogram(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            center=True,
            power=2,
            verbose=False,
        ).to(device)
        
        # Load BYOL-A model
        print("Loading BYOL-A model...")
        model = AudioNTT2022(n_mels=cfg.n_mels, d=cfg.feature_d).to(device)
        weights_path = os.path.join(byol_a_path, 'AudioNTT2022-BYOLA-64x96d2048.pth')
        load_pretrained_weights(model, weights_path)
        model.eval()
        
        # Print model's expected input shape
        print("Model's first conv layer weight shape:", 
              next(model.parameters()).shape)
        
        # Extract features
        print("Starting feature extraction...")
        features = extract_features(X, model, to_melspec, device)
        
        # Save features
        output_path = r"D:\research_project\preprocessed\byola_features.npz"
        np.savez(
            output_path,
            features=features,
            labels=y,
            genres=data["genres"]
        )
        print(f"Features saved with shape: {features.shape} to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Print available audio backends
        print(f"Available audio backends: {torchaudio.list_audio_backends()}")
        raise

if __name__ == "__main__":
    main()