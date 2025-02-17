import torch
import numpy as np
import os
import sys
from pathlib import Path
import nnAudio.features
import torchaudio

# Add BYOL-A to Python path
project_root = Path(_file_).parent.parent
byol_a_path = project_root / 'byol-a' / 'v2'
if str(byol_a_path) not in sys.path:
    sys.path.append(str(byol_a_path))

# Set audio backend
if 'soundfile' in torchaudio.list_audio_backends():
    torchaudio.set_audio_backend('soundfile')

from byol_a2.common import load_yaml_config
from byol_a2.models import AudioNTT2022, load_pretrained_weights

class AudioNTT2022WithPenultimate(AudioNTT2022):
    """Modified BYOL-A model to access penultimate layer"""
    def forward(self, x):
        # Get penultimate features (before final projection)
        x = self.features(x)  # Extract features up to penultimate layer
        return x  # Return penultimate layer features directly

def extract_penultimate_features(audio_data, model, to_melspec, device, batch_size=32):
    """Extract penultimate layer features from BYOL-A"""
    features = []
    stats = [-9.660292, 4.7219563]
    
    with torch.no_grad():
        for i in range(0, len(audio_data), batch_size):
            batch = torch.from_numpy(audio_data[i:i+batch_size]).float().to(device)
            lms = (to_melspec(batch) + torch.finfo(torch.float).eps).log()
            lms = (lms - stats[0]) / stats[1]
            
            if len(lms.shape) == 3:
                lms = lms.unsqueeze(1)
            
            batch_features = model(lms)
            features.append(batch_features.cpu().numpy())
            
            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"Processed {i + batch_size}/{len(audio_data)} samples")
                print(f"Features shape: {batch_features.shape}")
    
    return np.concatenate(features)

def main():
    try:
        # Load config
        cfg = load_yaml_config(os.path.join(byol_a_path, 'config_v2.yaml'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load preprocessed data
        data_path = r"D:\research_project\preprocessed\train.npz"
        print(f"Loading data from {data_path}")
        data = np.load(data_path)
        X, y = data["X"], data["y"]
        
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
        
        # Load modified BYOL-A model
        print("Loading modified BYOL-A model...")
        model = AudioNTT2022WithPenultimate(n_mels=cfg.n_mels, d=cfg.feature_d).to(device)
        weights_path = os.path.join(byol_a_path, 'AudioNTT2022-BYOLA-64x96d2048.pth')
        load_pretrained_weights(model, weights_path)
        model.eval()
        
        # Extract penultimate features
        print("Extracting penultimate layer features...")
        features = extract_penultimate_features(X, model, to_melspec, device)
        
        # Save features
        output_path = r"D:\research_project\preprocessed\byola_penultimate_features.npz"
        np.savez(
            output_path,
            features=features,
            labels=y,
            genres=data["genres"]
        )
        print(f"Penultimate features saved with shape: {features.shape} to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Available audio backends: {torchaudio.list_audio_backends()}")
        raise

if _name_ == "_main_":
    main()