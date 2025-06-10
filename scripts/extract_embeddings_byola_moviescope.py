import os
import numpy as np
import torch
import torchaudio
import nnAudio.features
from pathlib import Path
import sys
import gc
import psutil
import time
import warnings
import librosa
warnings.filterwarnings("ignore")


MAX_RETRIES = 3
MIN_MEMORY = 1 * 1024**3  
FILE_BATCH_SIZE = 1        
CHUNK_DURATION = 30       

def get_memory_status():
    mem = psutil.virtual_memory()
    return {
        'available': mem.available,
        'used_percent': mem.percent,
        'free': mem.free
    }

def wait_for_memory(min_available=MIN_MEMORY, timeout=60):
    """Wait until minimum memory is available"""
    start_time = time.time()
    while True:
        mem = get_memory_status()
        if mem['available'] > min_available:
            return True
        if time.time() - start_time > timeout:
            print(f"Timeout waiting for memory (Avail: {mem['available']/1024**2:.1f}MB)")
            return False
        print(f"Waiting for memory... (Avail: {mem['available']/1024**2:.1f}MB, Used: {mem['used_percent']}%)")
        time.sleep(10)

def process_single_chunk(chunk, model, to_melspec, device):
    """Process one small chunk of audio"""
    stats = [-9.660292, 4.7219563]
    try:
        chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        lms = (to_melspec(chunk_tensor) + torch.finfo(torch.float).eps).log()
        lms = (lms - stats[0]) / stats[1]
        if len(lms.shape) == 3:
            lms = lms.unsqueeze(1)
        features = model(lms).cpu().detach().numpy()  
        return features
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        return None
    finally:
        del chunk_tensor, lms
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def process_file_in_parts(filepath, model, to_melspec, device, sr=16000):
    """Process a file in small temporal parts"""
    try:
        # Get file duration without loading
        duration = torchaudio.info(filepath).num_frames / sr
        
        features = []
        for start in np.arange(0, duration, CHUNK_DURATION):
            if not wait_for_memory():
                print("Insufficient memory for next chunk")
                break
                
            # Load just this segment
            waveform, _ = torchaudio.load(
                filepath,
                frame_offset=int(start * sr),
                num_frames=int(CHUNK_DURATION * sr)
            )
            
            # Convert to mono and normalize
            chunk = waveform.mean(dim=0).numpy().astype(np.float32)
            chunk = librosa.util.normalize(chunk)
            
            # Pad if needed
            if len(chunk) < CHUNK_DURATION * sr:
                chunk = np.pad(chunk, (0, CHUNK_DURATION * sr - len(chunk)))
                
            # Process chunk
            chunk_features = process_single_chunk(chunk, model, to_melspec, device)
            if chunk_features is not None:
                features.append(chunk_features)
                
            # Clean up
            del waveform, chunk
            gc.collect()
            
        return np.vstack(features) if features else None
        
    except Exception as e:
        print(f"Error processing {Path(filepath).name}: {str(e)}")
        return None

def main():
    print("=== Starting Extreme Memory-Safe Feature Extraction ===")
    
    # Configuration
    wav_dir = r"D:\research_project\moviescope_dataset\audio_wav"
    output_folder = r"D:\research_project\moviescope_byola_embeddings"
    os.makedirs(output_folder, exist_ok=True)
    
    # Start index for processing 
    start_idx = 2300 + (10 * 2)
    
    # Load model
    try:
        project_root = Path(__file__).parent.parent
        byol_a_path = project_root / 'byol-a' / 'v2'
        sys.path.insert(0, str(byol_a_path))
        
        from byol_a2.common import load_yaml_config
        from byol_a2.models import AudioNTT2022, load_pretrained_weights
        
        cfg = load_yaml_config(os.path.join(byol_a_path, 'config_v2.yaml'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        
        # Initialize model components
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
        
        print("Loading BYOL-A model...")
        model = AudioNTT2022(n_mels=cfg.n_mels, d=cfg.feature_d).to(device)
        load_pretrained_weights(model, os.path.join(byol_a_path, 'AudioNTT2022-BYOLA-64x96d2048.pth'))
        model.eval()
        
    except Exception as e:
        print(f"\nFailed to initialize model: {str(e)}")
        return

    # Process files
    wav_files = sorted(list(Path(wav_dir).glob("*.wav")))[start_idx:]
    print(f"\nFound {len(wav_files)} files to process starting from index {start_idx}")
    
    # Process one file at a time with retries
    processed_count = 0
    for file_idx, file in enumerate(wav_files):
        output_path = os.path.join(output_folder, f"file_{start_idx + file_idx}.npz")
        
        if os.path.exists(output_path):
            print(f"\nFile {start_idx + file_idx} exists, skipping")
            processed_count += 1
            continue
            
        print(f"\n=== Processing file {start_idx + file_idx}/{start_idx + len(wav_files)-1} ===")
        print(f"Filename: {file.name}")
        
        for attempt in range(MAX_RETRIES):
            if not wait_for_memory():
                print("Memory unavailable, retrying...")
                time.sleep(30)
                continue
                
            features = process_file_in_parts(str(file), model, to_melspec, device)
            if features is not None:
                try:
                    np.savez(
                        output_path,
                        features=features,
                        filename=file.name,
                        movie_id=file.stem,
                        chunk_indices=np.arange(len(features))
                    )
                    processed_count += 1
                    print(f"Successfully processed {file.name}")
                    break
                except Exception as e:
                    print(f"Error saving {file.name}: {str(e)}")
            else:
                print(f"Failed to process {file.name} (attempt {attempt + 1}/{MAX_RETRIES})")
            
            # Clean up before retry
            gc.collect()
            time.sleep(10)
            
        # Report progress periodically
        if (file_idx + 1) % 10 == 0:
            print(f"\nProgress: {processed_count}/{len(wav_files)} files processed")
            
    print(f"\n=== Completed processing {processed_count}/{len(wav_files)} files ===")

if __name__ == "__main__":
    main()