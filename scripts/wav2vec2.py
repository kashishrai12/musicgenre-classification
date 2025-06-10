import os
import glob
import torch
import numpy as np
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load wav2vec2 base model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

def get_wav2vec2_embedding(audio_path, processor, model, target_sr=16000):
    try:
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=target_sr)
        # Ensure mono
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0)

        inputs = processor(waveform, sampling_rate=target_sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values)
        # Mean pooling over time dimension
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
    except Exception as e:
        print(f"Skipping {audio_path}: {e}")
        return None

def main():
    audio_dir = "dataset/genres"
    embeddings = []
    labels = []
    files = []

    genre_folders = [f for f in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, f))]
    for genre in tqdm(genre_folders, desc="Genres"):
        genre_path = os.path.join(audio_dir, genre)
        audio_files = glob.glob(os.path.join(genre_path, "*.wav"))
        for audio_file in tqdm(audio_files, desc=f"{genre}", leave=False):
            emb = get_wav2vec2_embedding(audio_file, processor, model)
            if emb is not None:
                embeddings.append(emb)
                labels.append(genre)
                files.append(audio_file)

    embeddings = np.stack(embeddings)
    labels = np.array(labels)
    files = np.array(files)
    np.savez("gtzan_wav2vec2_embeddings.npz", X=embeddings, y=labels, files=files)
    print(f"Saved embeddings to gtzan_wav2vec2_embeddings.npz")

if __name__ == "__main__":
    main()