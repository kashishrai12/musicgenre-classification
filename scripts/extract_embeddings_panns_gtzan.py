import sys
import os
import torch
import h5py
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

# Add the directory containing the models and pytorch_utils modules to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'panns_transfer_to_gtzan', 'pytorch')))

from models_panns import Transfer_Cnn14

# Load the pretrained model
model = Transfer_Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=10, freeze_base=True)

# Load the checkpoint with weights_only=False
checkpoint_path = 'D:/research_project/panns_transfer_to_gtzan/Cnn14_mAP=0.431.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Modify the keys in the checkpoint to include the 'base.' prefix
new_state_dict = {}
for key, value in checkpoint['model'].items():
    if not key.startswith('base.'):
        new_key = f'base.{key}'  # Add 'base.' prefix
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# Load the modified state dictionary into the model
model.load_state_dict(new_state_dict, strict=False)

# Initialize missing keys
if 'fc_transfer.weight' not in new_state_dict:
    model.fc_transfer.weight.data.normal_(0, 0.01)
if 'fc_transfer.bias' not in new_state_dict:
    model.fc_transfer.bias.data.zero_()

model.eval()

# Path to the HDF5 file
hdf5_path = 'D:/research_project/preprocessed_panns_gtzan/features/waveform.h5'

# Load waveform data from HDF5 file
with h5py.File(hdf5_path, 'r') as f:
    waveforms = f['waveform'][:]

# Generate labels for the GTZAN dataset
genres = [
    "blues", "classical", "country", "disco", "hiphop", 
    "jazz", "metal", "pop", "reggae", "rock"
]
num_samples_per_genre = 100  # 100 samples per genre
num_genres = len(genres)  # 10 genres
labels = np.repeat(np.arange(num_genres), num_samples_per_genre)  # Create labels

# Extract embeddings
embeddings = []
with torch.no_grad():
    for waveform in tqdm(waveforms, desc="Extracting embeddings"):  # Add progress bar
        input = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        output_dict = model(input)
        embedding = output_dict['embedding'].squeeze(0).numpy()  # Extracted embedding
        embeddings.append(embedding)

# Convert embeddings list to numpy array
embeddings = np.array(embeddings)
print("Shape of embeddings:", embeddings.shape)  # Should be (number_of_waveforms, 2048)

# Save embeddings and labels to a new HDF5 file
output_hdf5_path = 'D:/research_project/preprocessed_panns_gtzan/features/embeddings_with_labels.h5'
with h5py.File(output_hdf5_path, 'w') as f:
    f.create_dataset('embeddings', data=embeddings)
    f.create_dataset('labels', data=labels)

print(f"Embeddings and labels saved to {output_hdf5_path}")