import h5py

# Load the .h5 file
with h5py.File('D:/research_project/preprocessed_panns_gtzan/features/waveform.h5', 'r') as f:
    waveforms = f['waveform'][:]
    audio_names = f['audio_name'][:]
    folds = f['fold'][:]
    targets = f['target'][:]

# Check the shape of a single waveform
print("Shape of a single waveform:", waveforms[0].shape)
print("Shape of waveforms:", waveforms.shape)
print("First few samples of the first waveform:", waveforms[0][:10])