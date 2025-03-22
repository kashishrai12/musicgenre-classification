import h5py

def inspect_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        print("Contents of the HDF5 file:")
        for key in f.keys():
            print(f"{key}: {f[key].shape}")
            print(f"Data type of {key}: {f[key].dtype}")
            print(f"First few entries of {key}: {f[key][:5]}")  # Print first few entries for inspection

def print_keys(file_path):
    with h5py.File(file_path, 'r') as f:
        print("Keys in the HDF5 file:")
        for key in f.keys():
            print(key)

if __name__ == "__main__":
    file_path = "D:/research_project/preprocessed_panns_gtzan/features/embeddings_with_labels.h5"
    inspect_hdf5(file_path)
    print_keys(file_path)