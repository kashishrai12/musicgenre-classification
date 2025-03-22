import numpy as np
import h5py

def print_npz_shapes(npz_file_path):
    data = np.load(npz_file_path)
    for key in data:
        print(f"{npz_file_path} - {key}: {data[key].shape}")

def print_h5_shapes(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        for key in f.keys():
            print(f"{h5_file_path} - {key}: {f[key].shape}")

def main():
    # Print shapes of NPZ files
    print_npz_shapes(r"D:\research_project\preprocessed_panns_gtzan\extracted_train_panns_gtzan.npz")
    print_npz_shapes(r"D:\research_project\preprocessed_panns_gtzan\extracted_test_panns_gtzan.npz")

    
    # Print shapes of HDF5 file
    print_h5_shapes(r"D:\research_project\preprocessed_panns_gtzan\features\waveform.h5")

if __name__ == "__main__":
    main()