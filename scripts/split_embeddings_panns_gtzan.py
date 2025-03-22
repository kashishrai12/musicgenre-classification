import numpy as np
from sklearn.model_selection import train_test_split
import os
import h5py

def main():
    # Load extracted features from HDF5 file
    data_path = r"D:\research_project\preprocessed_panns_gtzan\features\embeddings_with_labels.h5"
    
    # Check if the file exists
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    with h5py.File(data_path, 'r') as f:
        X = f['embeddings'][:]  # Load the embeddings dataset
        y = f['labels'][:]  # Load the target labels dataset

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create output directory if it doesn't exist
    output_dir = r"D:\research_project\preprocessed_panns_gtzan"
    os.makedirs(output_dir, exist_ok=True)

    # Save training set
    train_output_path = os.path.join(output_dir, 'extracted_train_panns_gtzan.npz')
    np.savez(train_output_path, features=X_train, labels=y_train)
    print(f"Training set saved to {train_output_path}")

    # Save testing set
    test_output_path = os.path.join(output_dir, 'extracted_test_panns_gtzan.npz')
    np.savez(test_output_path, features=X_test, labels=y_test)
    print(f"Testing set saved to {test_output_path}")

    # Print the shapes of the arrays
    print(f"Training features shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Testing labels shape: {y_test.shape}")

if __name__ == "__main__":
    main()