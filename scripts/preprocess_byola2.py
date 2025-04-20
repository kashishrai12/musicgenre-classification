import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_byola(input_path, train_output_path, test_output_path, chunk_divisions=3, test_size=0.25, random_state=42):
    """
    Preprocess BYOLA embeddings by segmenting 30-second embeddings into chunks
    and splitting the dataset into training and testing sets.

    Args:
        input_path (str): Path to the BYOLA embeddings file (.npz).
        train_output_path (str): Path to save the training data (.npz).
        test_output_path (str): Path to save the testing data (.npz).
        chunk_divisions (int): Number of chunks to divide each 30-second embedding into.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    """
    # Load BYOLA embeddings and labels
    print("Loading BYOLA embeddings...")
    data = np.load(input_path)
    print("Keys in the .npz file:", data.files)
    X = data['features']  # Replace 'features' with the correct key for embeddings
    y = data['labels']    # Replace 'labels' with the correct key for genre labels
    print(f"Loaded embeddings: {X.shape}, Labels: {y.shape}")

    # Reshape embeddings to 3D (samples, time_steps, feature_dim)
    print("Reshaping embeddings...")
    time_steps = 128  # Number of time steps (e.g., 128 for 30-second files)
    feature_dim = X.shape[1] // time_steps  # Calculate feature dimension
    X = X.reshape(-1, time_steps, feature_dim)  # Reshape to (num_samples, time_steps, feature_dim)
    print(f"Reshaped embeddings: {X.shape}")

    # Segment embeddings into chunks
    print(f"Segmenting embeddings into {chunk_divisions} chunks...")
    chunk_size = X.shape[1] // chunk_divisions  # Divide time steps into equal chunks
    X_chunks = []
    y_chunks = []

    for i in range(len(X)):
        for j in range(chunk_divisions):  # Create chunks for each 30-second file
            X_chunks.append(X[i, j * chunk_size:(j + 1) * chunk_size, :])
            y_chunks.append(y[i])  # Each chunk gets the same label as the original file

    X_chunks = np.array(X_chunks)  # Shape: (num_chunks, chunk_size, feature_dim)
    y_chunks = np.array(y_chunks)  # Shape: (num_chunks,)
    print(f"Segmented embeddings: {X_chunks.shape}, Labels: {y_chunks.shape}")

    # Split into training and testing sets
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_chunks, y_chunks, test_size=test_size, random_state=random_state, stratify=y_chunks
    )

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Save training data
    print(f"Saving training data to {train_output_path}...")
    np.savez(train_output_path, X_train=X_train, y_train=y_train)

    # Save testing data
    print(f"Saving testing data to {test_output_path}...")
    np.savez(test_output_path, X_test=X_test, y_test=y_test)

    print("Preprocessing complete!")

# Example usage
if __name__ == "__main__":
    input_path = r"D:\research_project\preprocessed\byola_features.npz"
    train_output_path = r"D:\research_project\preprocessed\byola_train.npz"
    test_output_path = r"D:\research_project\preprocessed\byola_test.npz"
    preprocess_byola(input_path, train_output_path, test_output_path)