import numpy as np

# Load the embeddings and labels
data = np.load("D:/research_project/gtzan_wav2vec2_embeddings.npz")
X = data["X"]
y = data["y"]
files = data["files"]

# Set random seed for reproducibility
np.random.seed(42)

# Shuffle indices
num_samples = X.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Split 80% train, 20% test
split_idx = int(0.8 * num_samples)
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

# Create splits
X_train, y_train, files_train = X[train_idx], y[train_idx], files[train_idx]
X_test, y_test, files_test = X[test_idx], y[test_idx], files[test_idx]

# Save to separate files
np.savez("D:/research_project/gtzan_wav2vec2_train.npz", X=X_train, y=y_train, files=files_train)
np.savez("D:/research_project/gtzan_wav2vec2_test.npz", X=X_test, y=y_test, files=files_test)

print("Train and test splits saved.")