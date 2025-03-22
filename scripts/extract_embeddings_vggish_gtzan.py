import numpy as np

# Load the .npz file
data = np.load(r"D:\research_project\preprocessed_vggish_gtzan\preprocessed_vggish_gtzan.npz")

# Print the keys in the .npz file
print("Keys in the .npz file:", data.files)

# Extract features and labels using the correct keys
features = data['features']
labels = data['labels']

# Print the shapes of the arrays
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Optionally, print a few samples
print("Sample features:", features[:2])
print("Sample labels:", labels[:2])