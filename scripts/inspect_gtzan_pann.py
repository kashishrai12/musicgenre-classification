import numpy as np

# Load the .npz file
train_data = np.load('D:/research_project/preprocessed_panns_gtzan/extracted_train_panns_gtzan.npz')
test_data = np.load('D:/research_project/preprocessed_panns_gtzan/extracted_test_panns_gtzan.npz')

# Access features and labels
train_features = train_data['features']
test_features = test_data['features']

# Check the shape of a single sample
print("Shape of a single training sample:", train_features[0].shape)
print("Shape of a single test sample:", test_features[0].shape)

