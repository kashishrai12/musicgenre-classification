import numpy as np
from sklearn.model_selection import train_test_split

# Load the extracted features and labels
features_path = r"D:\research_project\preprocessed\byola_features.npz"
data = np.load(features_path)
X = data["features"]
y = data["labels"]
genres = data["genres"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the training set
train_output_path = r"D:\research_project\preprocessed\extracted_train.npz"
np.savez(train_output_path, X=X_train, y=y_train, genres=genres)
print(f"Training set saved with shape: {X_train.shape} to {train_output_path}")

# Save the testing set
test_output_path = r"D:\research_project\preprocessed\extracted_test.npz"
np.savez(test_output_path, X=X_test, y=y_test, genres=genres)
print(f"Testing set saved with shape: {X_test.shape} to {test_output_path}")