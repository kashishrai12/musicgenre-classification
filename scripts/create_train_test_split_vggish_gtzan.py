import numpy as np
from sklearn.model_selection import train_test_split

# Load the extracted features and labels
features_path = r"D:\research_project\preprocessed_vggish_gtzan\preprocessed_vggish_gtzan.npz"
data = np.load(features_path)
X = data["features"]
y = data["labels"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the training set
train_output_path = r"D:\research_project\preprocessed_vggish_gtzan\extracted_train_vggish_gtzan.npz"
np.savez(train_output_path, X=X_train, y=y_train)
print(f"Training set saved with shape: {X_train.shape} to {train_output_path}")

# Save the testing set
test_output_path = r"D:\research_project\preprocessed_vggish_gtzan\extracted_test_vggish_gtzan.npz"
np.savez(test_output_path, X=X_test, y=y_test)
print(f"Testing set saved with shape: {X_test.shape} to {test_output_path}")