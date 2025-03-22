from base_training2 import run_training, evaluate_model, save_results, set_seed
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import csv

class NoiseAugmentationModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

def plot_confusion_matrix(y_true, y_pred, genre_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_names, yticklabels=genre_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def add_noise(data, noise_factor=0.1):
    noise = np.random.normal(0, noise_factor, data.shape)
    return data + noise

def main():
    set_seed(42)  # Set a fixed random seed for reproducibility
    
    # Load training data
    train_data = np.load(r"D:\research_project\preprocessed_vggish_gtzan\extracted_train_vggish_gtzan.npz")
    train_features = train_data['X']
    train_labels = train_data['y']
    
    # Load testing data
    test_data = np.load(r"D:\research_project\preprocessed_vggish_gtzan\extracted_test_vggish_gtzan.npz")
    test_features = test_data['X']
    test_labels = test_data['y']
    
    # Encode genre names as numerical labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)
    
    # Add noise to a subset of the training data
    noise_factor = 0.1
    noisy_features = add_noise(train_features, noise_factor)
    augmented_features = np.concatenate((train_features, noisy_features), axis=0)
    augmented_labels = np.concatenate((train_labels, train_labels), axis=0)
    
    # Flatten the features to 2D
    augmented_features = augmented_features.reshape(augmented_features.shape[0], -1)
    test_features = test_features.reshape(test_features.shape[0], -1)
    
    config = {
        'batch_size': 64,
        'learning_rate': 0.0005,
        'max_epochs': 35,  # Set max_epochs to 35
        'patience': 5,  # Set patience to 5
        'num_classes': len(np.unique(train_labels)),
        'experiment_name': 'exp_noise_augmentation_once_vggish_gtzan',
        'weight_decay': 0.0
    }
    
    experiment_name = config['experiment_name']
    genre_names = label_encoder.classes_  # Use the label encoder to get the genre names
    
    # Perform cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(augmented_features, augmented_labels)):
        print(f"\nFold {fold + 1}/5")
        X_train, X_val = augmented_features[train_idx], augmented_features[val_idx]
        y_train, y_val = augmented_labels[train_idx], augmented_labels[val_idx]
        
        fold_result = run_training(X_train, y_train, X_val, y_val, config, NoiseAugmentationModel, experiment_name, genre_names)
        fold_results.append(fold_result)
    
    # Calculate average metrics
    avg_best_val_acc = np.mean([max(result['val_accuracies']) for result in fold_results]) * 100
    avg_train_losses = np.mean([np.mean(result['train_losses']) for result in fold_results])
    avg_val_losses = np.mean([np.mean(result['val_losses']) for result in fold_results])
    
    # Evaluate on test set
    test_result = evaluate_model(train_features.reshape(train_features.shape[0], -1), train_labels, test_features, test_labels, config, NoiseAugmentationModel, genre_names)
    avg_test_acc = test_result['test_accuracy']
    
    # Save results
    with open(r"D:\research_project\average_experiment_results.csv", 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([experiment_name, avg_best_val_acc, avg_train_losses, avg_val_losses, avg_test_acc])
    
    # Plot confusion matrix for the test set
    plot_confusion_matrix(test_result['y_true'], test_result['y_pred'], genre_names)

if __name__ == "__main__":
    main()