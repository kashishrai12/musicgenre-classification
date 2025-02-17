from base_training import run_training, save_results, set_seed
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

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

def add_noise(data, noise_factor=0.1):
    noise = np.random.normal(0, noise_factor, data.shape)
    return data + noise

def plot_accuracy_curves(train_accuracies, test_accuracies):
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training accuracy')
    plt.plot(epochs, test_accuracies, 'ro-', label='Test accuracy')
    plt.title('Training and test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_loss_curves(train_losses, test_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, test_losses, 'ro-', label='Test loss')
    plt.title('Training and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences

def main():
    set_seed(42)  # Set a fixed random seed for reproducibility
    data_path = r"D:\research_project\preprocessed\byola_features.npz"
    data = np.load(data_path)
    features = data['features']
    labels = data['labels']
    genre_names = data['genres']  # Assuming genres are stored in the npz file
    
    # Add noise to a subset of the training data
    noise_factor = 0.1
    noisy_features = add_noise(features, noise_factor)
    augmented_features = np.concatenate((features, noisy_features), axis=0)
    augmented_labels = np.concatenate((labels, labels), axis=0)
    
    config = {
        'batch_size': 64,
        'learning_rate': 0.0005,
        'max_epochs': 35,  # Set max_epochs to 35
        'patience': 5,  # Set patience to 5
        'num_classes': len(np.unique(labels)),
        'experiment_name': 'check_overfitting_noise_augmentation_once',
        'weight_decay': 0.0
    }
    
    experiment_name = config['experiment_name']
    fold_results = run_training(augmented_features, augmented_labels, config, NoiseAugmentationModel, experiment_name, genre_names)
    
    # Extract training and test accuracies and losses
    train_accuracies = [result['train_accuracies'] for result in fold_results]
    test_accuracies = [result['val_accuracies'] for result in fold_results]  # Assuming val_accuracies are used as test accuracies
    train_losses = [result['train_losses'] for result in fold_results]
    test_losses = [result['val_losses'] for result in fold_results]  # Assuming val_losses are used as test losses
    
    # Find the maximum length of the sequences
    maxlen = max(max(len(seq) for seq in train_accuracies), max(len(seq) for seq in test_accuracies))
    
    # Pad sequences to the same length
    train_accuracies = pad_sequences(train_accuracies, maxlen)
    test_accuracies = pad_sequences(test_accuracies, maxlen)
    train_losses = pad_sequences(train_losses, maxlen)
    test_losses = pad_sequences(test_losses, maxlen)
    
    # Compute average accuracies and losses across folds
    avg_train_accuracies = np.mean(train_accuracies, axis=0)
    avg_test_accuracies = np.mean(test_accuracies, axis=0)
    avg_train_losses = np.mean(train_losses, axis=0)
    avg_test_losses = np.mean(test_losses, axis=0)
    
    # Plot accuracy and loss curves
    plot_accuracy_curves(avg_train_accuracies, avg_test_accuracies)
    plot_loss_curves(avg_train_losses, avg_test_losses)

if __name__ == "__main__":
    main()