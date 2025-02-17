from base_training import run_training, save_results, set_seed
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from map_embeddings_to_labels import load_genre_labels  # Import the function

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

def evaluate_model(features, labels, model, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    dataset = TensorDataset(torch.FloatTensor(features), torch.LongTensor(labels))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")

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

def main():
    set_seed(42)  # Set a fixed random seed for reproducibility
    data_path = r"D:\research_project\dataset2\byola_features\byola_features.npz"
    tracks_csv = r"D:\research_project\dataset2\tracks.csv"
    
    # Load features and filenames
    data = np.load(data_path)
    features = data['features']
    filenames = data['filenames']
    
    # Load genre labels using the imported function
    genre_mapping = load_genre_labels(tracks_csv)
    
    # Map filenames to genres
    labels = []
    for filename in filenames:
        track_id = os.path.splitext(filename)[0]
        if track_id in genre_mapping:
            labels.append(genre_mapping[track_id])
        else:
            labels.append('Unknown')
            print(f"Warning: {track_id} not found in genre_mapping, assigning default genre: Unknown")
    
    # Debug: Print the label distribution
    label_distribution = Counter(labels)
    print("Label distribution:", label_distribution)
    
    # Filter out 'Unknown' labels
    valid_indices = [i for i, label in enumerate(labels) if label != 'Unknown']
    if not valid_indices:
        raise ValueError("No valid samples left after filtering 'Unknown' labels. Check the dataset.")
    
    features = features[valid_indices]
    labels = np.array(labels)[valid_indices]
    
    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
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
        'experiment_name': 'check_overfitting_fma',
        'weight_decay': 0.0
    }
    
    experiment_name = config['experiment_name']
    fold_results = run_training(augmented_features, augmented_labels, config, NoiseAugmentationModel, experiment_name, label_encoder.classes_)
    save_results(experiment_name, fold_results, config, label_encoder.classes_)
    
    # Debug: Print the structure of fold_results
    print("Fold results structure:", fold_results)
    
    # Extract training and test losses and accuracies from fold_results
    train_losses = fold_results[0]['train_losses']
    test_losses = fold_results[0]['val_losses']
    train_accuracies = fold_results[0]['train_accuracies']
    test_accuracies = fold_results[0]['val_accuracies']
    
    # Plot loss and accuracy curves
    plot_loss_curves(train_losses, test_losses)
    plot_accuracy_curves(train_accuracies, test_accuracies)
    
    model = NoiseAugmentationModel(input_dim=features.shape[1], num_classes=config['num_classes'])
    evaluate_model(features, labels, model, config)

if __name__ == "__main__":
    main()