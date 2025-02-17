from base_training import run_training, save_results, set_seed
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
        'experiment_name': 'exp_noise_augmentation_once',
        'weight_decay': 0.0
    }
    
    experiment_name = config['experiment_name']
    fold_results = run_training(augmented_features, augmented_labels, config, NoiseAugmentationModel, experiment_name, genre_names)
    save_results(experiment_name, fold_results, config, genre_names)

if __name__ == "__main__":
    main()