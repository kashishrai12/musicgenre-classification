from base_training import run_training, save_results
import torch
import torch.nn as nn
import numpy as np

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SwishClassifier5_3(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            Swish(),
            nn.Linear(256, 128),
            Swish(),
            nn.Linear(128, 64),
            Swish(),
            nn.Linear(64, 32),
            Swish(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

def main():
    data = np.load(r"D:\research_project\preprocessed\byola_features.npz")
    features = data['features']
    labels = data['labels']
    
    config = {
        'batch_size': 64,
        'learning_rate': 0.0005,
        'max_epochs': 50,
        'patience': 7,
        'num_classes': len(np.unique(labels)),
        'experiment_name': 'exp5_3_swish'
    }
    
    experiment_name = config['experiment_name']
    fold_results = run_training(features, labels, config, SwishClassifier6_3, experiment_name)
    save_results(experiment_name, fold_results, config)

if __name__ == "__main__":
    main()