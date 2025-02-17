from base_training import run_training, save_results
import torch.nn as nn
import numpy as np

class BatchNormDropoutClassifier7_1(nn.Module):
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

def main():
    data = np.load(r"D:\research_project\preprocessed\byola_penultimate_features.npz")
    features = data['features']  # Use the correct key for penultimate layer embeddings
    labels = data['labels']
    
    # Reshape features to 2 dimensions
    features = features.reshape(features.shape[0], -1)
    
    config = {
        'batch_size': 64,
        'learning_rate': 0.0005,
        'max_epochs': 50,
        'patience': 7,
        'num_classes': len(np.unique(labels)),
        'experiment_name': 'exp7_1_penultimate'
    }
    
    experiment_name = config['experiment_name']
    fold_results = run_training(features, labels, config, BatchNormDropoutClassifier8_1, experiment_name)
    save_results(experiment_name, fold_results, config)

if __name__ == "__main__":
    main()