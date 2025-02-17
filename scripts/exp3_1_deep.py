from base_training import run_training, save_results
import torch.nn as nn
import numpy as np

class DeepClassifier3(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, int(256 * 1.5)),  # 384 neurons
            nn.ReLU(),
            nn.Linear(int(256 * 1.5), int(128 * 1.5)),  # 192 neurons
            nn.ReLU(),
            nn.Linear(int(128 * 1.5), int(64 * 1.5)),  # 96 neurons
            nn.ReLU(),
            nn.Linear(int(64 * 1.5), int(32 * 1.5)),  # 48 neurons
            nn.ReLU(),
            nn.Linear(int(32 * 1.5), num_classes)  # Output layer
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
        'experiment_name': 'exp3_1_deep'
    }
    
    experiment_name = config['experiment_name']
    fold_results = run_training(features, labels, config, DeepClassifier3, experiment_name)
    save_results(experiment_name, fold_results, config)

if __name__ == "__main__":
    main()