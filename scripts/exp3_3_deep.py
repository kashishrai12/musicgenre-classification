from base_training import run_training, save_results
import torch.nn as nn
import numpy as np

class DeepClassifier3(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, int(256 * 0.5)),  # 128 neurons
            nn.ReLU(),
            nn.Linear(int(256 * 0.5), int(128 * 0.5)),  # 64 neurons
            nn.ReLU(),
            nn.Linear(int(128 * 0.5), int(64 * 0.5)),  # 32 neurons
            nn.ReLU(),
            nn.Linear(int(64 * 0.5), int(32 * 0.5)),  # 16 neurons
            nn.ReLU(),
            nn.Linear(int(32 * 0.5), num_classes)  # Output layer
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
        'experiment_name': 'exp3_3_deep'
    }
    
    experiment_name = config['experiment_name']
    fold_results = run_training(features, labels, config, DeepClassifier3, experiment_name)
    save_results(experiment_name, fold_results, config)

if __name__ == "__main__":
    main()