from base_training import run_training, save_results
import torch.nn as nn
import numpy as np

class DeepClassifier2(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
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
        'experiment_name': 'exp2_2_deep'
    }
    
    experiment_name = config['experiment_name']
    fold_results = run_training(features, labels, config, DeepClassifier2, experiment_name)
    save_results(experiment_name, fold_results, config)

if __name__ == "__main__":
    main()