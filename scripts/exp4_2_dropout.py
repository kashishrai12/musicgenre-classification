from base_training2 import run_training, evaluate_model, save_results, set_seed
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import csv

class DropoutClassifier4_2(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
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

def main():
    set_seed(42)  # Set a fixed random seed for reproducibility
    
    # Load training data
    train_data = np.load(r"D:\research_project\preprocessed\extracted_train.npz")
    train_features = train_data['X']
    train_labels = train_data['y']
    genre_names = train_data['genres']  # Assuming genres are stored in the npz file
    
    # Load testing data
    test_data = np.load(r"D:\research_project\preprocessed\extracted_test.npz")
    test_features = test_data['X']
    test_labels = test_data['y']
    
    config = {
        'batch_size': 64,
        'learning_rate': 0.0005,
        'max_epochs': 50,  # Set max_epochs to 50
        'patience': 7,  # Set patience to 7
        'num_classes': len(np.unique(train_labels)),
        'experiment_name': 'exp4_2_dropout',
        'weight_decay': 0.0
    }
    
    experiment_name = config['experiment_name']
    
    # Perform cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, train_labels)):
        print(f"\nFold {fold + 1}/5")
        X_train, X_val = train_features[train_idx], train_features[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]
        
        fold_result = run_training(X_train, y_train, X_val, y_val, config, DropoutClassifier4_2, experiment_name, genre_names)
        fold_results.append(fold_result)
    
    # Calculate average metrics
    avg_best_val_acc = np.mean([max(result['val_accuracies']) for result in fold_results]) * 100
    avg_train_losses = np.mean([np.mean(result['train_losses']) for result in fold_results])
    avg_val_losses = np.mean([np.mean(result['val_losses']) for result in fold_results])
    
    # Evaluate on test set
    test_result = evaluate_model(train_features, train_labels, test_features, test_labels, config, DropoutClassifier4_2, genre_names)
    avg_test_acc = test_result['test_accuracy']
    
    # Save results
    with open(r"D:\research_project\average_experiment_results.csv", 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([experiment_name, avg_best_val_acc, avg_train_losses, avg_val_losses, avg_test_acc])
    
    # Plot confusion matrix for the test set
    plot_confusion_matrix(test_result['y_true'], test_result['y_pred'], genre_names)

if __name__ == "__main__":
    main()