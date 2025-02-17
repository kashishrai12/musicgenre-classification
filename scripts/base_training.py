import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import os
import csv
import seaborn as sns  # For better visualization of the confusion matrix

# Set a fixed random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GenreDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class LinearClassifier(nn.Module):
    """Two-layer neural network classifier"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        hidden_dim = input_dim // 2  # Hidden layer size as half of input dimension
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class BatchNormDropoutClassifier7_2(nn.Module):
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


class BatchNormClassifier7_1(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
    
def reset_weights(model):
    """Reset model weights to avoid weight leakage between folds"""
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def plot_confusion_matrix(y_true, y_pred, genre_names):
    """
    Plots a confusion matrix using seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_names, yticklabels=genre_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def run_training(features, labels, config, model_class, experiment_name, genre_names):
    set_seed(42)  # Set a fixed random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split data into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_val, y_train_val)):
        print(f"\nFold {fold + 1}/5")
        
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        model = model_class(input_dim=features.shape[1], num_classes=config['num_classes']).to(device)
        reset_weights(model)  # Reset model weights
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])  # Added L2 regularization
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training and validation loop with early stopping
        for epoch in range(config['max_epochs']):
            model.train()
            correct_train = 0
            total_train = 0
            epoch_train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch.to(device))
                loss = criterion(outputs, y_batch.to(device))
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += y_batch.size(0)
                correct_train += (predicted == y_batch.to(device)).sum().item()
            
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)
            train_losses.append(epoch_train_loss / len(train_loader))
            
            model.eval()
            correct_val = 0
            total_val = 0
            epoch_val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch.to(device))
                    loss = criterion(outputs, y_batch.to(device))
                    epoch_val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += y_batch.size(0)
                    correct_val += (predicted == y_batch.to(device)).sum().item()
            
            val_loss = epoch_val_loss / len(val_loader)
            val_accuracy = correct_val / total_val
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch + 1}/{config['max_epochs']}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print("Early stopping triggered")
                    break
        
        # Evaluate on test set
        model.eval()
        y_test_true = []
        y_test_pred = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch.to(device))
                y_test_true.extend(y_batch.cpu().numpy())
                y_test_pred.extend(outputs.argmax(dim=1).cpu().numpy())
        
        fold_results.append({
            'fold': fold,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'y_true': y_test_true,
            'y_pred': y_test_pred
        })
    
    return fold_results

def save_results(experiment_name, fold_results, config, genre_names):
    results_dir = os.path.join('results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    for fold_result in fold_results:
        results.append({
            'fold': int(fold_result['fold']),  # Convert to regular int
            'y_true': [int(y) for y in fold_result['y_true']],  # Convert to regular int
            'y_pred': [int(y) for y in fold_result['y_pred']],  # Convert to regular int
            'train_losses': fold_result['train_losses'],
            'val_losses': fold_result['val_losses'],
            'val_accuracies': fold_result['val_accuracies']
        })
    
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Compute average best validation accuracy, average losses, and average test accuracy
    avg_best_val_acc = np.mean([max(result['val_accuracies']) for result in fold_results]) * 100
    avg_train_losses = np.mean([np.mean(result['train_losses']) for result in fold_results])
    avg_val_losses = np.mean([np.mean(result['val_losses']) for result in fold_results])
    
    test_accuracies = []
    for result in fold_results:
        y_true = result['y_true']
        y_pred = result['y_pred']
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        test_accuracies.append(accuracy)
    avg_test_acc = np.mean(test_accuracies) * 100  # Convert to percentage
    
    # Append results to CSV
    csv_filepath = r"D:\research_project\average_experiment_results.csv"
    with open(csv_filepath, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([experiment_name, avg_best_val_acc, avg_train_losses, avg_val_losses, avg_test_acc])
    
    # Plot confusion matrix for the last fold
    y_true_all = fold_results[-1]['y_true']
    y_pred_all = fold_results[-1]['y_pred']
    plot_confusion_matrix(y_true_all, y_pred_all, genre_names)

def main():
    set_seed(42)  # Set a fixed random seed for reproducibility
    data = np.load(r"D:\research_project\preprocessed\byola_features.npz")
    features = data['features']
    labels = data['labels']
    genre_names = data['genres']  # Assuming genres are stored in the npz file
    
    experiments = [
        {
            'name': 'exp7_2_batchnorm_dropout',
            'model_class': BatchNormDropoutClassifier7_2,  # Ensure correct model
            'config': {
                'batch_size': 64,
                'learning_rate': 0.0005,
                'max_epochs': 50,
                'patience': 7,
                'num_classes': len(np.unique(labels)),
                'weight_decay': 1e-5  # Added L2 regularization
            }
        },
    ]
    
    for experiment in experiments:
        experiment_name = experiment['name']
        model_class = experiment['model_class']
        config = experiment['config']
        config['input_dim'] = features.shape[1]
        
        fold_results = run_training(features, labels, config, model_class, experiment_name, genre_names)
        save_results(experiment_name, fold_results, config, genre_names)

if __name__ == "__main__":
    main()