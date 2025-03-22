import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import json
import csv
import seaborn as sns  # For better visualization of the confusion matrix
from sklearn.metrics import confusion_matrix

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

def run_training(train_features, train_labels, val_features, val_labels, config, model_class, experiment_name, genre_names):
    set_seed(42)  # Set a fixed random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    
    train_dataset = TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
    val_dataset = TensorDataset(torch.FloatTensor(val_features), torch.LongTensor(val_labels))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = model_class(input_dim=train_features.shape[1], num_classes=config['num_classes']).to(device)
    reset_weights(model)  # Reset model weights
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])  # Added L2 regularization
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop with early stopping
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
    
    return {
        'train_accuracies': train_accuracies,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

def evaluate_model(train_features, train_labels, test_features, test_labels, config, model_class, genre_names):
    set_seed(42)  # Set a fixed random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    
    train_dataset = TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
    test_dataset = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = model_class(input_dim=train_features.shape[1], num_classes=config['num_classes']).to(device)
    reset_weights(model)  # Reset model weights
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])  # Added L2 regularization
    criterion = nn.CrossEntropyLoss()
    
    # Train the model on the entire training set
    model.train()
    for epoch in range(config['max_epochs']):
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
        print(f"Epoch {epoch + 1}/{config['max_epochs']}, Train Accuracy: {train_accuracy:.4f}, Train Loss: {epoch_train_loss / len(train_loader):.4f}")
    
    # Evaluate the model on the test set
    model.eval()
    y_test_true = []
    y_test_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(device))
            y_test_true.extend(y_batch.cpu().numpy())
            y_test_pred.extend(outputs.argmax(dim=1).cpu().numpy())
    
    test_accuracy = np.mean(np.array(y_test_true) == np.array(y_test_pred)) * 100  # Convert to percentage
    
    return {
        'y_true': y_test_true,
        'y_pred': y_test_pred,
        'test_accuracy': test_accuracy
    }

def save_results(experiment_name, results, config, genre_names):
    results_dir = os.path.join('results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Append results to CSV
    csv_filepath = r"D:\research_project\average_experiment_results.csv"
    with open(csv_filepath, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([experiment_name, results['test_accuracy']])
    
    # Plot confusion matrix
    plot_confusion_matrix(results['y_true'], results['y_pred'], genre_names)

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
    
    experiments = [
        {
            'name': 'exp6_2_batchnorm_dropout',
            'model_class': BatchNormDropoutClassifier7_2,  # Ensure correct model
            'config': {
                'batch_size': 64,
                'learning_rate': 0.0005,
                'max_epochs': 50,
                'patience': 7,
                'num_classes': len(np.unique(train_labels)),
                'weight_decay': 1e-5  # Added L2 regularization
            }
        },
    ]
    
    for experiment in experiments:
        experiment_name = experiment['name']
        model_class = experiment['model_class']
        config = experiment['config']
        config['input_dim'] = train_features.shape[1]
        
        # Perform cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, train_labels)):
            print(f"\nFold {fold + 1}/5")
            X_train, X_val = train_features[train_idx], train_features[val_idx]
            y_train, y_val = train_labels[train_idx], train_labels[val_idx]
            
            fold_result = run_training(X_train, y_train, X_val, y_val, config, model_class, experiment_name, genre_names)
            fold_results.append(fold_result)
        
        # Calculate average metrics
        avg_best_val_acc = np.mean([max(result['val_accuracies']) for result in fold_results]) * 100
        avg_train_losses = np.mean([np.mean(result['train_losses']) for result in fold_results])
        avg_val_losses = np.mean([np.mean(result['val_losses']) for result in fold_results])
        
        # Evaluate on test set
        test_result = evaluate_model(train_features, train_labels, test_features, test_labels, config, model_class, genre_names)
        avg_test_acc = test_result['test_accuracy']
        
        # Save results
        with open(r"D:\research_project\average_experiment_results.csv", 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([experiment_name, avg_best_val_acc, avg_train_losses, avg_val_losses, avg_test_acc])
        
        # Plot confusion matrix for the test set
        plot_confusion_matrix(test_result['y_true'], test_result['y_pred'], genre_names)

if __name__ == "__main__":
    main()