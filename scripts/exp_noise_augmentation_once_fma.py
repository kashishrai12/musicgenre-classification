from base_training import run_training, save_results, set_seed
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
from map_embeddings_to_labels import load_genre_labels  # Import the function
import seaborn as sns

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

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

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
    train_data_path = r"D:\research_project\preprocessed_fma\byola_features_train.npz"
    train_data = np.load(train_data_path)
    train_features = train_data['features']
    train_filenames = train_data['filenames']
    
    # Load test data
    test_data_path = r"D:\research_project\preprocessed_fma\byola_features_test.npz"
    test_data = np.load(test_data_path)
    test_features = test_data['features']
    test_filenames = test_data['filenames']
    
    # Load genre labels
    tracks_csv = r"D:\research_project\dataset2\tracks.csv"
    genre_mapping = load_genre_labels(tracks_csv)
    
    # Map filenames to genres for training data
    train_labels = []
    for filename in train_filenames:
        track_id = os.path.splitext(filename)[0]
        if track_id in genre_mapping:
            train_labels.append(genre_mapping[track_id])
        else:
            train_labels.append('Unknown')
            print(f"Warning: {track_id} not found in genre_mapping, assigning default genre: Unknown")
    
    # Map filenames to genres for test data
    test_labels = []
    for filename in test_filenames:
        track_id = os.path.splitext(filename)[0]
        if track_id in genre_mapping:
            test_labels.append(genre_mapping[track_id])
        else:
            test_labels.append('Unknown')
            print(f"Warning: {track_id} not found in genre_mapping, assigning default genre: Unknown")
    
    # Filter out 'Unknown' labels from training data
    train_valid_indices = [i for i, label in enumerate(train_labels) if label != 'Unknown']
    train_features = train_features[train_valid_indices]
    train_labels = np.array(train_labels)[train_valid_indices]
    
    # Filter out 'Unknown' labels from test data
    test_valid_indices = [i for i, label in enumerate(test_labels) if label != 'Unknown']
    test_features = test_features[test_valid_indices]
    test_labels = np.array(test_labels)[test_valid_indices]
    
    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)
    
    # Add noise to a subset of the training data
    noise_factor = 0.1
    noisy_features = add_noise(train_features, noise_factor)
    augmented_features = np.concatenate((train_features, noisy_features), axis=0)
    augmented_labels = np.concatenate((train_labels, train_labels), axis=0)
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5)
    best_val_accuracies = []
    train_losses = []
    val_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(augmented_features, augmented_labels)):
        print(f"Fold {fold + 1}")
        
        train_subset = Subset(TensorDataset(torch.FloatTensor(augmented_features), torch.LongTensor(augmented_labels)), train_idx)
        val_subset = Subset(TensorDataset(torch.FloatTensor(augmented_features), torch.LongTensor(augmented_labels)), val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        
        model = NoiseAugmentationModel(input_dim=augmented_features.shape[1], num_classes=len(label_encoder.classes_)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        
        num_epochs = 35
        best_val_acc = 0.0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")
            
            # Evaluate the model on the validation set
            y_true_val, y_pred_val = evaluate_model(model, val_loader, device)
            val_acc = np.mean(np.array(y_true_val) == np.array(y_pred_val))
            val_losses.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            print(f"Validation Accuracy for fold {fold + 1}, epoch {epoch + 1}: {val_acc:.4f}")
        
        best_val_accuracies.append(best_val_acc)
    
    # Evaluate the final model on the test set
    test_dataset = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    y_true_test, y_pred_test = evaluate_model(model, test_loader, device)
    test_acc = np.mean(np.array(y_true_test) == np.array(y_pred_test))
    
    # Print test results
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Convert accuracies to percentages
    avg_best_val_acc_percent = np.mean(best_val_accuracies) * 100
    avg_test_acc_percent = test_acc * 100

    # Save results to CSV
    results = {
        'experiment_name': 'exp_noise_augmentation_once_fma',
        'avg_best_val_acc (%)': avg_best_val_acc_percent,
        'avg_train_losses': np.mean(train_losses),
        'avg_val_losses': np.mean(val_losses),
        'avg_test_acc (%)': avg_test_acc_percent
    }
    results_df = pd.DataFrame([results])
    results_csv_path = r"D:\research_project\average_experiment_results.csv"
    
    # Debug: Print results before saving
    print("Results to be saved:")
    print(results_df)
    
    if os.path.exists(results_csv_path):
        results_df.to_csv(results_csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    # Plot confusion matrix for the test set
    plot_confusion_matrix(y_true_test, y_pred_test, label_encoder.classes_)

if __name__ == "__main__":
    main()