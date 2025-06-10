import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
import csv

# --- Configuration ---
config = {
    'batch_size': 64,
    'learning_rate': 0.0005,
    'max_epochs': 35,
    'patience': 7,
    'weight_decay': 0.0,
    'noise_factor': 0.1,
    'data_paths': {
        'gtzan_train': r"D:\research_project\preprocessed\extracted_train.npz",
        'gtzan_test': r"D:\research_project\preprocessed\extracted_test.npz",
        'fma_train': r"D:\research_project\preprocessed_fma\byola_features_train.npz",
        'fma_test': r"D:\research_project\preprocessed_fma\byola_features_test.npz",
        'fma_tracks': r"D:\research_project\dataset2\tracks.csv"
    }
}

# --- Model Architecture ---
class GenreClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GenreClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

# --- Data Loading and Processing ---
def load_fma_data(embeddings_file, tracks_csv):
    data = np.load(embeddings_file)
    tracks_df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    small_tracks_df = tracks_df[tracks_df[('set', 'subset')] == 'small']
    genre_mapping = {str(track_id).zfill(6): genre for track_id, genre in small_tracks_df[('track', 'genre_top')].items()}
    
    features = data['features']
    filenames = data['filenames']
    labels = []
    valid_indices = []
    
    for i, filename in enumerate(filenames):
        track_id = os.path.splitext(filename)[0]
        if track_id in genre_mapping:
            labels.append(genre_mapping[track_id])
            valid_indices.append(i)
    
    return features[valid_indices], np.array(labels)

def load_gtzan_data(train_path, test_path):
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    return (train_data['X'], train_data['y'].astype(str)), (test_data['X'], test_data['y'].astype(str))

def prepare_datasets():
    # Load all data
    (X_gtzan_train, y_gtzan_train), (X_gtzan_test, y_gtzan_test) = load_gtzan_data(
        config['data_paths']['gtzan_train'], config['data_paths']['gtzan_test'])
    X_fma_train, y_fma_train = load_fma_data(
        config['data_paths']['fma_train'], config['data_paths']['fma_tracks'])
    X_fma_test, y_fma_test = load_fma_data(
        config['data_paths']['fma_test'], config['data_paths']['fma_tracks'])

    # Combine and encode labels
    all_labels = np.concatenate([y_gtzan_train, y_gtzan_test, y_fma_train, y_fma_test])
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    # Convert labels to indices
    y_gtzan_train_idx = label_encoder.transform(y_gtzan_train)
    y_gtzan_test_idx = label_encoder.transform(y_gtzan_test)
    y_fma_train_idx = label_encoder.transform(y_fma_train)
    y_fma_test_idx = label_encoder.transform(y_fma_test)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_combined = np.concatenate([X_gtzan_train, X_fma_train])
    scaler.fit(X_train_combined)
    
    X_gtzan_train = scaler.transform(X_gtzan_train)
    X_gtzan_test = scaler.transform(X_gtzan_test)
    X_fma_train = scaler.transform(X_fma_train)
    X_fma_test = scaler.transform(X_fma_test)
    
    # Create combined datasets
    X_train = np.concatenate([X_gtzan_train, X_fma_train])
    y_train = np.concatenate([y_gtzan_train_idx, y_fma_train_idx])
    
    # Add noise augmentation
    noisy_features = add_noise(X_train, config['noise_factor'])
    X_train = np.concatenate([X_train, noisy_features])
    y_train = np.concatenate([y_train, y_train])
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_gtzan_test': X_gtzan_test,
        'y_gtzan_test': y_gtzan_test_idx,
        'X_fma_test': X_fma_test,
        'y_fma_test': y_fma_test_idx,
        'label_encoder': label_encoder
    }

def add_noise(data, noise_factor):
    noise = np.random.normal(0, noise_factor, data.shape)
    return data + noise

# --- Training and Evaluation ---
def train_model(X_train, y_train, X_val, y_val, config, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenreClassifier(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                              torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    best_val_acc = 0
    best_model = None
    patience_counter = 0
    
    for epoch in range(config['max_epochs']):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                val_loss += criterion(outputs, yb).item()
                _, predicted = torch.max(outputs.data, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                break
    
    model.load_state_dict(best_model)
    return model

def evaluate_model(model, X_test, y_test, label_encoder, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                               torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            _, preds = torch.max(outputs, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    
    accuracy = correct / total
    print(f"{dataset_name} Test Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return accuracy

# --- Main Execution ---
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Prepare data
    data = prepare_datasets()
    num_classes = len(data['label_encoder'].classes_)
    config['num_classes'] = num_classes
    
    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total training samples: {len(data['y_train'])}")
    print(f"GTZAN test samples: {len(data['y_gtzan_test'])}")
    print(f"FMA test samples: {len(data['y_fma_test'])}")
    print(f"Number of classes: {num_classes}")
    print("Class distribution:", dict(zip(*np.unique(data['y_train'], return_counts=True))))
    
    # Cross-validation training
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data['X_train'], data['y_train'])):
        print(f"\n=== Fold {fold+1}/5 ===")
        X_train_fold, X_val_fold = data['X_train'][train_idx], data['X_train'][val_idx]
        y_train_fold, y_val_fold = data['y_train'][train_idx], data['y_train'][val_idx]
        
        model = train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, config, num_classes)
        fold_results.append(model)
    
    # Combine test sets for evaluation
    X_test_combined = np.concatenate([data['X_gtzan_test'], data['X_fma_test']], axis=0)
    y_test_combined = np.concatenate([data['y_gtzan_test'], data['y_fma_test']], axis=0)
    
    # Evaluate on combined test set
    print("\n=== Final Evaluation on Combined Test Set ===")
    combined_acc = evaluate_model(fold_results[0], X_test_combined, y_test_combined, 
                                 data['label_encoder'], "Combined (GTZAN + FMA)")
    
    # Save results
    with open(r"D:\research_project\combined_genre_results.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            config['learning_rate'],
            config['batch_size'],
            combined_acc,
            num_classes,
            len(data['X_train'])
        ])