import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from models import (
    get_model, 
    get_config, 
    EnsembleModel,
    LinearClassifier,
    DeepClassifier,
    BaseModel
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.decomposition import PCA
import time

class GenreDataset(Dataset):
    """Dataset for genre classification"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, accuracy

def evaluate(model, val_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy

def run_training(features, labels, run_idx, config, model_type='linear', model=None):
    """Run training with cross-validation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reshape and reduce features once at the start
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
        print(f"Reshaped features to: {features.shape}")
        
        # Apply PCA for dimensionality reduction
        n_components = min(512, features.shape[0] - 1)
        print(f"Reducing dimensions to {n_components} using PCA...")
        pca = PCA(n_components=n_components)
        features = pca.fit_transform(features)
        print(f"Features shape after PCA: {features.shape}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Store PCA dimensions in config
        config['input_dim'] = n_components
    
    # First, create a train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    # Use stratified k-fold
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42+run_idx)
    fold_accuracies = []
    best_model_state = None
    best_overall_acc = 0
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        print(f"\nFold {fold + 1}/5")
        
        # Split data for this fold
        X_train_fold, X_val = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val = y_train[train_idx], y_train[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val = scaler.transform(X_val)
        
        # Create data loaders
        train_dataset = GenreDataset(X_train_fold, y_train_fold)
        val_dataset = GenreDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
        
        # Initialize model
        if model is None:
            model = get_model(model_type, config['input_dim'], config['num_classes']).to(device)
        else:
            model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(config['max_epochs']):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if val_acc > best_overall_acc:
                    best_overall_acc = val_acc
                    best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config['max_epochs']}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        fold_accuracies.append(best_val_acc)
        print(f"Fold {fold + 1} best validation accuracy: {best_val_acc:.2f}%")
    
    mean_acc = np.mean(fold_accuracies)
    print(f"\nCross-validation accuracy: {mean_acc:.2f}% ± {np.std(fold_accuracies):.2f}%")
    
    return mean_acc, best_model_state

def train_single_model(model_type, features, labels, config):
    """Train a single model and return results"""
    print(f"\nTraining {model_type} model...")
    accuracy, model_state = run_training(
        features=features,
        labels=labels,
        run_idx=0,
        config=config,
        model_type=model_type
    )
    
    print(f"\n{model_type.capitalize()} Model Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return accuracy, model_state

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data as before
    data = np.load(r"D:\research_project\preprocessed\byola_penultimate_features.npz")
    features = data['features']
    labels = data['labels']
    
    # Setup directories and config as before
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(r"D:\research_project\experiments\penultimate")
    results_dir = base_dir / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create or load master results CSV
    results_csv = base_dir / 'penultimate_experiments.csv'
    if results_csv.exists():
        all_results_df = pd.read_csv(results_csv)
    else:
        all_results_df = pd.DataFrame(columns=[
            'timestamp', 'experiment_name', 'learning_rate', 
            'batch_size', 'accuracy', 'best_lr', 
            'model_architecture', 'additional_notes'
        ])
    
    # Get configuration
    config = get_config('ensemble')
    config['num_classes'] = len(np.unique(labels))
    
    # Split data for final test set
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Apply PCA
    if len(X_train_full.shape) > 2:
        X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
        n_components = min(512, X_train_full.shape[0] - 1)
        pca = PCA(n_components=n_components)
        X_train_full = pca.fit_transform(X_train_full)
        X_test_final = X_test_final.reshape(X_test_final.shape[0], -1)
        X_test_final = pca.transform(X_test_final)
        config['input_dim'] = n_components
    
    # Create untrained models
    model_types = ['linear', 'deep', 'base']
    individual_models = [
        get_model(model_type, config['input_dim'], config['num_classes'])
        for model_type in model_types
    ]
    
    # Create ensemble with untrained models
    print("\nCreating and training ensemble with untrained components...")
    ensemble_model = EnsembleModel(
        input_dim=config['input_dim'],
        num_classes=config['num_classes'],
        pretrained_models=individual_models
    )
    
    # Train ensemble (all components trained together)
    ensemble_acc, ensemble_state = run_training(
        X_train_full, y_train_full, 0, config, 
        model_type='ensemble', 
        model=ensemble_model
    )
    
    # Final evaluation on held-out test set
    ensemble_model.load_state_dict(ensemble_state)
    ensemble_model.eval()
    test_dataset = GenreDataset(X_test_final, y_test_final)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    _, final_test_acc = evaluate(ensemble_model, test_loader, nn.CrossEntropyLoss(), device)
    
    # Save ensemble results
    new_result = {
        'timestamp': timestamp,
        'experiment_name': f'exp_{timestamp}_joint_ensemble',
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'accuracy': final_test_acc,
        'best_lr': config['learning_rate'],
        'model_architecture': 'joint_ensemble_penultimate',
        'additional_notes': 'Penultimate features - Jointly trained ensemble model'
    }
    all_results_df = pd.concat([all_results_df, pd.DataFrame([new_result])], ignore_index=True)
    
    # Save results CSV with error handling
    max_retries = 3
    for attempt in range(max_retries):
        try:
            all_results_df.to_csv(results_csv, index=False)
            print(f"Results saved to {results_csv}")
            break
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"Unable to save results, file may be open. Attempt {attempt + 1} of {max_retries}")
                time.sleep(1)  # Wait a second before retrying
            else:
                print("Could not save results CSV. Please close any programs that might have the file open.")
                # Save to alternative location
                alt_path = results_dir / f"penultimate_experiments_{timestamp}.csv"
                all_results_df.to_csv(alt_path, index=False)
                print(f"Results saved to alternative location: {alt_path}")
    
    # Save model
    model_path = results_dir / f"joint_ensemble_model_penultimate.pth"
    torch.save(ensemble_model.state_dict(), model_path)
    
    # Save config
    with open(results_dir / "config_penultimate.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    # Print summary
    print("\nJoint Training Experiment Results:")
    print(f"Ensemble Accuracy: {ensemble_acc:.2f}%")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print("\nAll Penultimate Experiments:")
    print(all_results_df.sort_values('timestamp', ascending=False).head().to_string())

if __name__ == "__main__":
    main() 
    