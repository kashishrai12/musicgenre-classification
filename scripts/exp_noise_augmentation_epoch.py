from base_training import run_training, save_results, set_seed
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

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

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def run_training_with_noise(features, labels, config, model_class, experiment_name, genre_names):
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
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
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
                # Add noise to the training data during each epoch
                X_batch_noisy = torch.FloatTensor(add_noise(X_batch.numpy(), noise_factor=0.1))
                optimizer.zero_grad()
                outputs = model(X_batch_noisy)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += y_batch.size(0)
                correct_train += (predicted == y_batch).sum().item()
            
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)
            train_losses.append(epoch_train_loss / len(train_loader))
            
            model.eval()
            correct_val = 0
            total_val = 0
            epoch_val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    epoch_val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += y_batch.size(0)
                    correct_val += (predicted == y_batch).sum().item()
            
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
                outputs = model(X_batch)
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

def main():
    set_seed(42)  # Set a fixed random seed for reproducibility
    data_path = r"D:\research_project\preprocessed\byola_features.npz"
    data = np.load(data_path)
    features = data['features']
    labels = data['labels']
    genre_names = data['genres']  # Assuming genres are stored in the npz file
    
    config = {
        'batch_size': 64,
        'learning_rate': 0.0005,
        'max_epochs': 50,
        'patience': 7,
        'num_classes': len(np.unique(labels)),
        'experiment_name': 'exp_noise_augmentation_epoch',
        'weight_decay': 0.0
    }
    
    experiment_name = config['experiment_name']
    fold_results = run_training_with_noise(features, labels, config, NoiseAugmentationModel, experiment_name, genre_names)
    save_results(experiment_name, fold_results, config, genre_names)

if __name__ == "__main__":
    main()