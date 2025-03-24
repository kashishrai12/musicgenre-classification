from base_training2 import run_training, evaluate_model, save_results, set_seed
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import csv

class WeightClippingModel(nn.Module):
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

def clip_weights(model, clip_value=0.5):
    for param in model.parameters():
        param.data.clamp_(-clip_value, clip_value)

def plot_confusion_matrix(y_true, y_pred, genre_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_names, yticklabels=genre_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def run_training_with_clipping(X_train, y_train, X_val, y_val, config, model_class, experiment_name, genre_names):
    # Initialize model, optimizer, and loss function
    model = model_class(input_dim=X_train.shape[1], num_classes=config['num_classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(config['max_epochs']):
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
        loss.backward()
        optimizer.step()
        
        # Clip weights
        clip_weights(model, clip_value=0.5)
        
        train_losses.append(loss.item())
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.tensor(X_val, dtype=torch.float32))
            val_loss = criterion(val_outputs, torch.tensor(y_val, dtype=torch.long))
            val_losses.append(val_loss.item())
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = (val_preds == torch.tensor(y_val)).float().mean().item()
            val_accuracies.append(val_acc)
        
        # Print training progress
        train_acc = (torch.max(outputs, 1)[1] == torch.tensor(y_train)).float().mean().item()
        print(f"Epoch {epoch + 1}/{config['max_epochs']}, Train Accuracy: {train_acc:.4f}, Train Loss: {loss.item():.4f}, Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss.item():.4f}")
        
        # Check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

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
        'max_epochs': 35,  # Set max_epochs to 35
        'patience': 7,  # Set patience to 7
        'num_classes': len(np.unique(train_labels)),
        'experiment_name': 'exp_weight_clipping',
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
        
        fold_result = run_training_with_clipping(X_train, y_train, X_val, y_val, config, WeightClippingModel, experiment_name, genre_names)
        fold_results.append(fold_result)
    
    # Calculate average metrics
    avg_best_val_acc = np.mean([max(result['val_accuracies']) for result in fold_results]) * 100
    avg_train_losses = np.mean([np.mean(result['train_losses']) for result in fold_results])
    avg_val_losses = np.mean([np.mean(result['val_losses']) for result in fold_results])
    
    # Evaluate on test set
    test_result = evaluate_model(train_features, train_labels, test_features, test_labels, config, WeightClippingModel, genre_names)
    avg_test_acc = test_result['test_accuracy']
    
    # Save results
    with open(r"D:\research_project\average_experiment_results.csv", 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([experiment_name, avg_best_val_acc, avg_train_losses, avg_val_losses, avg_test_acc])
    
    # Plot confusion matrix for the test set
    plot_confusion_matrix(test_result['y_true'], test_result['y_pred'], genre_names)

if __name__ == "__main__":
    main()