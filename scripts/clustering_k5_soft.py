import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import csv

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

def plot_confusion_matrix(y_true, y_pred, genre_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=genre_names, yticklabels=genre_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def run_training_soft(X_train, y_train, X_val, y_val, config, model_class, group=None):
    model = model_class(input_dim=X_train.shape[1], num_classes=config['num_classes'])
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                         weight_decay=config['weight_decay'])
    
    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(config['max_epochs']):
        model.train()
        optimizer.zero_grad()
        outputs = torch.log_softmax(model(torch.tensor(X_train, dtype=torch.float32)), dim=1)
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_outputs = torch.log_softmax(model(torch.tensor(X_val, dtype=torch.float32)), dim=1)
            val_loss = criterion(val_outputs, torch.tensor(y_val, dtype=torch.float32))
            val_losses.append(val_loss.item())
            
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = (val_preds == torch.tensor(np.argmax(y_val, axis=1))).float().mean().item()
            val_accuracies.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    break
        
        prefix = f"Group {group} - " if group is not None else ""
        print(f"{prefix}Epoch {epoch + 1}/{config['max_epochs']}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def run_training(X_train, y_train, X_val, y_val, config, model_class, group=None):
    model = model_class(input_dim=X_train.shape[1], num_classes=config['num_classes'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                         weight_decay=config['weight_decay'])
    
    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(config['max_epochs']):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.tensor(X_val, dtype=torch.float32))
            val_loss = criterion(val_outputs, torch.tensor(y_val, dtype=torch.long))
            val_losses.append(val_loss.item())
            
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = (val_preds == torch.tensor(y_val)).float().mean().item()
            val_accuracies.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    break
        
        prefix = f"Group {group} - " if group is not None else ""
        print(f"{prefix}Epoch {epoch + 1}/{config['max_epochs']}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def evaluate_model(model, X_test, y_test, genre_names, group=None):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test, dtype=torch.float32))
        _, predictions = torch.max(outputs, 1)
        
        y_true = y_test
        y_pred = predictions.numpy()
        
        genre_names = [str(name) for name in genre_names]
        
        prefix = f"Group {group} - " if group is not None else ""
        print(f"\n{prefix}Classification Report:")
        print(classification_report(y_true, y_pred, target_names=genre_names))
        
        plot_confusion_matrix(y_true, y_pred, genre_names)
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'accuracy': (y_pred == y_true).mean()
        }

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    
    # Load datasets
    train_data = np.load(r"D:\research_project\preprocessed\extracted_train.npz")
    X_train = train_data['X']
    y_train = train_data['y']
    genres = train_data['genres']
    
    test_data = np.load(r"D:\research_project\preprocessed\extracted_test.npz")
    X_test = test_data['X']
    y_test = test_data['y']

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    genre_names = [str(name) for name in label_encoder.classes_]

    # Configuration
    config = {
        'batch_size': 64,
        'learning_rate': 0.0005,
        'max_epochs': 35,
        'patience': 7,
        'weight_decay': 0.0
    }

    # Step 1: K-Means Clustering with K=5
    print("Applying K-Means Clustering with K=5...")
    k = 5  
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_train)
    silhouette = silhouette_score(X_train, clusters)
    print(f"K = {k}, Silhouette Score: {silhouette:.4f}")

    # Save cluster results
    np.savez(rf"D:\research_project\results\kmeans_k{k}.npz", 
             clusters=clusters, 
             centers=kmeans.cluster_centers_)

    # Analyze cluster distribution
    df = pd.DataFrame({'Genre': y_train, 'Cluster': clusters})
    percentages = df.groupby(['Genre', 'Cluster']).size().groupby(level=0).apply(lambda x: x / x.sum())
    print(f"\nCluster Percentages for K = {k}:\n{percentages}")
    percentages.to_csv(rf"D:\research_project\results\cluster_percentages_k{k}.csv")

    # Step 2: Design Hierarchical Classification System with Soft Mapping
    print("\nDesigning Hierarchical Classification System with Soft Mapping...")
    cluster_percentages = percentages.unstack(fill_value=0)
    print(f"Cluster Percentages (Soft Mapping):\n{cluster_percentages}")

    # Prepare soft group labels for training
    soft_group_labels_train = np.array([cluster_percentages.loc[genre].values for genre in y_train])
    soft_group_labels_test = np.array([cluster_percentages.loc[genre].values for genre in y_test])

    # Step 3: Hierarchical Classification
    print("\nTraining Group-Level Classifier with Cross-Validation...")
    
    # Add noise augmentation
    noisy_X_train = add_noise(X_train)
    augmented_X_train = np.concatenate((X_train, noisy_X_train), axis=0)
    augmented_soft_group_labels = np.concatenate((soft_group_labels_train, soft_group_labels_train), axis=0)

    # Create proper hard labels for StratifiedKFold
    hard_labels = np.argmax(augmented_soft_group_labels.squeeze(), axis=1)
    
    # Cross-validation for group classifier
    config['num_classes'] = k
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    group_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(augmented_X_train, hard_labels)):
        print(f"\nGroup Classifier - Fold {fold + 1}/5")
        X_train_fold, X_val_fold = augmented_X_train[train_idx], augmented_X_train[val_idx]
        y_train_fold = augmented_soft_group_labels[train_idx].squeeze()
        y_val_fold = augmented_soft_group_labels[val_idx].squeeze()
        
        fold_result = run_training_soft(X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                                      config, NoiseAugmentationModel)
        group_fold_results.append(fold_result)
    
    # Train final group classifier on all data
    group_classifier = NoiseAugmentationModel(input_dim=X_train.shape[1], num_classes=k)
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(group_classifier.parameters(), lr=config['learning_rate'], 
                         weight_decay=config['weight_decay'])
    
    for epoch in range(config['max_epochs']):
        group_classifier.train()
        optimizer.zero_grad()
        outputs = torch.log_softmax(group_classifier(torch.tensor(augmented_X_train, dtype=torch.float32)), dim=1)
        loss = criterion(outputs, torch.tensor(augmented_soft_group_labels.squeeze(), dtype=torch.float32))
        loss.backward()
        optimizer.step()
        print(f"Group Classifier - Final Training - Epoch {epoch + 1}/{config['max_epochs']}, Loss: {loss.item():.4f}")
    
    # Evaluate group classifier
    group_names = [f"Group {i}" for i in range(k)]
    group_test_result = evaluate_model(group_classifier, X_test, np.argmax(soft_group_labels_test.squeeze(), axis=1), group_names)

    # Step 4: Genre-Level Classifiers
    print("\nTraining Genre-Level Classifiers...")
    genre_classifiers = {}
    genre_results = {}
    
    for group in range(k):
        group_indices_train = np.where(np.argmax(soft_group_labels_train.squeeze(), axis=1) == group)[0]
        group_indices_test = np.where(np.argmax(soft_group_labels_test.squeeze(), axis=1) == group)[0]
        
        if len(group_indices_train) == 0:
            print(f"\nNo samples found for Group {group}, skipping...")
            continue
            
        X_group_train = X_train[group_indices_train]
        y_group_train = y_train_encoded[group_indices_train]
        X_group_test = X_test[group_indices_test]
        y_group_test = y_test_encoded[group_indices_test]
        
        # Re-encode labels for current group
        group_label_encoder = LabelEncoder()
        y_group_train_encoded = group_label_encoder.fit_transform(y_group_train)
        y_group_test_encoded = group_label_encoder.transform(y_group_test)
        
        # Get genre names for this group
        group_genre_indices = group_label_encoder.classes_
        group_genre_names = [genre_names[i] for i in group_genre_indices]
        num_classes = len(group_genre_indices)
        
        # Add noise augmentation
        noisy_X_group = add_noise(X_group_train)
        augmented_X_group = np.concatenate((X_group_train, noisy_X_group), axis=0)
        augmented_y_group = np.concatenate((y_group_train_encoded, y_group_train_encoded), axis=0)
        
        # Cross-validation for genre classifier
        config['num_classes'] = num_classes
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        genre_fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(augmented_X_group, augmented_y_group)):
            print(f"\nGroup {group} Genre Classifier - Fold {fold + 1}/5")
            X_train_fold, X_val_fold = augmented_X_group[train_idx], augmented_X_group[val_idx]
            y_train_fold, y_val_fold = augmented_y_group[train_idx], augmented_y_group[val_idx]
            
            # Convert to one-hot for soft training
            y_train_fold_onehot = np.eye(num_classes)[y_train_fold]
            y_val_fold_onehot = np.eye(num_classes)[y_val_fold]
            
            fold_result = run_training_soft(X_train_fold, y_train_fold_onehot, X_val_fold, y_val_fold_onehot,
                                          config, NoiseAugmentationModel, group)
            genre_fold_results.append(fold_result)
        
        # Train final genre classifier on all data
        genre_classifier = NoiseAugmentationModel(input_dim=X_group_train.shape[1], 
                                               num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(genre_classifier.parameters(), 
                             lr=config['learning_rate'], 
                             weight_decay=config['weight_decay'])
        
        for epoch in range(config['max_epochs']):
            genre_classifier.train()
            optimizer.zero_grad()
            outputs = genre_classifier(torch.tensor(augmented_X_group, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(augmented_y_group, dtype=torch.long))
            loss.backward()
            optimizer.step()
            print(f"Group {group} Genre Classifier - Final Training - Epoch {epoch + 1}/{config['max_epochs']}, Loss: {loss.item():.4f}")
        
        # Evaluate genre classifier
        if len(group_indices_test) > 0:  # Only evaluate if test samples exist
            genre_test_result = evaluate_model(genre_classifier, X_group_test, 
                                             y_group_test_encoded, group_genre_names, group)
            genre_results[group] = genre_test_result
        else:
            print(f"\nNo test samples found for Group {group}, skipping evaluation")
            genre_results[group] = {'accuracy': 0.0}
    
    # Save performance metrics
    with open(r"D:\research_project\results\hierarchical_classification_results_k5_soft.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Level', 'Group', 'Accuracy'])
        writer.writerow(['Group', 'All', group_test_result['accuracy']])
        for group, result in genre_results.items():
            writer.writerow(['Genre', group, result['accuracy']])
    
    print("Results saved successfully.")

if __name__ == "__main__":
    main()