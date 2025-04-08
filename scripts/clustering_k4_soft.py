import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import matplotlib.pyplot as plt
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

def save_label_mapping(label_encoder, cluster_to_genre, output_path):
    """
    Save the label mapping for all classifiers.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Cluster', 'Genre', 'Probability'])
        for cluster_idx, genre_probs in enumerate(cluster_to_genre):
            for genre_idx, prob in enumerate(genre_probs):
                writer.writerow([cluster_idx, label_encoder.inverse_transform([genre_idx])[0], prob])
    print(f"Label mapping saved to {output_path}")


def plot_and_save_confusion_matrix(y_true, y_pred, genre_names, group, output_path):
    """
    Plot and save the confusion matrix for a classifier.
    """
    cm = pd.crosstab(pd.Series(y_true, name='Actual'), pd.Series(y_pred, name='Predicted'))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_names, yticklabels=genre_names)
    plt.title(f'Confusion Matrix for Group {group}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix for Group {group} saved to {output_path}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    train_data = np.load(r"D:\research_project\preprocessed\extracted_train.npz")
    X_train = train_data['X']
    y_train = train_data['y']
    
    test_data = np.load(r"D:\research_project\preprocessed\extracted_test.npz")
    X_test = test_data['X']
    y_test = test_data['y']

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    genre_names = label_encoder.classes_
    num_genres = len(genre_names)

    # Cluster with K=4
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_train)
    
    # Create cluster distribution matrix
    cluster_dist = pd.DataFrame({'Genre': y_train, 'Cluster': clusters})
    cluster_dist = cluster_dist.groupby(['Genre', 'Cluster']).size().unstack(fill_value=0)
    cluster_dist = cluster_dist.div(cluster_dist.sum(axis=1), axis=0)
    
    # Create genre-to-cluster mapping
    genre_to_cluster = cluster_dist.values
    
    # Create cluster-to-genre mapping (transpose and normalize)
    cluster_to_genre = cluster_dist.T.values
    cluster_to_genre = cluster_to_genre / cluster_to_genre.sum(axis=1, keepdims=True)
    
    # Save label mapping
    save_label_mapping(label_encoder, cluster_to_genre, r"D:\research_project\results\label_mapping.csv")
    
    # Create soft labels for group classification
    soft_labels_train = np.array([genre_to_cluster[label_encoder.transform([g])[0]] for g in y_train])
    soft_labels_test = np.array([genre_to_cluster[label_encoder.transform([g])[0]] for g in y_test])

    # Train group classifier
    group_classifier = NoiseAugmentationModel(X_train.shape[1], k)
    optimizer = optim.Adam(group_classifier.parameters(), lr=0.0005)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    # Training loop
    for epoch in range(35):
        group_classifier.train()
        optimizer.zero_grad()
        outputs = torch.log_softmax(group_classifier(torch.tensor(X_train, dtype=torch.float32)), dim=1)
        loss = criterion(outputs, torch.tensor(soft_labels_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        print(f"Group Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Train genre classifiers
    genre_results = {}
    for group in range(k):
        # Get samples for this group
        group_mask = np.argmax(soft_labels_train, axis=1) == group
        X_group = X_train[group_mask]
        y_group = y_train_encoded[group_mask]
        
        if len(X_group) == 0:
            print(f"No samples for group {group}")
            continue
            
        # Create genre classifier
        genre_classifier = NoiseAugmentationModel(X_train.shape[1], num_genres)
        optimizer = optim.Adam(genre_classifier.parameters(), lr=0.0005)
        
        # Use cluster-to-genre distribution as target
        y_group_soft = np.array([cluster_to_genre[group] for _ in y_group])
        
        # Training loop
        for epoch in range(35):
            genre_classifier.train()
            optimizer.zero_grad()
            outputs = torch.log_softmax(genre_classifier(torch.tensor(X_group, dtype=torch.float32)), dim=1)
            loss = criterion(outputs, torch.tensor(y_group_soft, dtype=torch.float32))
            loss.backward()
            optimizer.step()
            print(f"Group {group} Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # Evaluate
        test_mask = np.argmax(soft_labels_test, axis=1) == group
        if sum(test_mask) > 0:
            # Create evaluation targets using cluster-to-genre distribution
            y_eval = np.array([cluster_to_genre[group] for _ in range(sum(test_mask))])
            
            # Evaluation
            genre_classifier.eval()
            with torch.no_grad():
                outputs = torch.softmax(genre_classifier(torch.tensor(X_test[test_mask], dtype=torch.float32)), dim=1)
                y_pred = torch.argmax(outputs, dim=1).numpy()
                y_true = np.argmax(y_eval, axis=1)
                accuracy = (y_pred == y_true).mean()
                kl_div = criterion(torch.log(outputs), torch.tensor(y_eval)).item()
                
                print(f"\nGroup {group} Evaluation:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"KL Divergence: {kl_div:.4f}")
                
                # Save confusion matrix
                plot_and_save_confusion_matrix(y_true, y_pred, genre_names, group, 
                                               rf"D:\research_project\results\confusion_matrix_group_{group}.png")
                
                genre_results[group] = {
                    'accuracy': accuracy,
                    'kl_div': kl_div,
                    'num_samples': sum(test_mask)
                }
    
    # Save results
    with open(r"D:\research_project\results\metrics.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Group', 'Accuracy', 'KL Divergence', 'Samples'])
        for group, res in genre_results.items():
            writer.writerow([group, res['accuracy'], res['kl_div'], res['num_samples']])
        # Calculate averages
        if genre_results:
            avg_acc = sum(r['accuracy'] for r in genre_results.values()) / len(genre_results)
            avg_kl = sum(r['kl_div'] for r in genre_results.values()) / len(genre_results)
            writer.writerow(['Average', avg_acc, avg_kl, ''])

if __name__ == "__main__":
    main()