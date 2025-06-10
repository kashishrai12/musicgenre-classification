import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# Multi-task Model with two output heads
class MultiTaskNoiseAugmentationModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, num_classes=10):
        super().__init__()
        self.embedding_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim)
        )
        self.classification_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embedding = self.embedding_layers(x)
        class_out = self.classification_head(embedding)
        return embedding, class_out

# Create Triplet Dataset for Triplet Loss
def create_triplet_dataset(features, labels, batch_size=32, n_pos=16, n_neg=16):
    triplets = []
    anchor_indices = []
    num_samples = len(features)
    rng = np.random.default_rng(seed=42)
    for i in range(num_samples):
        anchor = features[i]
        positive_indices = np.where(labels == labels[i])[0]
        negative_indices = np.where(labels != labels[i])[0]
        positive_indices = positive_indices[positive_indices != i]
        if len(positive_indices) >= n_pos and len(negative_indices) >= n_neg:
            selected_positives = rng.choice(positive_indices, n_pos, replace=False)
            selected_negatives = rng.choice(negative_indices, n_neg, replace=False)
            for pos_idx in selected_positives:
                for neg_idx in selected_negatives:
                    positive = features[pos_idx]
                    negative = features[neg_idx]
                    triplets.append((anchor, positive, negative))
                    anchor_indices.append(i)
    anchor_t = torch.stack([torch.tensor(t[0], dtype=torch.float32) for t in triplets])
    positive_t = torch.stack([torch.tensor(t[1], dtype=torch.float32) for t in triplets])
    negative_t = torch.stack([torch.tensor(t[2], dtype=torch.float32) for t in triplets])
    anchor_indices = torch.tensor(anchor_indices, dtype=torch.long)
    dataset = TensorDataset(anchor_t, positive_t, negative_t, anchor_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Multi-task Training Loop with Triplet Loss
def train_multitask_triplet(train_features, train_labels, model, config, alpha=0.5, n_pos=16, n_neg=16):
    triplet_loader = create_triplet_dataset(train_features, train_labels, batch_size=config['batch_size'], n_pos=n_pos, n_neg=n_neg)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    triplet_criterion = TripletLoss(margin=1.0)
    classification_criterion = nn.CrossEntropyLoss()

    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0.0
        for anchor, positive, negative, anchor_idx in triplet_loader:
            anchor, positive, negative, anchor_idx = (
                anchor.to(config['device']),
                positive.to(config['device']),
                negative.to(config['device']),
                anchor_idx.to(config['device'])
            )
            class_labels = torch.tensor(train_labels[anchor_idx.cpu().numpy()], dtype=torch.long).to(config['device'])

            optimizer.zero_grad()
            anchor_emb, anchor_logits = model(anchor)
            positive_emb, _ = model(positive)
            negative_emb, _ = model(negative)
            triplet_loss = triplet_criterion(anchor_emb, positive_emb, negative_emb)
            classification_loss = classification_criterion(anchor_logits, class_labels)
            loss = alpha * triplet_loss + (1 - alpha) * classification_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{config['max_epochs']}, Multi-task Loss: {total_loss / len(triplet_loader):.4f}")

# Test the Model
def test_model(test_loader, model, config):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(config['device']), y_batch.to(config['device'])
            _, logits = model(X_batch)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")

# Example Usage
if __name__ == "__main__":
    # Load data
    train_data = np.load(r"D:\research_project\preprocessed\extracted_train.npz")
    print(f"Keys in train_data: {train_data.keys()}")

    test_data = np.load(r"D:\research_project\preprocessed\extracted_test.npz")
    print(f"Keys in test_data: {test_data.keys()}")

    # Use the correct keys to load features and labels
    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']

    # Convert string labels to integer indices if needed
    if y_train.dtype.type is np.str_ or y_test.dtype.type is np.str_:
        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_train = np.array([label_to_idx[label] for label in y_train])
        y_test = np.array([label_to_idx[label] for label in y_test])

    # Split training set into training and validation sets
    train_size = int(0.80 * len(X_train))
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    X_train_split, y_train_split = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # Create data loaders
    batch_size = 32
    train_dataset = TensorDataset(torch.tensor(X_train_split, dtype=torch.float32),
                                 torch.tensor(y_train_split, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model configuration
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    config = {
        'learning_rate': 1e-3,
        'max_epochs': 20,
        'batch_size': 32,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # Initialize multi-task model
    model = MultiTaskNoiseAugmentationModel(input_dim=input_dim, embedding_dim=128, num_classes=num_classes).to(config['device'])

    # Train with multi-task triplet loss (alpha=0.5 for equal weighting)
    train_multitask_triplet(X_train, y_train, model, config, alpha=0.3, n_pos=16, n_neg=16)

    # Test the model
    test_model(test_loader, model, config)