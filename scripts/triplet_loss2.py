import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute pairwise distances
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)
        # Triplet loss: max(0, margin + pos_distance - neg_distance)
        loss = torch.clamp(self.margin + pos_distance - neg_distance, min=0.0)
        return loss.mean()

# Noise Augmentation Model with Classification Head
class NoiseAugmentationModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, num_classes=10):
        super().__init__()
        # Embedding layers
        self.embedding_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim)  # Output embeddings
        )
        # Classification head
        self.classification_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, classify=False):
        embeddings = self.embedding_layers(x)
        if classify:
            return self.classification_head(embeddings)  # Return class logits
        return embeddings  # Return embeddings for triplet loss

# Create Triplet Dataset for Triplet Loss
def create_triplet_dataset(features, labels, batch_size=32):
    triplets = []
    num_samples = len(features)
    
    rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility

    # Create triplets (anchor, positive, negative)
    for i in range(num_samples):
        anchor = features[i]
        positive_indices = np.where(labels == labels[i])[0]
        negative_indices = np.where(labels != labels[i])[0]
        
        # Remove the anchor itself from positive_indices
        positive_indices = positive_indices[positive_indices != i]
        
        # Only proceed if enough positives and negatives
        if len(positive_indices) >= 14 and len(negative_indices) >= 14:
            # Randomly select 8 positives and 8 negatives (without replacement, reproducible)
            selected_positives = rng.choice(positive_indices, 14, replace=False)
            selected_negatives = rng.choice(negative_indices, 14, replace=False)
            # Create all combinations
            for pos_idx in selected_positives:
                for neg_idx in selected_negatives:
                    positive = features[pos_idx]
                    negative = features[neg_idx]
                    triplets.append((anchor, positive, negative))
    
    # Convert to tensors
    anchors = torch.stack([torch.tensor(t[0], dtype=torch.float32) for t in triplets])
    positives = torch.stack([torch.tensor(t[1], dtype=torch.float32) for t in triplets])
    negatives = torch.stack([torch.tensor(t[2], dtype=torch.float32) for t in triplets])
    
    # Create dataset and dataloader
    dataset = TensorDataset(anchors, positives, negatives)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Training with Triplet Loss
def train_with_triplet_loss(train_features, train_labels, model, config):
    # Create dataloader for triplets
    triplet_loader = create_triplet_dataset(train_features, train_labels, batch_size=config['batch_size'])
    
    optimizer = torch.optim.Adam(model.embedding_layers.parameters(), lr=config['learning_rate'])
    criterion = TripletLoss(margin=1.0)

    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0.0

        for anchor, positive, negative in triplet_loader:
            anchor, positive, negative = anchor.to(config['device']), positive.to(config['device']), negative.to(config['device'])

            optimizer.zero_grad()
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)
            loss = criterion(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config['max_epochs']}, Triplet Loss: {total_loss / len(triplet_loader):.4f}")

# Fine-Tuning with Classification Loss
def fine_tune_with_classification(train_loader, val_loader, model, config):
    # Use a lower learning rate for the classification head
    optimizer = torch.optim.Adam([
        {'params': model.embedding_layers.parameters(), 'lr': config['learning_rate']* 0.1},  # Lower LR for embedding layers
        {'params': model.classification_head.parameters(), 'lr': config['learning_rate'] }  # Lower LR for classification head
    ])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config['device']), y_batch.to(config['device'])

            optimizer.zero_grad()
            logits = model(X_batch, classify=True)  # Use classification head
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(config['device']), y_val.to(config['device'])
                logits = model(X_val, classify=True)
                val_loss += criterion(logits, y_val).item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == y_val).sum().item()
                total += y_val.size(0)

        val_accuracy = correct / total
        print(f"Epoch {epoch + 1}/{config['max_epochs']}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

# Test the Model
def test_model(test_loader, model, config):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(config['device']), y_batch.to(config['device'])
            logits = model(X_batch, classify=True)
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

    # Split training set into training and validation sets
    train_size = int(0.80 * len(X_train))
    val_size = len(X_train) - train_size
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
    num_classes = len(np.unique(y_train))  # Number of unique genres
    config = {
        'learning_rate': 1e-4,
        'max_epochs': 20,
        'batch_size': 32,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # Initialize model
    model = NoiseAugmentationModel(input_dim=input_dim, embedding_dim=128, num_classes=num_classes).to(config['device'])

    # Train with triplet loss
    train_with_triplet_loss(X_train, y_train, model, config)

    # Fine-tune with classification loss
    fine_tune_with_classification(train_loader, val_loader, model, config)

    # Test the model
    test_model(test_loader, model, config)