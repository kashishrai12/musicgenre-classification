import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) + \
               label * torch.pow(distance, 2)
        return loss.mean()

# Multi-task Model with three output heads
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
        self.contrastive_head = nn.Identity()  # For contrastive loss
        self.classification_head1 = nn.Linear(embedding_dim, num_classes)  # For x1
        self.classification_head2 = nn.Linear(embedding_dim, num_classes)  # For x2

    def forward(self, x):
        embedding = self.embedding_layers(x)
        contrastive_out = self.contrastive_head(embedding)
        class_out1 = self.classification_head1(embedding)
        class_out2 = self.classification_head2(embedding)
        return contrastive_out, class_out1, class_out2

# Create Pairs Dataset for Contrastive Loss
def create_pairs_dataset(features, labels, batch_size=32):
    pairs = []
    pair_labels = []
    anchor_indices = []
    pair2_indices = []
    num_samples = len(features)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            pairs.append((features[i], features[j]))
            pair_labels.append(1 if labels[i] == labels[j] else 0)
            anchor_indices.append(i)  # Track anchor index (for x1)
            pair2_indices.append(j)   # Track index for x2
    pair1 = torch.stack([torch.tensor(p[0], dtype=torch.float32) for p in pairs])
    pair2 = torch.stack([torch.tensor(p[1], dtype=torch.float32) for p in pairs])
    pair_labels = torch.tensor(pair_labels, dtype=torch.float32)
    anchor_indices = torch.tensor(anchor_indices, dtype=torch.long)
    pair2_indices = torch.tensor(pair2_indices, dtype=torch.long)
    dataset = TensorDataset(pair1, pair2, pair_labels, anchor_indices, pair2_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Multi-task Training Loop (with two classification heads and weighted losses)
def train_multitask(train_features, train_labels, model, config, alpha=0.3, w1=0.5, w2=0.5):
    pair_loader = create_pairs_dataset(train_features, train_labels, batch_size=config['batch_size'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    contrastive_criterion = ContrastiveLoss(margin=1.0)
    classification_criterion = nn.CrossEntropyLoss()

    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0.0
        for x1, x2, label, anchor_idx, pair2_idx in pair_loader:
            x1 = x1.to(config['device'])
            x2 = x2.to(config['device'])
            label = label.to(config['device'])
            anchor_idx = anchor_idx.to(config['device'])
            pair2_idx = pair2_idx.to(config['device'])

            class_labels1 = torch.tensor(train_labels[anchor_idx.cpu().numpy()], dtype=torch.long).to(config['device'])
            class_labels2 = torch.tensor(train_labels[pair2_idx.cpu().numpy()], dtype=torch.long).to(config['device'])

            optimizer.zero_grad()
            out1, class_out1, _ = model(x1)
            out2, _, class_out2 = model(x2)
            contrastive_loss = contrastive_criterion(out1, out2, label)
            classification_loss1 = classification_criterion(class_out1, class_labels1)
            classification_loss2 = classification_criterion(class_out2, class_labels2)
            classification_loss = w1 * classification_loss1 + w2 * classification_loss2
            loss = alpha * contrastive_loss + (1 - alpha) * classification_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{config['max_epochs']}, Multi-task Loss: {total_loss / len(pair_loader):.4f}")

# Test the Model (using head1 for classification)
def test_model(test_loader, model, config):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(config['device']), y_batch.to(config['device'])
            _, logits1, _ = model(X_batch)
            _, predicted = torch.max(logits1, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

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

    # Try different weights for classification losses
    weight_pairs = [
        (0.5, 0.5),
        (0.7, 0.3),
        (0.3, 0.7),
        (0.9, 0.1),
        (0.1, 0.9)
    ]
    test_accuracies = []
    for w1, w2 in weight_pairs:
        print(f"\nTraining with classification weights w1={w1}, w2={w2}")
        model = MultiTaskNoiseAugmentationModel(input_dim=input_dim, embedding_dim=128, num_classes=num_classes).to(config['device'])
        train_multitask(X_train, y_train, model, config, alpha=0.3, w1=w1, w2=w2)
        acc = test_model(test_loader, model, config)
        test_accuracies.append(acc)

    # Plot the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    w1s = [w[0] for w in weight_pairs]
    plt.plot(w1s, test_accuracies, marker='o')
    plt.xlabel('w1 (Classification Loss Weight for x1)')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Classification Loss Weight w1 (alpha=0.3)')
    plt.grid(True)
    plt.show()