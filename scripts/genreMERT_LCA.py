import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# Attention Modules
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)



# Main Deep Attention Network
class AttentionBasedDeepNet(nn.Module):
    def __init__(self, input_dim, num_genres, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.attn1 = SpatialAttention(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.attn2 = SpatialAttention(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.attn3 = SpatialAttention(256)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_genres)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        a1 = self.attn1.attention(x1)

        x2 = torch.relu(self.conv2(self.attn1(x1)))
        a2 = self.attn2.attention(x2)

        x3 = torch.relu(self.conv3(self.attn2(x2)))
        a3 = self.attn3.attention(x3)

        x_out = self.attn3(x3)
        x_out = self.gap(x_out).squeeze(-1)
        x_out = torch.relu(self.fc1(x_out))
        x_out = self.dropout(x_out)
        x_out = self.fc2(x_out)
        return self.softmax(x_out), [a1, a2, a3]



# Counterfactual Branch 
class CounterfactualAttentionBranch(nn.Module):
    def __init__(self, input_dim, num_genres):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.attn1 = SpatialAttention(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.attn2 = SpatialAttention(128)
        # Add global average pooling and final layer to match dimensions with main branch
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_genres)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        a1 = self.attn1.attention(x1)

        x2 = torch.relu(self.conv2(self.attn1(x1)))
        a2 = self.attn2.attention(x2)

        x_out = self.attn2(x2)
        # Use global average pooling like in the main branch
        x_out = self.gap(x_out).squeeze(-1)
        # Project to the same number of genres
        x_out = self.fc(x_out)
        x_out = self.softmax(x_out)
        return x_out, [a1, a2]



# Complete Model
class GenreMERT_LCA(nn.Module):
    def __init__(self, input_dim, num_genres, dropout_rate=0.5):
        super().__init__()
        self.deep_net = AttentionBasedDeepNet(input_dim, num_genres, dropout_rate)
        self.counterfactual_branch = CounterfactualAttentionBranch(input_dim, num_genres)

    def forward(self, x):
        y_main, A_main = self.deep_net(x)
        y_cf, A_cf = self.counterfactual_branch(x)
        return y_main, A_main, y_cf, A_cf


# Loss Function
def compute_losses(y_main, y_cf, A_main, A_cf, y_true, lambda_weights):
    L_main_ce = nn.CrossEntropyLoss()(y_main, y_true)
    L_cf_ce = nn.CrossEntropyLoss()(y_cf, y_true)
    L_effect_ce = nn.CrossEntropyLoss()(y_main - y_cf, y_true)

    L_cf_ent = -torch.sum(y_cf * torch.log(y_cf + 1e-8), dim=1).mean()
    L_main_ent = -torch.sum(y_main * torch.log(y_main + 1e-8), dim=1).mean()

    A_diff = [torch.mean((A_main[i] - A_cf[i])**2) for i in range(min(len(A_main), len(A_cf)))]
    L_att_1 = torch.mean(torch.stack(A_diff))

    L_total = (
        lambda_weights['main_ce'] * L_main_ce +
        lambda_weights['effect_ce'] * L_effect_ce +
        lambda_weights['cf_ce'] * L_cf_ce -
        lambda_weights['cf_ent'] * L_cf_ent -
        lambda_weights['att_1'] * L_att_1 -
        lambda_weights['main_ent'] * L_main_ent
    )
    return L_total



# Training Function
def train_genreMERT_LCA(train_loader, val_loader, input_dim, num_genres, num_epochs=300, initial_epochs=20, learning_rate=1e-4, dropout_rate=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenreMERT_LCA(input_dim, num_genres, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lambda_weights = {
        'main_ce': 0.7,
        'effect_ce': 1.0,
        'cf_ce': 0.6,
        'cf_ent': 2.0,
        'att_1': 2.0,
        'main_ent': 1.5
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_main, A_main, y_cf, A_cf = model(X_batch)

            if epoch > initial_epochs:
                loss = compute_losses(y_main, y_cf, A_main, A_cf, y_batch, lambda_weights)
            else:
                # Only use main classification loss for initial epochs
                loss = nn.CrossEntropyLoss()(y_main, y_batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                
                y_main, A_main, y_cf, A_cf = model(X_val)
                
                # Calculate validation loss
                if epoch > initial_epochs:
                    val_loss += compute_losses(y_main, y_cf, A_main, A_cf, y_val, lambda_weights).item()
                else:
                    val_loss += nn.CrossEntropyLoss()(y_main, y_val).item()
                
                # Calculate accuracy
                _, predicted = torch.max(y_main.data, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()

        val_accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    return model



# Testing Function
def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            y_main, _, _, _ = model(X_test)
            _, predicted = torch.max(y_main.data, 1)
            
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return test_accuracy, all_predictions, all_labels



if __name__ == "__main__":
    # Load data
    train_data = np.load(r"D:\research_project\preprocessed\byola_train.npz")
    test_data = np.load(r"D:\research_project\preprocessed\byola_test.npz")

    X_train = torch.tensor(train_data['X_train'], dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.tensor(train_data['y_train'], dtype=torch.long)
    X_test = torch.tensor(test_data['X_test'], dtype=torch.float32).permute(0, 2, 1)
    y_test = torch.tensor(test_data['y_test'], dtype=torch.long)

    # Sanity Check
    print("X_train shape:", X_train.shape)  # Should be (batch, channels, time)
    input_dim = X_train.shape[1]            # Get the actual input dimension from data

    # Split data
    train_size = int(0.75 * len(X_train))
    val_size = len(X_train) - train_size
    train_dataset, val_dataset = random_split(TensorDataset(X_train, y_train), [train_size, val_size])

    # DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Print dataset information
    print(f"Total samples: {len(X_train)}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of genres: {len(torch.unique(y_train))}")
    
    # Train & Evaluate
    num_genres = len(torch.unique(y_train))  # Get number of genres from data
    model = train_genreMERT_LCA(train_loader, val_loader, input_dim, num_genres)
    
    # Test the model
    test_accuracy, predictions, true_labels = test_model(model, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), "genreMERT_LCA_model.pth")
    print("Model saved successfully!")