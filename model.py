import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalImageCNN(nn.Module):
    def __init__(self, output_dim=128):
        super(MedicalImageCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # [B, 64, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):  # x: [B, 1, H, W]
        x = self.features(x)
        x = self.fc(x)
        return x  # shape: [B, output_dim]


class CNNToRNA(nn.Module):
    """
    CNN to RNA regressor. Takes a batch of images and outputs gene expression values.
    Meant to be used as a wrapper for a CNN encoder
    """
    def __init__(self, cnn_encoder, embedding_dim=128, output_dim=1000):
        super().__init__()
        self.encoder = cnn_encoder
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, images):  # [B, N, 1, H, W]
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)  # [B*N, 1, H, W]

        features = self.encoder(images)      # [B*N, emb_dim]
        features = features.view(B, N, -1)   # [B, N, emb_dim]

        pooled = features.mean(dim=1)        # [B, emb_dim]
        return self.regressor(pooled)        # [B, num_genes]

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for images, _, gene_values, _ in train_loader:
            images=torch.stack(images)
            images = images.to(device)         # [B, N, 1, H, W]
            gene_values = gene_values.to(device)  # [B, num_genes]

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, gene_values)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for images, _, gene_values, _ in val_loader:
                images = images.to(device)
                gene_values = gene_values.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, gene_values)
                running_val_loss += val_loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    return train_losses, val_losses


def test_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, _, gene_values, _ in test_loader:
            images = images.to(device)            # [B, N, 1, H, W]
            gene_values = gene_values.to(device)  # [B, num_genes]

            outputs = model(images)               # [B, num_genes]
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(gene_values.cpu().numpy())

    return np.concatenate(all_predictions), np.concatenate(all_labels)