import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create figures directory
os.makedirs("figures", exist_ok=True)

class DenoisingUNetLike(nn.Module):
    def __init__(self):
        super(DenoisingUNetLike, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU()
        )

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.pool2(x3)
        x5 = self.bottleneck(x4)
        x6 = self.up2(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.dec2(x6)
        x7 = self.up1(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.dec1(x7)
        return self.output_layer(x7)

def normalize_images(images):
    return (images - images.min()) / (images.max() - images.min())

def train_model(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        if (epoch + 1) % 5 == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

def evaluate_test_set(model, test_loader):
    model.eval()
    mse = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            mse += loss.item()
    mse /= len(test_loader)
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse:.4f}")

def plot_loss(train_losses, val_losses, fold):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f"Training vs Validation Loss (Fold {fold})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figures/loss_plot_fold{fold}.png")
    plt.close()

def save_test_predictions(model, test_loader, fold, num_images=3):
    model.eval()
    images_saved = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            output = model(x)
            x = x.cpu().numpy()
            y = y.cpu().numpy()
            output = output.cpu().numpy()

            for i in range(num_images):
                fig, axs = plt.subplots(1, 3, figsize=(9, 3))
                axs[0].imshow(x[i, 0], cmap='gray')
                axs[0].set_title("Noisy")
                axs[0].axis('off')

                axs[1].imshow(y[i, 0], cmap='gray')
                axs[1].set_title("Clean")
                axs[1].axis('off')

                axs[2].imshow(output[i, 0], cmap='gray')
                axs[2].set_title("Predicted")
                axs[2].axis('off')

                plt.tight_layout()
                plt.savefig(f"figures/test_prediction_fold{fold}_img{i}.png")
                plt.close()
                images_saved += 1

            if images_saved >= num_images:
                break


# MAIN
if __name__ == "__main__":
    noisy = np.load("noisy_train_19k.npy").astype(np.float32)
    clean = np.load("clean_train_19k.npy").astype(np.float32)
    noisy_norm = noisy/255.0
    clean_norm = clean/255.0
    noisy_norm = noisy_norm[:1000]
    clean_norm = clean_norm[:1000]

    x_tensor = torch.tensor(noisy_norm[:, np.newaxis, :, :])
    y_tensor = torch.tensor(clean_norm[:, np.newaxis, :, :])
    dataset = TensorDataset(x_tensor, y_tensor)

    # Fixed test set (10%)
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    test_size = int(0.10 * num_samples)
    test_indices = indices[:test_size]
    remaining_indices = indices[test_size:]

    test_set = Subset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=50, shuffle=False)

    # K-Fold Cross-Validation (5 Folds) on remaining data
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(remaining_indices)):
        print(f"\nFold {fold+1}")
        train_subset = Subset(dataset, [remaining_indices[i] for i in train_ids])
        val_subset = Subset(dataset, [remaining_indices[i] for i in val_ids])

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        model = DenoisingUNetLike().to(device)
        train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3)

        if fold == 0:
             train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3)

        plot_loss(train_losses, val_losses, fold)
        evaluate_test_set(model, test_loader)
        save_test_predictions(model, test_loader, fold, num_images=3)
        torch.save(model.cpu().state_dict(), f"cnn_denoiser_weights_1k_fold{fold}.pth")
