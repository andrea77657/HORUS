import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import random
from torchvision.models import vgg16

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Perceptual Loss with VGG16
vgg = vgg16(pretrained=True).features[:8].to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

def perceptual_loss(pred, target):
    pred_features = vgg(pred.repeat(1, 3, 1, 1))
    target_features = vgg(target.repeat(1, 3, 1, 1))
    return F.mse_loss(pred_features, target_features)

# Weight initialization
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class DenoisingUNetLike(nn.Module):
    def __init__(self):
        super(DenoisingUNetLike, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), Swish(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), Swish(),
            nn.Dropout(0.1)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), Swish(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), Swish(),
            nn.Dropout(0.2)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), Swish(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), Swish(),
            nn.Dropout(0.3)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), Swish(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), Swish(),
            nn.Dropout(0.2)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), Swish(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), Swish(),
            nn.Dropout(0.1)
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

def denormalize_image(image):
    return (image * 255.0).clip(0, 255).astype(np.uint8)

def train_model(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = 0.3 * mse_loss(output, y) + 0.7 * perceptual_loss(output, y)
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
                loss = 0.3 * mse_loss(output, y) + 0.7 * perceptual_loss(output, y)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        scheduler.step()

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
    plt.savefig(f"figures/easyswish_loss_plot_fold{fold}.png")
    plt.close()

def show_multiple_test_samples(model, test_dataset, num_samples=5, filename="swish_AndreaBilderrunsheasy.png"):
    model.eval()
    model.to(device)
    indices = random.sample(range(len(test_dataset)), num_samples)

    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axs = np.expand_dims(axs, axis=0)

    for row, idx in enumerate(indices):
        noisy = test_dataset[idx][0][0].numpy()
        clean = test_dataset[idx][1][0].numpy()
        with torch.no_grad():
            input_tensor = test_dataset[idx][0].unsqueeze(0).to(device)
            output_tensor = model(input_tensor)
            output = output_tensor.squeeze().cpu().numpy()

        noisy = denormalize_image(noisy)
        clean = denormalize_image(clean)
        output = denormalize_image(output)

        axs[row, 0].imshow(noisy, cmap='gray')
        axs[row, 0].set_title(f"Sample {idx} - Noisy")
        axs[row, 1].imshow(clean, cmap='gray')
        axs[row, 1].set_title("Clean")
        axs[row, 2].imshow(output, cmap='gray')
        axs[row, 2].set_title("Predicted")

        for ax in axs[row]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# MAIN
if __name__ == "__main__":
    noisy = np.load("noisy_train_19k.npy").astype(np.float32) / 255.0
    clean = np.load("clean_train_19k.npy").astype(np.float32) / 255.0

    noisy_images = noisy
    clean_images = clean

    x_tensor = torch.tensor(noisy[:, np.newaxis, :, :])
    y_tensor = torch.tensor(clean[:, np.newaxis, :, :])
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
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(remaining_indices)):
        print(f"\nFold {fold+1}")
        train_subset = Subset(dataset, [remaining_indices[i] for i in train_ids])
        val_subset = Subset(dataset, [remaining_indices[i] for i in val_ids])

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        model = DenoisingUNetLike().to(device)
        initialize_weights(model)

        train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3)
        plot_loss(train_losses, val_losses, fold)
        evaluate_test_set(model, test_loader)
        show_multiple_test_samples(model, test_set, num_samples=5)
