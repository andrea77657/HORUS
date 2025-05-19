import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

device = torch. device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# define CNN model
class CNN_Denoiser(nn.Module):
    def __init__(self):
        super(CNN_Denoiser, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # added
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # match encoder
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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

def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_filters(model):
    first_layer = model.encoder[0]
    filters = first_layer.weight.data.clone()
    filters = filters - filters.min()
    filters = filters / filters.max()
    
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))
    axs = axs.flatten()
    for i in range(32):
        axs[i].imshow(filters[i, 0], cmap='gray')
        axs[i].axis('off')
    plt.suptitle("Learned Filters from First Conv Layer")
    plt.show()

def show_image_comparison(noisy, clean, output, title_prefix=""):
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(noisy, cmap='gray')
    axs[0].set_title(f"{title_prefix} Noisy")
    axs[1].imshow(clean, cmap='gray')
    axs[1].set_title(f"{title_prefix} Clean")
    axs[2].imshow(output, cmap='gray')
    axs[2].set_title(f"{title_prefix} Output")
    for ax in axs:
        ax.axis('off')
    plt.show()

# MAIN
if __name__ == "__main__":
    # Load 19k images instead of 1k
    noisy = np.load("noisy_train_19k.npy").astype(np.float32)
    clean = np.load("clean_train_19k.npy").astype(np.float32)

    noisy_norm = normalize_images(noisy)
    clean_norm = normalize_images(clean)

    x_tensor = torch.tensor(noisy_norm[:, np.newaxis, :, :])  # (N, 1, 64, 64)
    y_tensor = torch.tensor(clean_norm[:, np.newaxis, :, :])
    dataset = TensorDataset(x_tensor, y_tensor)

    val_size = int(0.05 * len(dataset))  # 5% validation
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = CNN_Denoiser().to(device)
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3)

    # Save model weights to CPU
    torch.save(model.cpu().state_dict(), "cnn_denoiser_weights_19k.pth")