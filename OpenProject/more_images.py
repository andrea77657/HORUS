import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import random

# Flags
EULER = True  # Set to False to load pre-trained weights

# Options
output_name = "autoencoder_19k"
batch_size = 64  # default batch size

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
    return images / 255.0

def denormalize_image(img):
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

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

def plot_loss(train_losses, val_losses, filename=None):
    if filename is None:
        filename = f"loss_curve_{output_name}.png"
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def compute_test_rmse(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_mse = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_mse += criterion(output, y).item()
    avg_mse = total_mse / len(test_loader)
    rmse = np.sqrt(avg_mse)
    return rmse

def show_multiple_test_samples(model, test_dataset, num_samples=5, filename=None):
    if filename is None:
        filename = f"test_examples_{output_name}.png"
    model.eval()
    indices = random.sample(range(len(test_dataset)), num_samples)

    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axs = np.expand_dims(axs, axis=0)  # handle edge case

    for row, idx in enumerate(indices):
        noisy = test_dataset[idx][0][0].numpy()
        clean = test_dataset[idx][1][0].numpy()
        with torch.no_grad():
            input_tensor = test_dataset[idx][0].unsqueeze(0).to(device)
            output_tensor = model(input_tensor)
            output = output_tensor.squeeze().cpu().numpy()

        # Denormalize for display
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
    # Load data
    noisy = np.load("noisy_train_19k.npy").astype(np.float32)
    clean = np.load("clean_train_19k.npy").astype(np.float32)

    # Normalize
    noisy_norm = normalize_images(noisy)
    clean_norm = normalize_images(clean)

    x_tensor = torch.tensor(noisy_norm[:, np.newaxis, :, :])
    y_tensor = torch.tensor(clean_norm[:, np.newaxis, :, :])
    dataset = TensorDataset(x_tensor, y_tensor)

    # Split into train/val/test
    total_size = len(dataset)
    test_size = int(0.10 * total_size)
    trainval_size = total_size - test_size

    generator = torch.Generator().manual_seed(42)
    trainval_dataset, test_dataset = random_split(dataset, [trainval_size, test_size], generator=generator)

    val_size = int(0.05 * trainval_size)
    train_size = trainval_size - val_size
    train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate model
    model = CNN_Denoiser().to(device)

    if EULER:
        print("\nTraining model on train/val split...")
        train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3)
        plot_loss(train_losses, val_losses, filename=f"train_val_loss_{output_name}.png")

        torch.save(model.cpu().state_dict(), f"weights_{output_name}.pth")
        model.to(device)

        rmse = compute_test_rmse(model, test_loader)
        print(f"\nTest RMSE: {rmse:.4f}")

        show_multiple_test_samples(model, test_dataset, num_samples=5, filename=f"test_examples_{output_name}.png")

    else:
        print(f"\nLoading pre-trained weights: weights_{output_name}.pth")
        model.load_state_dict(torch.load(f"weights_{output_name}.pth", map_location=device))
        model.eval()

        rmse = compute_test_rmse(model, test_loader)
        print(f"\nTest RMSE: {rmse:.4f}")

        show_multiple_test_samples(model, test_dataset, num_samples=5, filename=f"test_examples_{output_name}.png")
