import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

EULER = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# define CNN model
class DenoisingUNetLike(nn.Module):
    def __init__(self):
        super(DenoisingUNetLike, self).__init__()
        # Encoder
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

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU()
        )

        # Decoder
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
    return images / 255.0

def denormalize_image(img):
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def train_model_kfold(dataset, test_dataset, k=5, epochs=50, lr=1e-3):
    fold_size = len(dataset) // k
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator)

    test_loader = DataLoader(test_dataset, batch_size=64)

    for fold in range(k):
        print(f"\n--- Fold {fold + 1}/{k} ---")
        val_indices = indices[fold * fold_size: (fold + 1) * fold_size]
        train_indices = torch.cat([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64)

        model = DenoisingUNetLike().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            running_train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
            train_losses.append(running_train_loss / len(train_loader))

            # Validation loss berechnen
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    val_loss = criterion(output, y)
                    running_val_loss += val_loss.item()
            val_losses.append(running_val_loss / len(val_loader))

        val_rmse = compute_test_rmse(model, val_loader)
        test_rmse = compute_test_rmse(model, test_loader)
        print(f"Validation RMSE: {val_rmse:.4f} | Hidden Test RMSE: {test_rmse:.4f}")

        
        plot_loss(train_losses, val_losses, filename=f"loss_plot_fold{fold + 1}.png")

def compute_test_rmse(model, loader):
    model.eval()
    criterion = nn.MSELoss()
    total_mse = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_mse += criterion(output, y).item()
    avg_mse = total_mse / len(loader)
    rmse = np.sqrt(avg_mse)
    return rmse

def plot_loss(train_losses, val_losses, filename):

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")



def show_multiple_test_samples(model, test_dataset, num_samples=5, filename="test_examples.png"):
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


if __name__ == "__main__":
    noisy = np.load("noisy_train_19k.npy").astype(np.float32)
    clean = np.load("clean_train_19k.npy").astype(np.float32)

    noisy_norm = normalize_images(noisy)
    clean_norm = normalize_images(clean)

    x_tensor = torch.tensor(noisy_norm[:, np.newaxis, :, :])
    y_tensor = torch.tensor(clean_norm[:, np.newaxis, :, :])
    dataset = TensorDataset(x_tensor, y_tensor)

    total_size = len(dataset)
    test_size = int(0.10 * total_size)
    trainval_size = total_size - test_size

    generator = torch.Generator().manual_seed(10)
    trainval_dataset, test_dataset = random_split(dataset, [trainval_size, test_size], generator=generator)

    if EULER:
        train_model_kfold(trainval_dataset, test_dataset, k=5, epochs=50, lr=1e-3)
        model = DenoisingUNetLike().to(device)
        model.eval()
        show_multiple_test_samples(model, test_dataset, num_samples=5)
        
    else:
        model = DenoisingUNetLike().to(device)
        model.load_state_dict(torch.load("cnn_denoiser_weights_19k_v2.pth", map_location=device))
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=64)
        print(f"Test RMSE: {compute_test_rmse(model, test_loader):.4f}")
        show_multiple_test_samples(model, test_dataset, num_samples=5)
