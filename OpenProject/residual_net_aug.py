import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

EULER = True

device = torch. device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DenoisingDataset(Dataset):
    def __init__(self, noisy_imgs, clean_imgs, augment=False):
        self.noisy_imgs = noisy_imgs
        self.clean_imgs = clean_imgs
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.noisy_imgs)

    def __getitem__(self, idx):
        noisy = self.noisy_imgs[idx]
        clean = self.clean_imgs[idx]

        if self.augment:
            seed = np.random.randint(2147483647)  # consistent transforms
            torch.manual_seed(seed)
            noisy = self.transform(noisy.squeeze(0)).unsqueeze(0)

            torch.manual_seed(seed)
            clean = self.transform(clean.squeeze(0)).unsqueeze(0)
        else:
            noisy = torch.tensor(noisy)
            clean = torch.tensor(clean)

        return noisy, clean

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


# define CNN model
class DenoisingUNetLike(nn.Module):
    def __init__(self):
        super(DenoisingUNetLike, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128)
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256)
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64)
        )

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)      # (N, 64, H, W)
        x2 = self.pool1(x1)    # (N, 64, H/2, W/2)

        x3 = self.enc2(x2)     # (N, 128, H/2, W/2)
        x4 = self.pool2(x3)    # (N, 128, H/4, W/4)

        # Bottleneck
        x5 = self.bottleneck(x4)  # (N, 256, H/4, W/4)

        # Decoder
        x6 = self.up2(x5)         # (N, 128, H/2, W/2)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.dec2(x6)

        x7 = self.up1(x6)         # (N, 64, H, W)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.dec1(x7)

        return self.output_layer(x7)




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

def plot_loss(train_losses, val_losses, filename="loss_curve.png"):
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


def show_multiple_test_samples(model, test_dataset, num_samples=5, filename="test_examples_aug.png"):
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
    noisy = np.load("noisy_train_19k.npy").astype(np.float32)
    clean = np.load("clean_train_19k.npy").astype(np.float32)

    noisy_norm = normalize_images(noisy)
    clean_norm = normalize_images(clean)

    # Add channel dimension (N, 1, H, W)
    noisy_tensor = torch.tensor(noisy_norm[:, np.newaxis, :, :])
    clean_tensor = torch.tensor(clean_norm[:, np.newaxis, :, :])

    # Custom dataset with optional augmentation
    full_dataset = DenoisingDataset(noisy_tensor, clean_tensor, augment=True)

    # Split into train/val/test
    total_size = len(full_dataset)
    test_size = int(0.10 * total_size)
    trainval_size = total_size - test_size

    generator = torch.Generator().manual_seed(42)
    trainval_dataset, test_dataset = random_split(full_dataset, [trainval_size, test_size], generator=generator)

    val_size = int(0.05 * trainval_size)
    train_size = trainval_size - val_size
    train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


    model = DenoisingUNetLike().to(device)

    if EULER:
        train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3)
        torch.save(model.cpu().state_dict(), "residual_aug_weights_19k_v0.pth")
        plot_loss(train_losses, val_losses, filename="train_val_loss_aug.png")
        # Prepare test loader
        test_loader = DataLoader(test_dataset, batch_size=64)

        # Compute RMSE
        model.to(device)
        rmse = compute_test_rmse(model, test_loader)
        print(f"Test RMSE: {rmse:.4f}")

    else:
        model.load_state_dict(torch.load("residual_aug_weights_19k_v0.pth", map_location=device))
        model.eval()  # set to evaluation mode
        test_loader = DataLoader(test_dataset, batch_size=64)
        print(f"Test RMSE: {compute_test_rmse(model, test_loader):.4f}")


    show_multiple_test_samples(model, test_dataset, num_samples=5)
