import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec
from math import ceil
import random

# FLAGS
EULER = False
RETRAIN_FINAL_MODEL = False  
AUGMENT_DATA = False

# Options
output_name = "resnet_harder_aug_final_model"
batch_size = 16  # default batch size

device = torch. device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def rotate_images(x, y, k):
    return torch.rot90(x, k, dims=[2, 3]), torch.rot90(y, k, dims=[2, 3])

def extract_tensors(subset):
            x_list, y_list = [], []
            for i in range(len(subset)):
                x, y = subset[i]
                x_list.append(x.unsqueeze(0))
                y_list.append(y.unsqueeze(0))
            return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)

def train_model(model, train_loader, val_loader, epochs, lr, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

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

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        # Early stopping check
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[!] No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses

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

def show_comparisons_from_all_splits(model, train_dataset, val_dataset, test_dataset, filename=None):
    if filename is None:
        filename = f"split_comparisons_{output_name}.png"
    model.eval()

    def get_samples(dataset, n=2):
        indices = random.sample(range(len(dataset)), n)
        samples = []
        for idx in indices:
            noisy = dataset[idx][0][0].numpy()
            clean = dataset[idx][1][0].numpy()
            with torch.no_grad():
                input_tensor = dataset[idx][0].unsqueeze(0).to(device)
                output_tensor = model(input_tensor)
                predicted = output_tensor.squeeze().cpu().numpy()
            samples.append((noisy, clean, predicted))
        return samples

    n = 2  # number per split
    splits = [
        ("Train", get_samples(train_dataset, n)),
        ("Val", get_samples(val_dataset, n)),
        ("Test", get_samples(test_dataset, n)),
    ]

    fig, axs = plt.subplots(len(splits) * n, 3, figsize=(12, 3 * len(splits) * n))
    if axs.ndim == 1:
        axs = np.expand_dims(axs, axis=0)

    for i, (split_name, samples) in enumerate(splits):
        for j, (noisy, clean, predicted) in enumerate(samples):
            row = i * n + j
            axs[row, 0].imshow(denormalize_image(noisy), cmap='gray')
            axs[row, 0].set_title(f"{split_name} Noisy")
            axs[row, 1].imshow(denormalize_image(clean), cmap='gray')
            axs[row, 1].set_title("Clean")
            axs[row, 2].imshow(denormalize_image(predicted), cmap='gray')
            axs[row, 2].set_title("Predicted")

            for ax in axs[row]:
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_background_grid(model, dataset, num_pairs=48, output_filename="background_grid.png"):
    dpi = 300
    width_in, height_in = 8.27, 11.69  # A4 in inches
    cols = 4
    rows = ceil(num_pairs / cols)

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    gs = gridspec.GridSpec(rows, cols, wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    selected_indices = random.sample(range(len(dataset)), num_pairs)

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            ax = fig.add_subplot(gs[i])
            input_tensor = dataset[idx][0].unsqueeze(0).to(device)
            prediction = model(input_tensor).squeeze().cpu().numpy()
            noisy = dataset[idx][0][0].numpy()

            noisy_img = denormalize_image(noisy)
            pred_img = denormalize_image(prediction)

            # Combine side-by-side
            combined = np.concatenate([noisy_img, pred_img], axis=1)
            ax.imshow(combined, cmap="gray", aspect="auto")
            ax.axis("off")

    fig.savefig(output_filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# MAIN
if __name__ == "__main__":
    noisy = np.load("noisy_train_19k_harder.npy").astype(np.float32)
    clean = np.load("clean_train_19k_harder.npy").astype(np.float32)

    noisy_norm = normalize_images(noisy)
    clean_norm = normalize_images(clean)

    x_tensor = torch.tensor(noisy_norm[:, np.newaxis, :, :])  # (N, 1, 64, 64)
    y_tensor = torch.tensor(clean_norm[:, np.newaxis, :, :])
    dataset = TensorDataset(x_tensor, y_tensor)

    total_size = len(dataset)
    test_size = int(0.10 * total_size)
    trainval_size = total_size - test_size

    generator = torch.Generator().manual_seed(42)
    trainval_dataset, test_dataset = random_split(dataset, [trainval_size, test_size], generator=generator)

    if AUGMENT_DATA:
        x_tv, y_tv = extract_tensors(trainval_dataset)

        x_aug = [x_tv]
        y_aug = [y_tv]

        for k in [1, 2, 3]:
            x_rot, y_rot = rotate_images(x_tv, y_tv, k)
            x_aug.append(x_rot)
            y_aug.append(y_rot)

        x_all = torch.cat(x_aug, dim=0)
        y_all = torch.cat(y_aug, dim=0)

        print(f"Augmented train+val dataset size: {x_all.shape[0]} samples")
        trainval_dataset = TensorDataset(x_all, y_all)

    # Split train+val into train and val
    val_size = int(0.05 * len(trainval_dataset))
    train_size = len(trainval_dataset) - val_size
    train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = DenoisingUNetLike().to(device)
    

    if EULER:
        train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3)
        torch.save(model.cpu().state_dict(), f"weights_{output_name}.pth")
        plot_loss(train_losses, val_losses, filename=f"train_val_loss_{output_name}.png")

        model.to(device)
        rmse = compute_test_rmse(model, test_loader)
        print(f"Test RMSE: {rmse:.4f}")
        show_multiple_test_samples(model, test_dataset, num_samples=5)

    elif RETRAIN_FINAL_MODEL:
        print("\nRetraining final model on full train+val with best hyperparameters...")

        # Best params from previous tuning
        best_lr = 0.0001
        best_bs = 16
        best_patience = 5
        best_epochs = 18

        trainval_loader = DataLoader(trainval_dataset, batch_size=best_bs, shuffle=True)

        final_model = DenoisingUNetLike().to(device)
        train_model(final_model, trainval_loader, val_loader=trainval_loader,
                    epochs=best_epochs, lr=best_lr, patience=best_patience)

        torch.save(final_model.cpu().state_dict(), f"weights_{output_name}_final_model.pth")
        final_model.to(device)

        final_rmse = compute_test_rmse(final_model, test_loader)
        print(f"\nFinal Test RMSE (on held-out test set): {final_rmse:.4f}")

        show_multiple_test_samples(final_model, test_dataset, num_samples=5, filename=f"test_examples_{output_name}_final_model.png")

    else:
        model.load_state_dict(torch.load(f"weights_{output_name}.pth", map_location=device))
        model.eval()
        print(f"Test RMSE: {compute_test_rmse(model, test_loader):.4f}")
        show_multiple_test_samples(model, test_dataset, num_samples=5)
        show_comparisons_from_all_splits(model, train_dataset, val_dataset, test_dataset)
        # create_background_grid(model, test_dataset, num_pairs=48, output_filename=f"background_{output_name}.png")