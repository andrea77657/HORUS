import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import random
from itertools import product

EULER = True

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

def train_model(model, train_loader, val_loader=None, epochs=50, lr=1e-3, patience=5):
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

        if val_loader is not None:
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
                print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}", flush=True)

            # Early stopping check
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"[!] No improvement for {epochs_no_improve} epoch(s).", flush=True)

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} due to overfitting.", flush=True)
                break
        else:
            # No validation set: just print training loss
            if (epoch + 1) % 5 == 0:
                print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f}", flush=True)

    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)

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


def show_multiple_test_samples(model, test_dataset, num_samples=5, filename="test_examples_crossvali_search.png"):
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
    # Load and normalize data
    noisy = np.load("noisy_train_19k_harder.npy").astype(np.float32)
    clean = np.load("clean_train_19k_harder.npy").astype(np.float32)
    noisy_norm = normalize_images(noisy)
    clean_norm = normalize_images(clean)

    x_tensor = torch.tensor(noisy_norm[:, np.newaxis, :, :])  # (N, 1, 64, 64)
    y_tensor = torch.tensor(clean_norm[:, np.newaxis, :, :])
    full_dataset = TensorDataset(x_tensor, y_tensor)

    # Split off a fixed test set (10%) for final evaluation
    total_size = len(full_dataset)
    test_size = int(0.10 * total_size)
    trainval_size = total_size - test_size

    generator = torch.Generator().manual_seed(42)
    trainval_dataset, test_dataset = random_split(full_dataset, [trainval_size, test_size], generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=64)

    if EULER:
        from itertools import product
        k_folds = 5
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        trainval_indices = list(range(len(trainval_dataset)))

        # Hyperparameter grid
        learning_rates = [1e-2, 1e-3, 5e-4, 1e-4]
        batch_sizes = [32, 64]
        patiences = [5]

        best_rmse = float('inf')
        best_params = None
        results = []

        for lr, bs, patience in product(learning_rates, batch_sizes, patiences):
            print(f"\nTrying config: LR={lr}, Batch={bs}, Patience={patience}")
            fold_rmses = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(trainval_indices)):
                print(f"\n--- Fold {fold + 1}/{k_folds} ---")
                train_subset = torch.utils.data.Subset(trainval_dataset, [trainval_indices[i] for i in train_idx])
                val_subset = torch.utils.data.Subset(trainval_dataset, [trainval_indices[i] for i in val_idx])
                train_loader = DataLoader(train_subset, batch_size=bs, shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=bs, shuffle=False)

                model = DenoisingUNetLike().to(device)
                train_model(model, train_loader, val_loader, epochs=50, lr=lr, patience=patience)

                rmse = compute_test_rmse(model, test_loader)
                fold_rmses.append(rmse)
                print(f"Fold {fold + 1} Test RMSE: {rmse:.4f}")

            avg_rmse = np.mean(fold_rmses)
            results.append(((lr, bs, patience), avg_rmse))
            print(f"Config RMSE average: {avg_rmse:.4f}")

            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params = (lr, bs, patience)

        # Summary
        print("\nHyperparameter tuning results:")
        for config, rmse in results:
            print(f"LR={config[0]}, BS={config[1]}, Pat={config[2]} => RMSE: {rmse:.4f}")
        print(f"\nBest config: LR={best_params[0]}, BS={best_params[1]}, Patience={best_params[2]} with RMSE={best_rmse:.4f}")

        # Final training on full train+val with best config
        print("\nRetraining final model on full train+val with best hyperparameters...")
        best_lr, best_bs, best_patience = best_params
        trainval_loader = DataLoader(trainval_dataset, batch_size=best_bs, shuffle=True)

        final_model = DenoisingUNetLike().to(device)
        train_model(final_model, trainval_loader, val_loader=None, epochs=50, lr=best_lr, patience=best_patience)

        torch.save(final_model.cpu().state_dict(), "crossvali_search_final.pth")
        final_model.to(device)

        final_rmse = compute_test_rmse(final_model, test_loader)
        print(f"\nFinal Test RMSE: {final_rmse:.4f}")
        show_multiple_test_samples(final_model, test_dataset, num_samples=5)

    else:
        # Load and evaluate previously saved model
        model = DenoisingUNetLike().to(device)
        model.load_state_dict(torch.load("crossvali_search_final.pth", map_location=device))
        model.eval()
        rmse = compute_test_rmse(model, test_loader)
        print(f"Test RMSE: {rmse:.4f}")
        show_multiple_test_samples(model, test_dataset, num_samples=5)