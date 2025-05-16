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
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
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
            x = x.to(device)
            y = y.to(device)
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
                x = x.to(device)
                y = y.to(device)
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
    plt.savefig("loss_plot.png")

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
    plt.savefig("learned_filters.png")

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
    plt.savefig("image_comparison.png")

# MAIN
if __name__ == "__main__":
    # load data
    noisy = np.load("noisy_images_small_1k.npy").astype(np.float32)
    clean = np.load("clean_images_small_1k.npy").astype(np.float32)

    # normalizing
    noisy_norm = normalize_images(noisy)
    clean_norm = normalize_images(clean)

    x_tensor = torch.tensor(noisy_norm[:, np.newaxis, :, :])  # (N, 1, 64, 64)
    y_tensor = torch.tensor(clean_norm[:, np.newaxis, :, :])
    dataset = TensorDataset(x_tensor, y_tensor)

    # Split including a seed, so that the split is reproducible
    val_size = int(0.05 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    train_on_euler = torch.cuda.is_available()  # oder einfach: True/False zum Testen lokal
    manual_train_switch = True  # HIER kannst du Training manuell aktivieren/deaktivieren

if train_on_euler and manual_train_switch:
    print("Training is enabled and will run on GPU.")
    model = CNN_Denoiser().to(device)
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=30, lr=1e-3)
    torch.save(model.state_dict(), "cnn_weights.pth")
else:
    print("Training skipped. Loading pre-trained model weights.")
    model = CNN_Denoiser().to(device)
    model.load_state_dict(torch.load("cnn_weights.pth", map_location=device))

    # Plot Loss
    plot_loss(train_losses, val_losses)

    # show filters
    # visualize_filters(model)
    torch.save(model.state_dict(), "cnn_denoiser_weights.pth")
    # show image comparison
    idx_train = 0
    noisy_img = train_dataset[idx_train][0][0].numpy()
    clean_img = train_dataset[idx_train][1][0].numpy()
    with torch.no_grad():
        output_img = model(train_dataset[idx_train][0].unsqueeze(0).to(device)).squeeze().cpu().numpy()
    show_image_comparison(noisy_img, clean_img, output_img, "Train")

    idx_val = 0
    noisy_img = val_dataset[idx_val][0][0].numpy()
    clean_img = val_dataset[idx_val][1][0].numpy()
    with torch.no_grad():
        output_img = model(val_dataset[idx_val][0].unsqueeze(0).to(device)).squeeze().cpu().numpy()
    show_image_comparison(noisy_img, clean_img, output_img, "Val")

