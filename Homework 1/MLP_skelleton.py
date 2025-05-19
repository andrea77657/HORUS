import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, output_size=64, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Normalization function
def normalize_data(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std, mean, std

# Mean Squared Error function (just for completeness)
def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Training function
def train_model(train_loader, val_loader, input_size=64, hidden_size=128, lr=1e-3, epochs=50):
    model = MLP(input_size=input_size, hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                val_preds = model(x_val)
                val_loss += criterion(val_preds, y_val).item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return model, train_losses, val_losses

# Plotting noisy, clean and predicted profiles
def plot_noisy_clean_predicted(model, x_tensor, y_tensor, mean, std, title="Sample"):
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor.unsqueeze(0)).squeeze(0)

    # De-normalize
    noisy = x_tensor * std + mean
    clean = y_tensor * std + mean
    predicted = pred * std + mean

    plt.figure(figsize=(10, 4))
    plt.plot(noisy.numpy(), label="Noisy")
    plt.plot(clean.numpy(), label="Clean")
    plt.plot(predicted.numpy(), label="Predicted")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Display noisy, clean, and predicted image side-by-side
def display_image_triplet(noisy_img, clean_img, predicted_profile, mean, std):
    # De-normalize profile
    predicted_profile = predicted_profile.detach().cpu().numpy() * std + mean
    predicted_img = np.tile(predicted_profile[:, None], (1, 64))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(noisy_img, cmap='gray')
    axs[0].set_title("Noisy Image")
    axs[1].imshow(clean_img, cmap='gray')
    axs[1].set_title("Clean Image")
    axs[2].imshow(predicted_img, cmap='gray')
    axs[2].set_title("Predicted Image")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load and preprocess data
    noisy_data = np.load("noisy_images_small_1k.npy").astype(np.float32)
    clean_data = np.load("clean_images_small_1k.npy").astype(np.float32)

    noisy_profiles = noisy_data.mean(axis=2)
    clean_profiles = clean_data.mean(axis=2)

    # Normalize the profiles
    noisy_profiles, mean, std = normalize_data(noisy_profiles)
    clean_profiles = (clean_profiles - mean) / std

    # Convert to tensors
    inputs = torch.tensor(noisy_profiles)
    targets = torch.tensor(clean_profiles)
    dataset = TensorDataset(inputs, targets)

    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Train the model
    model, train_losses, val_losses = train_model(train_loader, val_loader)

    # Plot training/validation loss
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot example profiles
    train_sample = train_dataset[0]
    val_sample = val_dataset[0]

    plot_noisy_clean_predicted(model, train_sample[0], train_sample[1], mean, std, title="Training Sample")
    plot_noisy_clean_predicted(model, val_sample[0], val_sample[1], mean, std, title="Validation Sample")

    # Show side-by-side image visualization for the same sample
    model.eval()
    with torch.no_grad():
        input_tensor = inputs[0].unsqueeze(0)  # use tensor, not np array!
        predicted_profile = model(input_tensor).squeeze(0)

    display_image_triplet(
        noisy_img=noisy_data[0],
        clean_img=clean_data[0],
        predicted_profile=predicted_profile,
        mean=mean,
        std=std
    )
