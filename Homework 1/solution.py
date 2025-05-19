import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__() #Inherit from torch.nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x


class MLP_dynamic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        A flexible MLP that supports an arbitrary number of hidden layers.

        Args:
            input_size (int): Dimensionality of the input features.
            hidden_size (int or list of int): If an integer, build a single hidden layer.
                                              If a list, build len(hidden_size) hidden layers,
                                              each with the specified number of neurons.
            output_size (int): Dimensionality of the output layer.
        """
        super().__init__() #Inherit from torch.nn.Module

        # If hidden_size is a single integer, convert it to a single-element list
        # so we can uniformly handle the "multiple layer" scenario.
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        layers = []
        in_features = input_size

        # Create each hidden layer
        for h in hidden_size:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h

        # Finally, create the output layer
        layers.append(nn.Linear(in_features, output_size))

        # Wrap them in nn.Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def normalize_data(data, axis=0):
    """
    Normalize the data to range [0, 1] using Min-Max Scaling.
    """
    min_val = np.min(data, axis=axis, keepdims=True)
    max_val = np.max(data, axis=axis, keepdims=True)
    return (data - min_val) / (max_val - min_val), min_val, max_val

def mean_squared_error(ground_truth,prediction):
    return ((ground_truth - prediction) ** 2).mean()


def train_model(model,train_dataset, val_dataset, num_epochs,lr,weight_decay,optimizer):
    """
    Train an MLP given train_dataset and val_dataset.
    Returns:
      - model
      - train_losses, val_losses
    """

    # --------------------
    # 1) Create DataLoaders
    #    (again, not strictly necessary but it's the de facto standard)
    # --------------------
    batch_size = 1000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --------------------
    # 2) Define  optimizer
    # --------------------

    if optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --------------------
    # 3) Training loop
    # --------------------
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        # (A) Training step
        for noisy_batch, clean_batch in train_loader:
            outputs = model(noisy_batch)
            loss = mean_squared_error(outputs, clean_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # Compute average training loss
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # --------------------
        # 5) Validation step
        # --------------------
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for noisy_batch, clean_batch in val_loader:
                outputs = model(noisy_batch)
                loss = mean_squared_error(outputs, clean_batch)
                epoch_val_loss += loss.item()

        # Compute average validation loss
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        # Print progress (optional)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"[Epoch {epoch + 1}/{num_epochs}] "
                  f"Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

    # Return the trained model and loss curves
    return model, train_losses, val_losses


def plot_image_and_profiles(noisy_data, clean_data, index=0, rowA=10, rowB=50):
    """
    Plot the noisy and clean images at a given index, along with their
    row-wise brightness profiles used in training.

    Additionally, highlight two rows (rowA, rowB) in both images and
    show the corresponding points in the brightness plots to help
    students see how the brightness profile is derived.

    Args:
        noisy_data (np.ndarray): Array of noisy images, shape (N, H, W).
        clean_data (np.ndarray): Array of clean images, shape (N, H, W).
        index (int): Which image index to plot.
        rowA (int): First row index to highlight.
        rowB (int): Second row index to highlight.
    """
    # 1) Extract a single noisy & clean image
    noisy_image = noisy_data[index]
    clean_image = clean_data[index]
    height, width = noisy_image.shape

    # 2) Compute row-wise brightness (mean across width dimension)
    noisy_profile = noisy_image.mean(axis=1)
    clean_profile = clean_image.mean(axis=1)

    # 3) Create subplots for images (top row) and profiles (bottom row)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # --- (A) TOP LEFT: Noisy Image ---
    axs[0, 0].imshow(noisy_image, cmap='gray', aspect='equal')
    axs[0, 0].set_title(f"Noisy Image (index={index})")
    axs[0, 0].axis('off')
    # Highlight rowA (red) and rowB (blue) in the noisy image
    axs[0, 0].axhline(y=rowA, color='red', lw=2, alpha=0.8)
    axs[0, 0].axhline(y=rowB, color='blue', lw=2, alpha=0.8)

    # --- (B) TOP RIGHT: Clean Image ---
    axs[0, 1].imshow(clean_image, cmap='gray', aspect='equal')
    axs[0, 1].set_title(f"Clean Image (index={index})")
    axs[0, 1].axis('off')
    # Highlight rowA (red) and rowB (blue) in the clean image
    axs[0, 1].axhline(y=rowA, color='red', lw=2, alpha=0.8)
    axs[0, 1].axhline(y=rowB, color='blue', lw=2, alpha=0.8)

    # --- (C) BOTTOM LEFT: Noisy Brightness Profile ---
    axs[1, 0].plot(noisy_profile, color='black')
    axs[1, 0].set_title("Noisy Row-wise Brightness")
    # Mark rowA and rowB on the profile
    axs[1, 0].plot(rowA, noisy_profile[rowA], 'ro', markersize=8)
    axs[1, 0].plot(rowB, noisy_profile[rowB], 'bo', markersize=8)

    # --- (D) BOTTOM RIGHT: Clean Brightness Profile ---
    axs[1, 1].plot(clean_profile, color='black')
    axs[1, 1].set_title("Clean Row-wise Brightness")
    # Mark rowA and rowB on the profile
    axs[1, 1].plot(rowA, clean_profile[rowA], 'ro', markersize=8)
    axs[1, 1].plot(rowB, clean_profile[rowB], 'bo', markersize=8)

    plt.tight_layout()
    plt.show()

def plot_noisy_clean_predicted(noisy, clean, predicted, index=0):
    """
    Plot noisy, clean, and predicted brightness profiles for a specific sample index.

    Args:
        noisy (np.ndarray): Array of noisy profiles.
        clean (np.ndarray): Array of clean profiles.
        predicted (np.ndarray): Array of predicted profiles.
        index (int): The index of the sample to plot.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(noisy[index], label="Noisy", linestyle="--", color="gray")
    plt.plot(clean[index], label="Clean", color="blue")
    plt.plot(predicted[index], label="Predicted", color="red")

    plt.title(f"Noisy vs Clean vs Predicted Profiles (Index={index})")
    plt.xlabel("Row Index")
    plt.ylabel("Brightness")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_loss(train_losses, val_losses):
    """
    Plot training and validation loss against epochs.

    Args:
        train_losses (list): List of training losses over epochs.
        val_losses (list): List of validation losses over epochs.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # (A) Load data (example: shape (N, H, W))
    noisy_data = np.load("noisy_images_small_1k.npy").astype(np.float32)
    clean_data = np.load("clean_images_small_1k.npy").astype(np.float32)

    # Convert to row-wise brightness profiles, shape (N, H)
    noisy_profiles = noisy_data.mean(axis=2)
    clean_profiles = clean_data.mean(axis=2)

    # (B) Normalize
    noisy_norm, x_min, x_max = normalize_data(noisy_profiles, axis=1)
    clean_norm, y_min, y_max = normalize_data(clean_profiles, axis=1)

    # Make a single dataset
    x_tensor = torch.tensor(noisy_norm, dtype=torch.float32)
    y_tensor = torch.tensor(clean_norm, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)

    # (C) 95/5 train validation split t
    val_size = int(0.05 * len(dataset))  # 5% for validation
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # (D) Train the model
    input_size =  noisy_norm.shape[1]  # number of features (H)
    output_size = clean_norm.shape[1]  # same shape as x
    hidden_size =256
    num_epochs =2000
    lr =0.0001
    weight_decay =1e-4
    optimizer = 'adam'
    #Simple if else to decide if we want to train a new model or only run inference on an existing model
    train = False

    print(input_size,hidden_size)
    model = MLP(input_size, hidden_size, output_size)

    if train:
        model, train_losses, val_losses = train_model(model,train_dataset, val_dataset, num_epochs,lr,weight_decay,optimizer)
        #save the weights to rerun plotting later without having to retrain each time
        torch.save(model.state_dict(), 'MLP_solution_weights.pth')

        # (E) Plot loss curves
        plot_loss(train_losses, val_losses)

        # (F) For final demonstration, we can just pick a sample from val_dataset
        model.eval()

        # Take an arbitrary index in val_dataset
        val_idx = 10
        val_noisy, val_clean = val_dataset[val_idx]
        val_noisy = val_noisy.unsqueeze(0)

        with torch.no_grad():
            val_pred = model(val_noisy).cpu().numpy()

        plot_noisy_clean_predicted(
            noisy=np.expand_dims(val_dataset[val_idx][0].numpy(), axis=0),
            clean=np.expand_dims(val_dataset[val_idx][1].numpy(), axis=0),
            predicted=val_pred,
            index=0
        )

    else:
        model.load_state_dict(torch.load('MLP_solution_weights.pth'))
        model.eval()

        #Plotting 10 random idx's from the validation set
        n_plots = 10
        rand_idx = np.random.choice(len(val_dataset),
                                    size=n_plots,
                                    replace=False)

        for idx in rand_idx:
            val_noisy, val_clean = val_dataset[idx]
            val_noisy = val_noisy.unsqueeze(0)

            with torch.no_grad():
                val_pred = model(val_noisy).cpu().numpy()

            plot_noisy_clean_predicted(
                noisy=np.expand_dims(val_noisy.squeeze(0).numpy(), 0),
                clean=np.expand_dims(val_clean.numpy(), 0),
                predicted=val_pred,
                index=0,
            )







