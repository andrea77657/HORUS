# horus_parallel_train.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os
from mpi4py import MPI

# Parallel setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        if rank == 0 and (epoch + 1) % 5 == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

if __name__ == "__main__":
    if rank == 0:
        print(f"Running on {size} MPI processes")

    # load data (only rank 0 loads and broadcasts)
    if rank == 0:
        noisy = np.load("noisy_images_small_1k.npy").astype(np.float32)
        clean = np.load("clean_images_small_1k.npy").astype(np.float32)
    else:
        noisy = None
        clean = None

    noisy = comm.bcast(noisy, root=0)
    clean = comm.bcast(clean, root=0)

    # normalize and split among processes
    noisy_norm = normalize_images(noisy)
    clean_norm = normalize_images(clean)

    x_tensor = torch.tensor(noisy_norm[:, np.newaxis, :, :])  # (N, 1, 64, 64)
    y_tensor = torch.tensor(clean_norm[:, np.newaxis, :, :])
    dataset = TensorDataset(x_tensor, y_tensor)

    val_size = int(0.05 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_sampler = torch.utils.data.Subset(train_dataset, list(range(rank, len(train_dataset), size)))
    val_sampler = torch.utils.data.Subset(val_dataset, list(range(rank, len(val_dataset), size)))

    train_loader = DataLoader(train_sampler, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_sampler, batch_size=64, shuffle=False)

    # train model
    model = CNN_Denoiser()
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3)

    # only rank 0 saves the model
    if rank == 0:
        torch.save(model.state_dict(), "cnn_denoiser_parallel.pth")
        print("Model saved.")
