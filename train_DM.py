import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from DM import UNetDM
import os
import numpy as np

# Validation function
def validate_model(model, dataloader, loss_fn, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            timesteps = torch.randint(0, model.timesteps, (images.size(0),), device=device)
            loss = model.compute_loss(images, timesteps, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def cosine_beta_schedule(timesteps, s=0.008):
    x = np.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0.0001, a_max=0.02)

# Main training function
def train_diffusion_model(args, path_to_save):
    # Prepare the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    val_size = int(args.validation_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Device and model setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNetDM(args.input_channels, args.hidden_channels, args.num_layers, args.timesteps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler for adaptive learning rate adjustment
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=4, verbose=True, min_lr=1e-6)

    # Precomputed beta values
    betas = torch.tensor(cosine_beta_schedule(args.timesteps)).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        for images, _ in train_dataloader:
            images = images.to(device)
            timesteps = torch.randint(0, args.timesteps, (images.size(0),), device=device)

            loss = model.compute_loss(images, timesteps, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_dataloader)
        val_loss = validate_model(model, val_dataloader, nn.MSELoss(), device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.5f}, Validation Loss: {val_loss:.5f}")
        with open(f'DM_models/UNet_{args.num_layers}layers_{args.hidden_channels}hc_{args.timesteps}steps_logs.csv', 'a') as f:
            if epoch == 0:
                f.write('epoch,train_loss,val_loss\n')
            f.write(f'{epoch},{train_loss},{val_loss}\n')
        # Step the scheduler based on training loss
        scheduler.step(train_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
            torch.save(model.state_dict(), path_to_save)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a diffusion model on FashionMNIST")
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels in the model')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the model')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1500, help='Number of diffusion steps')
    parser.add_argument('--validation_ratio', type=float, default=0.2, help='Ratio of dataset to use for validation')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--save_path', type=str, default="./", help='Path to save the best model')

    args = parser.parse_args()
    path_to_save = args.save_path + f'DM_models/UNet_{args.num_layers}layers_{args.hidden_channels}hc_{args.timesteps}steps.pth'
    train_diffusion_model(args, path_to_save)
