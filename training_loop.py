import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from DM import SimpleDM
import os

def save_model(model, path, input_channels, hidden_channels, num_layers):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_channels': input_channels,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
    }, path)
    print(f"Model saved to {path}")


def validate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, _ in dataloader:
            timesteps = torch.randint(1, args.timesteps, (images.size(0),), device=device, dtype=torch.float32)
            images = images.to(device)
            outputs = model(images, timesteps)
            loss = loss_fn(outputs, images)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_diffusion_model(input_channels=1, hidden_channels=64, num_layers=3, epochs=10, batch_size=64, lr=0.001, save_path="./DM_model.pth", validation_ratio=0.2, patience=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    val_size = int(validation_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleDM(input_channels, hidden_channels, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    loss_fn = nn.MSELoss()

    # Early stopping stuff
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for images, _ in train_dataloader:
            timesteps = torch.randint(1, args.timesteps, (images.size(0),), device=device, dtype=torch.float32)
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images, timesteps)

            # Diffusion loss (placeholder; replace with actual diffusion objectives)
            loss = loss_fn(outputs, images)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()

        train_loss = total_train_loss / len(train_dataloader)
        val_loss = validate_model(model, val_dataloader, loss_fn, device)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Save the model after training
    save_model(model, args.save_path, args.input_channels, args.hidden_channels, args.num_layers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a diffusion model on FashionMNIST")
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels in the model')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', type=str, default="./DM_model.pth", help='Path to save the trained model')
    parser.add_argument('--validation_ratio', type=float, default=0.2, help='Ratio of dataset to use for validation')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of images to sample after training')
    parser.add_argument('--timesteps', type=int, default=50, help='Number of diffusion steps for sampling')

    args = parser.parse_args()

    train_diffusion_model(
        input_channels=args.input_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        validation_ratio=args.validation_ratio,
        patience=args.patience
    )