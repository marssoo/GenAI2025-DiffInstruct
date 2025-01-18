import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from GAN import Generator, Discriminator, GANLoss
import os

def train_gan(args):
    # Set up the dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the generator and discriminator
    generator = Generator(args.latent_dim, args.img_channels, args.hidden_dim).to(device)
    discriminator = Discriminator(args.img_channels, args.hidden_dim).to(device)
    loss_fn = GANLoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_generator, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_discriminator, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        total_d_loss, total_g_loss = 0.0, 0.0

        for imgs, _ in dataloader:
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)

            real_labels = torch.ones((batch_size, 1), device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            # Train Discriminator
            z = torch.randn((batch_size, args.latent_dim), device=device)
            fake_imgs = generator(z)

            optimizer_D.zero_grad()
            d_loss = loss_fn.discriminator_loss(discriminator, real_imgs, fake_imgs, real_labels, fake_labels)
            d_loss.backward()
            optimizer_D.step()

            # Train Generator multiple times for each discriminator update
            for _ in range(args.generator_update_ratio):
                z = torch.randn((batch_size, args.latent_dim), device=device)
                fake_imgs = generator(z)

                optimizer_G.zero_grad()
                g_loss = loss_fn.generator_loss(discriminator, fake_imgs, real_labels)
                g_loss.backward()
                optimizer_G.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        train_d_loss = total_d_loss / len(dataloader)
        train_g_loss = total_g_loss / len(dataloader)

        print(f"Epoch [{epoch + 1}/{args.epochs}]	Train D Loss: {train_d_loss:.6f} G Loss: {train_g_loss:.6f}")

        # Save logs
        with open('GAN_models/GAN_logs.csv', 'a') as f:
            if epoch == 0:
                f.write('epoch,train_d_loss,train_g_loss\n')
            f.write(f'{epoch},{train_d_loss},{train_g_loss}\n')

        # Save models
    torch.save(generator.state_dict(), os.path.join(args.model_dir, "best_generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(args.model_dir, "best_discriminator.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN on FashionMNIST")
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimensionality of the latent space')
    parser.add_argument('--img_channels', type=int, default=1, help='Number of image channels')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for both generator and discriminator')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--generator_update_ratio', type=int, default=1, help='Number of generator updates per discriminator update')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr_generator', type=float, default=0.0001, help='Learning rate for the generator')
    parser.add_argument('--lr_discriminator', type=float, default=0.0001, help='Learning rate for the discriminator')
    parser.add_argument('--model_dir', type=str, default="GAN_models", help='Directory to save model checkpoints')

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    train_gan(args)
