import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from DM import UNetDM, q_sample
from GAN import Generator

def load_models(dm_path, generator_path, latent_dim, device):
    # Load pre-trained Diffusion Models (s_p(t) and s_phi)
    sp_t = UNetDM(input_channels=1, hidden_channels=128, num_layers=4, timesteps=2000).to(device)
    sp_t.load_state_dict(torch.load(dm_path, map_location=device))
    sp_t.eval()

    s_phi = UNetDM(input_channels=1, hidden_channels=128, num_layers=4, timesteps=2000).to(device)
    s_phi.load_state_dict(torch.load(dm_path, map_location=device))
    s_phi.train()

    # Load pre-trained Generator (g_theta)
    generator = Generator(latent_dim=latent_dim, img_channels=1, hidden_dim=128).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.train()

    return sp_t, s_phi, generator

def loss_phi(s_phi, batch_size, device, x0):
    # Sample x_t from the diffusion process
    t = torch.randint(0, s_phi.timesteps, (batch_size,), device=device).long()
    xt = q_sample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    weights = w(t).to(device).view(-1, 1, 1, 1)
    s_phi_prediction = s_phi(xt, t)

    # Loss and update on the diffusion model
    phi_loss = weights * nn.functional.mse_loss(s_phi_prediction, xt, reduction='none')
    
    return phi_loss

def loss_theta(s_phi, sp_t, batch_size, device, x0):
    t = torch.randint(0, s_phi.timesteps, (batch_size,), device=device).long()
    #print(t.shape)
    #t = torch.ones(batch_size, device=device).long()
    weights = w(t).to(device).view(-1, 1, 1, 1)
    xt = q_sample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

    s_phi.train(), sp_t.train()
    with torch.no_grad():
        rng_state = torch.cuda.get_rng_state()
        s_phi_prediction = s_phi(xt, t)
        torch.cuda.set_rng_state(rng_state)
        sp_t_prediction = sp_t(xt, t)
    sp_t.eval()

    loss = weights * ((s_phi_prediction - sp_t_prediction) * x0)

    return loss

def train_diff_instruct(dm_path, generator_path, betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, w, lr_phi, lr_theta, save_path_phi, save_path_theta, patience, device, epochs, timesteps, latent_dim, batch_size):
    sp_t, s_phi, generator = load_models(dm_path, generator_path, latent_dim, device=device)

    optimizer_phi = optim.Adam(s_phi.parameters(), lr=lr_phi)
    optimizer_theta = optim.Adam(generator.parameters(), lr=lr_theta)
    scheduler_phi = ReduceLROnPlateau(optimizer_phi, mode='min', factor=0.8, patience=4, verbose=True, min_lr=1e-7)
    #scheduler_theta = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=4, verbose=True, min_lr=1e-6)
    best_phi_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        optimizer_phi.zero_grad(set_to_none=True)
        
        #### Update s_phi (DM)
        #generate samples
        with torch.no_grad():
            generator.eval()
            z = torch.randn((batch_size, latent_dim), device=device)
            x0 = generator(z)
            generator.train()

        phi_loss = loss_phi(s_phi, batch_size, device, x0)
        phi_loss = phi_loss.sum() / batch_size

        phi_loss.backward()
        optimizer_phi.step()

        #### Update generator (theta)
        optimizer_theta.zero_grad(set_to_none=True)
        # generate images
        z = torch.randn((batch_size, latent_dim), device=device)
        x0 = generator(z)

        s_phi.eval()
        theta_loss = loss_theta(s_phi, sp_t, batch_size, device, x0)
        s_phi.train()                                                   #there is some redundancy in the original repo, for now I am keeping it

        theta_loss = theta_loss.sum([1, 2, 3])
        #print(theta_loss.shape)

        theta_loss = theta_loss.sum() / batch_size
        theta_loss.backward()
        optimizer_theta.step()

        print(f"Epoch {epoch + 1}/{epochs}\t - Phi Loss: {phi_loss.item():.8f}, Theta Loss: {theta_loss.item():.6f}")
        scheduler_phi.step(phi_loss)
        
        if phi_loss.item() < best_phi_loss:
            best_phi_loss = phi_loss.item()
            patience_counter = 0

            # Save the models
            torch.save(s_phi.state_dict(), save_path_phi)
            torch.save(generator.state_dict(), save_path_theta)
        else:
            patience_counter += 1
        # Early stopping based on phi_loss
        if  patience_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diff-Instruct Algorithm")

    parser.add_argument('--dm_path', type=str, default="DM_models/UNet_4layers_128hc_2000steps.pth", help="Path to the pre-trained DM model")
    parser.add_argument('--generator_path', type=str, default="GAN_models/best_generator.pth", help="Path to the pre-trained GAN generator")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--lr_phi', type=float, default=2e-5, help="Learning rate for updating phi (DM)")
    parser.add_argument('--lr_theta', type=float, default=2e-5, help="Learning rate for updating theta (GAN generator)")
    parser.add_argument('--save_path_phi', type=str, default="DI_models/DI_phi.pth", help="Path to save the updated DM model")
    parser.add_argument('--save_path_theta', type=str, default="DI_models/DI_generator.pth", help="Path to save the updated GAN generator")
    parser.add_argument('--patience', type=int, default=50, help="Patience for early stopping")
    parser.add_argument('--epochs', type=int, default=7000, help="Number of training epochs")
    parser.add_argument('--timesteps', type=int, default=2000, help="Number of diffusion timesteps")
    parser.add_argument('--latent_dim', type=int, default=128, help="Latent dimension of the generator")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Diffusion parameters (precomputed)
    timesteps = args.timesteps
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1.0 - betas
    sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(alphas, axis=0)).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - torch.cumprod(alphas, axis=0)).to(device)

    # Define weighting function w(t)
    def w(t):
        return torch.ones_like(t, dtype=torch.float32)  # Uniform weighting for simplicity

    # Train Diff-Instruct
    train_diff_instruct(
        dm_path=args.dm_path,
        generator_path=args.generator_path,
        betas=betas,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        w=w,
        lr_phi=args.lr_phi,
        lr_theta=args.lr_theta,
        save_path_phi=args.save_path_phi,
        save_path_theta=args.save_path_theta,
        patience=args.patience,
        device=device,
        epochs=args.epochs,
        timesteps=timesteps,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size
    )
