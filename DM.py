import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    t = t.long()  
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=time.device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)

class UNetDM(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, timesteps=50):
        super(UNetDM, self).__init__()
        self.timesteps = timesteps

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(ResidualBlock(input_channels + hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.encoder.append(ResidualBlock(hidden_channels, hidden_channels))

        # Bottleneck
        self.bottleneck = ResidualBlock(hidden_channels, hidden_channels)

        # Decoder
        self.decoder = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.decoder.append(ResidualBlock(hidden_channels * 2, hidden_channels))
        # The last decoder layer needs to handle single hidden_channels from the bottleneck
        self.decoder.append(ResidualBlock(hidden_channels + hidden_channels, hidden_channels))


        self.final_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x, t):
        x = x.float()  # Cast input to float32
        t = t.float()  # Cast timesteps to float32
        t_embed = self.time_embedding(t.view(-1, 1))
        t_embed = t_embed.view(t.size(0), -1, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, t_embed], dim=1)

        # Encoder
        enc_features = []
        for layer in self.encoder:
            x = layer(x)
            enc_features.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for layer, enc_feature in zip(self.decoder, reversed(enc_features)):
            x = torch.cat([x, enc_feature], dim=1)
            x = layer(x)

        return self.final_conv(x)

    def compute_loss(self, x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber"):
        noise = torch.randn_like(x_start)
        x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise)
        predicted_noise = self(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError("Loss type not supported: choose 'l1', 'l2', or 'huber'.")

        return loss

    def sample_images(self, num_samples, betas, device):
        self.eval()
        with torch.no_grad():
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, axis=0)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
            posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

            samples = []
            for _ in range(num_samples):
                img = torch.randn(1, 1, 28, 28, device=device)
                for i in reversed(range(0, self.timesteps)):
                    t_tensor = torch.full((1,), i, device=device, dtype=torch.long)
                    img = self.p_sample(img, t_tensor, i, betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance)
                samples.append(img.squeeze().cpu().numpy())
            return samples

    @torch.no_grad()
    def p_sample(self, x, t, t_index, betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance):
        alphas_cumprod_prev = F.pad(sqrt_alphas_cumprod[:-1], (1, 0), value=1.0)
        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / (1 - betas_t))

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise