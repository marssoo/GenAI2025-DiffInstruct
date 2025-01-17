import torch
import torch.nn as nn

class SimpleDM(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers):
        super(SimpleDM, self).__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        layers = []
        for i in range(num_layers):
            if i == 0:
                # First layer accounts for input_channels + hidden_channels from timestep embedding
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()
                    )
                )

        self.model = nn.Sequential(*layers, nn.Conv2d(hidden_channels, 1, kernel_size=1))

    def forward(self, x, t):
        time_embed = self.time_embedding(t.view(-1, 1)).view(t.size(0), -1, 1, 1)
        time_embed = time_embed.expand(-1, -1, x.size(2), x.size(3))  # Match spatial dimensions
        x = torch.cat([x, time_embed], dim=1)  # Concatenate along channel dimension
        return self.model(x)


    def sample_images(self, num_samples, timesteps, device):
        """
        Generate sample images using the trained diffusion model.

        Args:
            num_samples (int): Number of images to generate.
            timesteps (int): Number of diffusion steps.
            device (torch.device): Device to perform the computations.

        Returns:
            list: List of generated sample images as numpy arrays.
        """
        self.eval()
        samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                x = torch.randn(1, 1, 28, 28).to(device)  # Start with random noise
                for t in reversed(range(1, timesteps + 1)):
                    t_tensor = torch.full((1,), t, device=device, dtype=torch.float32)
                    x = self(x, t_tensor)
                samples.append(x.squeeze().cpu().numpy())
        return samples