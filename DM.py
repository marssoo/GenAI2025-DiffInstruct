import torch
import torch.nn as nn

# Define the U-Net architecture
class UNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers):
        super(UNet, self).__init__()
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        # Down-sampling layers
        for _ in range(num_layers):
            self.down_layers.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
            )
            input_channels = hidden_channels

        # Up-sampling layers
        for _ in range(num_layers):
            self.up_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=2, stride=2),
                    nn.ReLU()
                )
            )
            hidden_channels //= 2

        self.final_layer = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        skips = []

        # Down-sampling
        for layer in self.down_layers:
            x = layer(x)
            skips.append(x)

        # Up-sampling
        for layer in self.up_layers:
            x = layer(x)
            x += skips.pop()

        return self.final_layer(x)