import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, hidden_dim):
        super(Generator, self).__init__()
        self.init_size = 7  # Initial size before upsampling (for 28x28 images)
        self.hidden_dim = hidden_dim
        
        self.fc = nn.Linear(latent_dim, hidden_dim * self.init_size * self.init_size)

        self.model = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim // 4, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), self.hidden_dim, self.init_size, self.init_size)
        img = self.model(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_channels, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(hidden_dim // 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.adversarial_loss = nn.BCELoss()

    def discriminator_loss(self, discriminator, real_imgs, fake_imgs, real_labels, fake_labels):
        real_loss = self.adversarial_loss(discriminator(real_imgs), real_labels)
        fake_loss = self.adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)
        return (real_loss + fake_loss) / 2

    def generator_loss(self, discriminator, fake_imgs, real_labels):
        return self.adversarial_loss(discriminator(fake_imgs), real_labels)
