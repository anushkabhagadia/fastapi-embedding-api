import torch
import torch.nn as nn

LATENT_DIM = 100


class Generator(nn.Module):
    """(B, 100) noise -> (B, 1, 28, 28) image in [-1, 1]."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    """(B, 1, 28, 28) image -> (B, 1) probability that the image is real."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(7 * 7 * 128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(start_dim=1)
        return torch.sigmoid(self.fc(h))


if __name__ == "__main__":
    # Quick shape check
    g, d = Generator(), Discriminator()
    z = torch.randn(4, LATENT_DIM)
    fake = g(z)
    print("Generator out:", tuple(fake.shape))      # (4, 1, 28, 28)
    print("Discriminator out:", tuple(d(fake).shape))  # (4, 1)
