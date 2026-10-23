import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from gan_model import Generator, Discriminator, LATENT_DIM

EPOCHS = 20
BATCH_SIZE = 128
LR = 2e-4
WEIGHTS_PATH = "gan_mnist.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scale pixels to [-1, 1] to match the generator's Tanh output
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
loader = DataLoader(
    datasets.MNIST("data", train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
)

G, D = Generator().to(device), Discriminator().to(device)
criterion = nn.BCELoss()
opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

for epoch in range(1, EPOCHS + 1):
    for real, _ in loader:
        real = real.to(device)
        b = real.size(0)
        ones = torch.ones(b, 1, device=device)
        zeros = torch.zeros(b, 1, device=device)

        # --- Discriminator step: real -> 1, fake -> 0 ---
        fake = G(torch.randn(b, LATENT_DIM, device=device))
        loss_d = criterion(D(real), ones) + criterion(D(fake.detach()), zeros)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # --- Generator step: try to make D output 1 on fakes ---
        loss_g = criterion(D(fake), ones)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    print(f"Epoch {epoch}/{EPOCHS}  loss_D={loss_d.item():.3f}  loss_G={loss_g.item():.3f}")

torch.save(G.state_dict(), WEIGHTS_PATH)
print(f"Saved generator to {WEIGHTS_PATH}")
