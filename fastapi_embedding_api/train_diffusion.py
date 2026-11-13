"""Train the diffusion model on CIFAR-10.   python train_diffusion.py --epochs 20"""
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from diffusion_model import DiffusionModel, UNet

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--max_batches", type=int, default=0, help="cap batches/epoch for a quick run")
args = p.parse_args()

torch.manual_seed(42)
device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
print("device:", device)

tf = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])          # pixels in [0, 1]
train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=T.ToTensor())
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, drop_last=True)

# per-channel mean/std of the training pixels
data = torch.tensor(train_ds.data, dtype=torch.float32).div(255)   # (50000,32,32,3)
mean, std = data.mean(dim=(0, 1, 2)), data.std(dim=(0, 1, 2))
print("mean:", mean.tolist(), "std:", std.tolist())

model = DiffusionModel(UNet(image_size=32, num_channels=3)).to(device)
model.set_normalizer(mean, std)
print(f"UNet parameters: {sum(p.numel() for p in model.network.parameters()):,}")
opt = torch.optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.L1Loss()

for epoch in range(args.epochs):
    model.train()
    losses = []
    for i, (x, _) in enumerate(tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")):
        if args.max_batches and i >= args.max_batches:
            break
        losses.append(model.train_step(x.to(device), opt, loss_fn))
    val = [model.test_step(x.to(device), loss_fn) for _, (x, _) in zip(range(20), test_loader)]
    print(f"epoch {epoch+1}: train loss {sum(losses)/len(losses):.4f} | val loss {sum(val)/len(val):.4f}")

    model.eval()
    save_image(model.generate(16, steps=20), "diffusion_samples.png", nrow=8)
    torch.save({"ema_state": model.ema_network.state_dict(),
                "mean": mean.tolist(), "std": std.tolist()}, "diffusion_cifar10.pt")
print("saved diffusion_cifar10.pt and diffusion_samples.png")
