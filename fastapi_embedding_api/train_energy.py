"""Train the energy model on CIFAR-10.   python train_energy.py --epochs 5"""
import argparse

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from energy_model import EnergyModel, EBM

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--steps", type=int, default=60, help="Langevin steps per batch (lower = faster)")
p.add_argument("--max_batches", type=int, default=0, help="cap batches/epoch for a quick run (0 = all)")
args = p.parse_args()

torch.manual_seed(42)
device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
print("device:", device)

# pixels -> [-1, 1]
tf = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, drop_last=True)

model = EnergyModel().to(device)
ebm = EBM(model, alpha=0.1, steps=args.steps, step_size=10, noise=0.005, device=device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))

for epoch in range(args.epochs):
    last = {}
    for i, (x, _) in enumerate(tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")):
        if args.max_batches and i >= args.max_batches:
            break
        last = ebm.train_step(x.to(device), opt)
    x, _ = next(iter(test_loader))
    val = ebm.test_step(x.to(device))
    print(f"epoch {epoch+1}: " + ", ".join(f"{k}={v:.3f}" for k, v in last.items())
          + f" | val real={val['real']:.3f} fake(noise)={val['fake']:.3f}")
    save_image((torch.cat(ebm.buffer.examples[:16]) + 1) / 2, "energy_samples.png", nrow=8)

# Save weights AND a small slice of the replay buffer (used by the API as a warm start)
buf = torch.cat(ebm.buffer.examples[:256]).cpu().half()
torch.save({"state_dict": model.state_dict(), "buffer": buf}, "energy_cifar10.pt")
print("saved energy_cifar10.pt and energy_samples.png")
