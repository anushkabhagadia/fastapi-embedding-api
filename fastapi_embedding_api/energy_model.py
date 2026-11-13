import random

import numpy as np
import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
   
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)  # 32 -> 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)           # 16 -> 8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)          # 8 -> 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)         # 4 -> 2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


def generate_samples(model, inp_imgs, steps, step_size, noise_std):
    model.eval()
    for _ in range(steps):
        with torch.no_grad():
            inp_imgs = (inp_imgs + torch.randn_like(inp_imgs) * noise_std).clamp(-1.0, 1.0)
        inp_imgs.requires_grad_(True)

        energy = model(inp_imgs)
        # gradient of energy w.r.t. the images (one energy per image -> grad_outputs of ones)
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))

        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)                      # stabilise sampling
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)
    return inp_imgs.detach()


class Buffer:

    def __init__(self, model, device, channels=3, size=32, batch=128):
        self.model, self.device = model, device
        self.channels, self.size, self.batch = channels, size, batch
        self.examples = [self._noise(1) for _ in range(batch)]

    def _noise(self, n):
        return torch.rand((n, self.channels, self.size, self.size), device=self.device) * 2 - 1

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(self.batch, 0.05)             # ~5% restart from pure noise
        new_rand = self._noise(n_new)
        old = torch.cat(random.choices(self.examples, k=self.batch - n_new), dim=0)
        inp = torch.cat([new_rand, old], dim=0)

        new_imgs = generate_samples(self.model, inp, steps, step_size, noise)

        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:8192]
        return new_imgs


class EBM:

    def __init__(self, model, alpha, steps, step_size, noise, device):
        self.model, self.device = model, device
        self.buffer = Buffer(model, device)
        self.alpha, self.steps, self.step_size, self.noise = alpha, steps, step_size, noise

    def train_step(self, real_imgs, optimizer):
        real_imgs = (real_imgs + torch.randn_like(real_imgs) * self.noise).clamp(-1.0, 1.0)
        fake_imgs = self.buffer.sample_new_exmps(self.steps, self.step_size, self.noise)

        inp = torch.cat([real_imgs, fake_imgs], dim=0).detach()
        out = self.model(inp)
        real_out, fake_out = torch.split(out, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

        cdiv_loss = real_out.mean() - fake_out.mean()
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
        loss = cdiv_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        optimizer.step()

        return {"loss": loss.item(), "cdiv": cdiv_loss.item(),
                "real": real_out.mean().item(), "fake": fake_out.mean().item()}

    @torch.no_grad()
    def test_step(self, real_imgs):
        fake = torch.rand_like(real_imgs) * 2 - 1
        out = self.model(torch.cat([real_imgs, fake], dim=0))
        real_out, fake_out = torch.split(out, real_imgs.size(0), dim=0)
        return {"real": real_out.mean().item(), "fake": fake_out.mean().item()}
