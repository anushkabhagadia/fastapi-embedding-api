import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * freqs.view(1, 1, 1, -1))

    def forward(self, x):                      # x: (B,1,1,1) -> (B,1,1,2*num_frequencies)
        x = x.expand(-1, 1, 1, self.num_frequencies)
        return torch.cat([torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1)


def offset_cosine_diffusion_schedule(t, min_signal_rate=0.02, max_signal_rate=0.95):
    shape = t.shape
    t = t.flatten()
    start = torch.acos(torch.tensor(max_signal_rate, device=t.device))
    end = torch.acos(torch.tensor(min_signal_rate, device=t.device))
    angles = start + t * (end - start)
    return torch.sin(angles).reshape(shape), torch.cos(angles).reshape(shape)   # noise, signal


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        r = self.proj(x)
        x = self.conv1(x)
        x = x * torch.sigmoid(x)               # swish
        return self.conv2(x) + r


class DownBlock(nn.Module):
    def __init__(self, width, depth, in_ch):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(ResidualBlock(in_ch, width))
            in_ch = width
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, skips):
        for b in self.blocks:
            x = b(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, width, depth, in_ch):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(ResidualBlock(in_ch + width, width))
            in_ch = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for b in self.blocks:
            x = b(torch.cat([x, skips.pop()], dim=1))
        return x


class UNet(nn.Module):
    """Input: noisy image + noise variance.  Output: predicted noise (same shape as the image)."""

    def __init__(self, image_size=32, num_channels=3, embedding_dim=32):
        super().__init__()
        self.image_size, self.num_channels = image_size, num_channels
        self.initial = nn.Conv2d(num_channels, 32, 1)
        self.embedding = SinusoidalEmbedding(num_frequencies=embedding_dim // 2)
        self.down1 = DownBlock(32, 2, in_ch=64)     # 32 (image) + 32 (embedding) channels
        self.down2 = DownBlock(64, 2, in_ch=32)
        self.down3 = DownBlock(96, 2, in_ch=64)
        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)
        self.up1 = UpBlock(96, 2, in_ch=128)
        self.up2 = UpBlock(64, 2, in_ch=96)
        self.up3 = UpBlock(32, 2, in_ch=64)
        self.final = nn.Conv2d(32, num_channels, 1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        emb = self.embedding(noise_variances).permute(0, 3, 1, 2)
        emb = F.interpolate(emb, size=(self.image_size, self.image_size), mode="nearest")
        x = torch.cat([x, emb], dim=1)
        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)
        x = self.mid2(self.mid1(x))
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)


class DiffusionModel(nn.Module):
    def __init__(self, network, schedule_fn=offset_cosine_diffusion_schedule, ema_decay=0.99):
        super().__init__()
        self.network = network
        self.ema_network = copy.deepcopy(network).eval()
        self.ema_decay, self.schedule_fn = ema_decay, schedule_fn
        self.register_buffer("mean", torch.zeros(1, network.num_channels, 1, 1))
        self.register_buffer("std", torch.ones(1, network.num_channels, 1, 1))

    def set_normalizer(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean).view(1, -1, 1, 1))
        self.std.copy_(torch.as_tensor(std).view(1, -1, 1, 1))

    def denormalize(self, x):
        return torch.clamp(x * self.std + self.mean, 0.0, 1.0)

    def denoise(self, noisy, noise_rates, signal_rates, training):
        net = self.network if training else self.ema_network
        net.train(training)
        pred_noises = net(noisy, noise_rates ** 2)
        pred_images = (noisy - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    @torch.no_grad()
    def reverse_diffusion(self, initial_noise, steps):
        step_size = 1.0 / steps
        x = initial_noise
        for s in range(steps):
            t = torch.ones((x.shape[0], 1, 1, 1), device=x.device) * (1 - s * step_size)
            nr, sr = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(x, nr, sr, training=False)
            nnr, nsr = self.schedule_fn(t - step_size)
            x = nsr * pred_images + nnr * pred_noises
        return pred_images

    @torch.no_grad()
    def generate(self, num_images, steps, image_size=32):
        device = next(self.parameters()).device
        noise = torch.randn((num_images, self.network.num_channels, image_size, image_size), device=device)
        return self.denormalize(self.reverse_diffusion(noise, steps))

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.mean) / self.std
        noises = torch.randn_like(images)
        t = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        nr, sr = self.schedule_fn(t)
        noisy = sr * images + nr * noises                         # forward (noising) process
        pred_noises, _ = self.denoise(noisy, nr, sr, training=True)
        loss = loss_fn(pred_noises, noises)                       # compare predicted vs true noise

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():                                     # EMA copy used for sampling
            for e, p in zip(self.ema_network.parameters(), self.network.parameters()):
                e.copy_(self.ema_decay * e + (1 - self.ema_decay) * p)
            for e, b in zip(self.ema_network.buffers(), self.network.buffers()):
                e.copy_(b)
        return loss.item()

    @torch.no_grad()
    def test_step(self, images, loss_fn):
        images = (images - self.mean) / self.std
        noises = torch.randn_like(images)
        t = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        nr, sr = self.schedule_fn(t)
        pred_noises, _ = self.denoise(sr * images + nr * noises, nr, sr, training=False)
        return loss_fn(pred_noises, noises).item()
