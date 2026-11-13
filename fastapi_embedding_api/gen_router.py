"""API endpoints for the Energy and Diffusion models (CIFAR-10). Models load on first use."""
import io
import math
from pathlib import Path
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from PIL import Image
from torchvision.utils import make_grid

from diffusion_model import DiffusionModel, UNet
from energy_model import EnergyModel, generate_samples

router = APIRouter()
HERE = Path(__file__).parent
_cache = {}


def _png(images: torch.Tensor) -> Response:
    """images in [0,1], shape (N,3,32,32) -> one PNG grid, upscaled 4x for visibility."""
    grid = make_grid(images.cpu(), nrow=max(1, math.ceil(math.sqrt(images.shape[0]))), padding=2)
    arr = (grid.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    img = Image.fromarray(arr).resize((arr.shape[1] * 4, arr.shape[0] * 4), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


def _energy():
    if "energy" not in _cache:
        path = HERE / "energy_cifar10.pt"
        if not path.exists():
            raise HTTPException(503, "energy_cifar10.pt not found - run train_energy.py first.")
        ckpt = torch.load(path, map_location="cpu")
        model = EnergyModel()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _cache["energy"] = (model, ckpt["buffer"].float())
    return _cache["energy"]


def _diffusion():
    if "diffusion" not in _cache:
        path = HERE / "diffusion_cifar10.pt"
        if not path.exists():
            raise HTTPException(503, "diffusion_cifar10.pt not found - run train_diffusion.py first.")
        ckpt = torch.load(path, map_location="cpu")
        model = DiffusionModel(UNet(image_size=32, num_channels=3))
        model.ema_network.load_state_dict(ckpt["ema_state"])
        model.set_normalizer(ckpt["mean"], ckpt["std"])
        model.eval()
        _cache["diffusion"] = model
    return _cache["diffusion"]


@router.get("/generate/energy", response_class=Response)
def generate_energy(
    num_images: int = Query(8, ge=1, le=16),
    steps: int = Query(150, ge=1, le=500, description="Langevin steps"),
    init: str = Query("buffer", pattern="^(buffer|noise)$",
                      description="start from saved replay-buffer samples or from pure noise"),
    seed: Optional[int] = None,
):
    """Sample images from the energy model with Langevin dynamics. Returns a PNG grid."""
    model, buffer = _energy()
    if seed is not None:
        torch.manual_seed(seed)
    if init == "buffer":
        x = buffer[torch.randint(0, buffer.shape[0], (num_images,))].clone()
    else:
        x = torch.rand((num_images, 3, 32, 32)) * 2 - 1
    x = generate_samples(model, x, steps=steps, step_size=10.0, noise_std=0.01)
    return _png((x + 1) / 2)                      # [-1,1] -> [0,1]


@router.get("/generate/diffusion", response_class=Response)
def generate_diffusion(
    num_images: int = Query(8, ge=1, le=16),
    steps: int = Query(50, ge=1, le=200, description="reverse-diffusion steps"),
    seed: Optional[int] = None,
):
    """Generate images by iteratively denoising pure Gaussian noise. Returns a PNG grid."""
    model = _diffusion()
    if seed is not None:
        torch.manual_seed(seed)
    return _png(model.generate(num_images, steps=steps))
