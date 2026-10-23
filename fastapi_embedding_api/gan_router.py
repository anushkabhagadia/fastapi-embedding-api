import io

import torch
from fastapi import APIRouter, Query
from fastapi.responses import Response
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from gan_model import Generator, LATENT_DIM

router = APIRouter()

_generator = Generator()
_generator.load_state_dict(torch.load("gan_mnist.pt", map_location="cpu"))
_generator.eval()  # BatchNorm must use running stats at inference time


@router.get("/generate_digits", response_class=Response)
def generate_digits(n: int = Query(16, ge=1, le=64), seed: int | None = None):
    """Return a PNG grid of n generated handwritten digits."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    z = torch.randn(n, LATENT_DIM, generator=gen)
    with torch.no_grad():
        imgs = _generator(z)
    imgs = (imgs + 1) / 2  # [-1, 1] -> [0, 1]
    grid = make_grid(imgs, nrow=int(n ** 0.5) or 1, padding=2)
    buf = io.BytesIO()
    to_pil_image(grid).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
