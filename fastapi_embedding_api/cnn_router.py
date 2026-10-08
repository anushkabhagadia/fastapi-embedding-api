import io
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from cnn_model import CNN

router = APIRouter()

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

_model = CNN()
_model.load_state_dict(torch.load(Path(__file__).parent / "cnn_cifar10.pt", map_location="cpu"))
_model.eval()   # inference mode

_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])


@router.post("/classify_image")
async def classify_image(file: UploadFile = File(...)):
    """Upload an image; returns the predicted CIFAR10 class."""
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read file as an image.")
    x = _transform(img).unsqueeze(0)           # (1,3,64,64)
    with torch.no_grad():
        probs = F.softmax(_model(x), dim=1)[0]
    idx = int(probs.argmax())
    return {"class": CLASSES[idx], "class_index": idx, "confidence": round(float(probs[idx]), 4)}
