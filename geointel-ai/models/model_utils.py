from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import MODEL_WEIGHTS
from models.siamese_unet import SiameseUNet


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return selected


def load_siamese_model(weights_path: Optional[str] = None, device: Optional[torch.device] = None):
    device = device or get_device()
    path = Path(weights_path) if weights_path else MODEL_WEIGHTS
    model = SiameseUNet().to(device)
    if not path.exists():
        return None, f"Trained weights not found at {path}. Use Classical Demo Mode or train on LEVIR-CD."
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        image_size = state.get("image_size", "unknown")
        message = f"Deep learning model loaded from {path} (trained image size: {image_size})."
    else:
        model.load_state_dict(state)
        message = f"Deep learning model loaded from {path}."
    model.eval()
    return model, message


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)
