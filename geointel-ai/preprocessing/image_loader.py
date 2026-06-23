from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image, UnidentifiedImageError


def _open_rgb(source: Union[str, Path, BinaryIO]) -> Image.Image:
    try:
        image = Image.open(source)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Invalid image file. Please upload a PNG, JPG, JPEG, or TIFF image.") from exc


def load_image(path: Union[str, Path]) -> np.ndarray:
    return np.array(_open_rgb(path))


def load_uploaded_image(uploaded_file) -> np.ndarray:
    if uploaded_file is None:
        raise ValueError("No image uploaded.")
    return np.array(_open_rgb(uploaded_file))
