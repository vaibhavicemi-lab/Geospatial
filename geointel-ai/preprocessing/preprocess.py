from typing import Optional, Tuple

import cv2
import numpy as np


def resize_to_match(before: np.ndarray, after: np.ndarray, target_size: Optional[Tuple[int, int]] = None):
    if before is None or after is None:
        raise ValueError("Both before and after images are required.")
    if before.ndim != 3 or after.ndim != 3:
        raise ValueError("Images must be RGB arrays.")

    if target_size:
        width, height = target_size
    else:
        height = min(before.shape[0], after.shape[0])
        width = min(before.shape[1], after.shape[1])
    before_resized = cv2.resize(before, (width, height), interpolation=cv2.INTER_AREA)
    after_resized = cv2.resize(after, (width, height), interpolation=cv2.INTER_AREA)
    return before_resized, after_resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.shape[2] == 3:
        return image
    raise ValueError("Unsupported image channel format.")


def preprocess_pair(before: np.ndarray, after: np.ndarray, target_size: Optional[Tuple[int, int]] = None, normalize: bool = False):
    before = ensure_rgb(before)
    after = ensure_rgb(after)
    before, after = resize_to_match(before, after, target_size)
    if normalize:
        return normalize_image(before), normalize_image(after)
    return before, after
