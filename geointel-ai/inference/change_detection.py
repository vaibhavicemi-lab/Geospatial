from typing import Dict, Optional

import cv2
import numpy as np
import torch

from config import CLASSICAL_THRESHOLD, MIN_REGION_AREA
from inference.object_detection import analyze_changed_regions, draw_bounding_boxes
from models.model_utils import get_device, image_to_tensor, load_siamese_model
from preprocessing.preprocess import preprocess_pair


def create_overlay(after: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    red = np.zeros_like(after)
    red[:, :, 0] = 255
    mask_bool = mask > 0
    overlay = after.copy()
    overlay[mask_bool] = cv2.addWeighted(after, 1 - alpha, red, alpha, 0)[mask_bool]
    return overlay


def classical_change_detection(before: np.ndarray, after: np.ndarray, threshold: int = CLASSICAL_THRESHOLD, min_area: int = MIN_REGION_AREA) -> Dict:
    before, after = preprocess_pair(before, after)
    before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(before_gray, after_gray)
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(closed)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    analysis = analyze_changed_regions(mask, min_area)
    overlay = create_overlay(after, mask)
    boxed_overlay = draw_bounding_boxes(overlay, analysis["regions"])
    return {
        "before": before,
        "after": after,
        "diff": diff,
        "mask": mask,
        "overlay": boxed_overlay,
        "analysis": analysis,
        "mode_message": "Classical demo mode completed using image differencing, thresholding, morphology, and contours.",
        "used_deep_learning": False,
    }


def deep_learning_change_detection(
    before: np.ndarray,
    after: np.ndarray,
    weights_path: Optional[str] = None,
    fallback_threshold: int = CLASSICAL_THRESHOLD,
    fallback_min_area: int = MIN_REGION_AREA,
) -> Dict:
    before, after = preprocess_pair(before, after, target_size=(256, 256))
    device = get_device()
    model, message = load_siamese_model(weights_path, device)
    if model is None:
        result = classical_change_detection(before, after, threshold=fallback_threshold, min_area=fallback_min_area)
        result["mode_message"] = message + " Fallback classical mode was used."
        result["used_deep_learning"] = False
        return result

    with torch.no_grad():
        before_tensor = image_to_tensor(before, device)
        after_tensor = image_to_tensor(after, device)
        prob = model(before_tensor, after_tensor).squeeze().cpu().numpy()
    mask = (prob >= 0.5).astype(np.uint8) * 255
    analysis = analyze_changed_regions(mask)
    overlay = draw_bounding_boxes(create_overlay(after, mask), analysis["regions"])
    return {
        "before": before,
        "after": after,
        "diff": (prob * 255).astype(np.uint8),
        "mask": mask,
        "overlay": overlay,
        "analysis": analysis,
        "mode_message": message,
        "used_deep_learning": True,
    }
