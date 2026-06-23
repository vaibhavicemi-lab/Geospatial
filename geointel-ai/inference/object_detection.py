from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import MIN_REGION_AREA


def analyze_changed_regions(mask: np.ndarray, min_area: int = MIN_REGION_AREA) -> Dict:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[Dict] = []
    total_area = mask.shape[0] * mask.shape[1]
    changed_pixels = int(np.count_nonzero(binary))

    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        regions.append({"bbox": (int(x), int(y), int(w), int(h)), "area": area})

    change_percentage = (changed_pixels / total_area) * 100 if total_area else 0.0
    largest_region = max((r["area"] for r in regions), default=0)
    confidence_score = min(99.0, 35.0 + change_percentage * 2.2 + min(largest_region / max(total_area, 1), 1) * 40)

    return {
        "changed_pixels": changed_pixels,
        "change_percentage": round(float(change_percentage), 3),
        "number_of_regions": len(regions),
        "regions": regions,
        "confidence_score": round(float(confidence_score), 2),
    }


def draw_bounding_boxes(image: np.ndarray, regions: List[Dict]) -> np.ndarray:
    boxed = image.copy()
    for region in regions:
        x, y, w, h = region["bbox"]
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (255, 220, 0), 2)
    return boxed
