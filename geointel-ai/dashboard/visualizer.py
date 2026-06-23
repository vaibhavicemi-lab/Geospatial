from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    rgb[mask > 0] = [255, 42, 42]
    return rgb


def create_mask_preview(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    preview = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    preview[:, :, :] = [8, 18, 32]
    preview[mask > 0] = [255, 50, 50]
    return preview


def draw_bounding_boxes(image: np.ndarray, regions: List[Dict]) -> np.ndarray:
    boxed = image.copy()
    for region in regions:
        x, y, w, h = region.get("bbox", (0, 0, 0, 0))
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (56, 189, 248), 2)
        cv2.putText(boxed, "change", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)
    return boxed


def create_overlay_comparison(after: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    red_layer = np.zeros_like(after)
    red_layer[:, :, 0] = 255
    overlay = after.copy()
    mask_bool = mask > 0
    blended = cv2.addWeighted(after, 1 - alpha, red_layer, alpha, 0)
    overlay[mask_bool] = blended[mask_bool]
    return overlay


def create_side_by_side_view(before: np.ndarray, after: np.ndarray, mask: np.ndarray, overlay: np.ndarray):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), facecolor="#060b16")
    titles = ["Before", "After", "Change Mask", "Overlay"]
    images = [before, after, create_mask_preview(mask), overlay]
    for ax, title, image in zip(axes, titles, images):
        ax.imshow(image)
        ax.set_title(title, color="#e0f2fe", fontsize=11, fontweight="bold")
        ax.axis("off")
        ax.set_facecolor("#060b16")
    fig.tight_layout()
    return fig


def comparison_figure(before: np.ndarray, after: np.ndarray, mask: np.ndarray, overlay: np.ndarray):
    return create_side_by_side_view(before, after, mask, overlay)


def format_change_percentage(value: float) -> str:
    return f"{float(value):.2f}%"


def format_alert_level_badge(level: str) -> str:
    colors = {
        "Low": "#22c55e",
        "Medium": "#f59e0b",
        "High": "#ef4444",
        "Critical": "#7f1d1d",
    }
    color = colors.get(level, "#38bdf8")
    return f'<span style="background:{color}; color:white; padding:0.25rem 0.55rem; border-radius:999px; font-weight:800;">{level}</span>'
