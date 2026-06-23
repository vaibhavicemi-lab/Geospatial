from collections import Counter
from typing import Dict, List, Sequence

import cv2
import numpy as np


def crop_changed_regions(after_image: np.ndarray, change_regions: Sequence[Dict]) -> List[np.ndarray]:
    crops = []
    for region in change_regions:
        x, y, w, h = region.get("bbox", (0, 0, 0, 0))
        if w > 0 and h > 0:
            crops.append(after_image[y : y + h, x : x + w].copy())
    return crops


def _features(crop: np.ndarray) -> Dict[str, float]:
    if crop.size == 0:
        return {"green": 0, "blue": 0, "gray": 0, "edge_density": 0, "rectangularity": 0}
    rgb = crop.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangularity = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = max(cv2.contourArea(contour), 1)
        x, y, w, h = cv2.boundingRect(contour)
        rectangularity = min(1.0, area / max(w * h, 1))
    return {
        "green": float(np.mean(g - np.maximum(r, b))),
        "blue": float(np.mean(b - np.maximum(r, g))),
        "gray": float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))),
        "brightness": float(np.mean(gray)),
        "edge_density": float(np.count_nonzero(edges) / edges.size),
        "rectangularity": rectangularity,
    }


def classify_changed_region(before_crop: np.ndarray, after_crop: np.ndarray) -> Dict[str, str]:
    before = _features(before_crop)
    after = _features(after_crop)
    green_drop = before["green"] - after["green"]
    blue_gain = after["blue"] - before["blue"]
    edge_gain = after["edge_density"] - before["edge_density"]
    brightness_gain = after["brightness"] - before["brightness"]

    label = "unknown change"
    confidence = "Low"

    if green_drop > 18 and after["edge_density"] > 0.04:
        label, confidence = "vegetation removed", "High"
    elif blue_gain > 15 and after["blue"] > 5:
        label, confidence = "water expansion", "Medium"
    elif after["rectangularity"] > 0.45 and edge_gain > 0.025 and brightness_gain > 5:
        label, confidence = "new building", "Medium"
    elif before["rectangularity"] > 0.4 and after["edge_density"] < before["edge_density"] * 0.75:
        label, confidence = "removed building", "Medium"
    elif after["edge_density"] > 0.08 and after["gray"] < 34 and after_crop.shape[0] < after_crop.shape[1] * 0.45:
        label, confidence = "road extension", "Medium"
    elif after["green"] > 5 and after["rectangularity"] > 0.35 and after["edge_density"] > 0.05:
        label, confidence = "possible sports court / tennis court", "Low"
    elif edge_gain > 0.04 or brightness_gain > 20:
        label, confidence = "construction activity", "Medium"
    elif abs(brightness_gain) > 12:
        label, confidence = "bare land change", "Low"

    return {"type": label, "confidence": confidence}


def generate_plain_english_explanation(change_info: Dict) -> str:
    change_type = change_info.get("type", "unknown change")
    confidence = change_info.get("confidence", "Low")
    prefix = "Possible" if confidence == "Low" else "Likely" if confidence == "Medium" else ""
    phrases = {
        "new building": "new building detected in the changed region.",
        "removed building": "building removal detected in the changed region.",
        "road extension": "road-like extension detected.",
        "vegetation removed": "vegetation appears to be removed and replaced by non-vegetated land.",
        "water expansion": "water-covered area appears to have expanded.",
        "bare land change": "bare land or exposed soil change detected.",
        "possible sports court / tennis court": "sports court or tennis court-like structure detected.",
        "construction activity": "construction activity or newly developed surface detected.",
        "unknown change": "unknown change detected, analyst review required.",
    }
    sentence = phrases.get(change_type, phrases["unknown change"])
    return f"{prefix} {sentence}".strip()


def explain_all_changes(before_image: np.ndarray, after_image: np.ndarray, change_mask: np.ndarray, regions: Sequence[Dict]) -> Dict:
    explanations = []
    for index, region in enumerate(regions, start=1):
        x, y, w, h = region.get("bbox", (0, 0, 0, 0))
        before_crop = before_image[y : y + h, x : x + w]
        after_crop = after_image[y : y + h, x : x + w]
        classification = classify_changed_region(before_crop, after_crop)
        explanation = generate_plain_english_explanation(classification)
        if classification["confidence"] == "Low":
            explanation += " Analyst verification is recommended before making a final conclusion."
        explanations.append(
            {
                "region": index,
                "bbox": (int(x), int(y), int(w), int(h)),
                "detected_change_type": classification["type"],
                "confidence": classification["confidence"],
                "explanation": explanation,
            }
        )

    if not explanations:
        return {
            "main_summary": "No significant changed regions were detected.",
            "region_explanations": [],
            "detected_change_type": "no significant change",
            "overall_confidence": "High",
            "analyst_recommendation": "Routine monitoring is sufficient.",
        }

    counts = Counter(item["detected_change_type"] for item in explanations)
    main_type = counts.most_common(1)[0][0]
    confidences = [item["confidence"] for item in explanations]
    overall = "High" if "High" in confidences and "Low" not in confidences else "Medium" if "Medium" in confidences else "Low"
    summary = generate_plain_english_explanation({"type": main_type, "confidence": overall})
    recommendation = "Analyst review recommended."
    if overall == "Low":
        recommendation = "Low confidence explanation: analyst must verify the changed regions manually."
    return {
        "main_summary": summary,
        "region_explanations": explanations,
        "detected_change_type": main_type,
        "overall_confidence": overall,
        "analyst_recommendation": recommendation,
    }
