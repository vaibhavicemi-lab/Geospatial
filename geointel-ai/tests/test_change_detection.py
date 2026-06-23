import numpy as np

from inference.change_detection import classical_change_detection


def test_classical_change_detection_finds_synthetic_square():
    before = np.zeros((128, 128, 3), dtype=np.uint8)
    after = before.copy()
    after[40:80, 40:80] = [255, 255, 255]
    result = classical_change_detection(before, after, threshold=20, min_area=20)
    assert result["analysis"]["changed_pixels"] > 0
    assert result["analysis"]["number_of_regions"] >= 1
    assert result["mask"].shape == (128, 128)
