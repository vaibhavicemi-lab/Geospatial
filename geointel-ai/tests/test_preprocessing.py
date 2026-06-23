import numpy as np

from preprocessing.preprocess import preprocess_pair


def test_preprocess_pair_resizes_to_common_shape():
    before = np.zeros((120, 100, 3), dtype=np.uint8)
    after = np.zeros((80, 150, 3), dtype=np.uint8)
    b, a = preprocess_pair(before, after)
    assert b.shape == a.shape
    assert b.shape[:2] == (80, 100)


def test_preprocess_pair_normalizes():
    before = np.full((10, 10, 3), 255, dtype=np.uint8)
    after = np.zeros((10, 10, 3), dtype=np.uint8)
    b, a = preprocess_pair(before, after, normalize=True)
    assert b.max() == 1.0
    assert a.min() == 0.0
