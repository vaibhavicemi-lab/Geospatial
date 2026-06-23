from typing import List, Tuple

import numpy as np


def tile_image(image: np.ndarray, tile_size: int = 256, stride: int = 256) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive.")
    tiles = []
    height, width = image.shape[:2]
    for y in range(0, max(height - tile_size + 1, 1), stride):
        for x in range(0, max(width - tile_size + 1, 1), stride):
            tile = image[y : y + tile_size, x : x + tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append((tile, (x, y)))
    if not tiles:
        padded = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
        padded[:height, :width] = image
        tiles.append((padded, (0, 0)))
    return tiles
