"""Croping images."""

import numpy as np


def crop_image(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Croping image."""
    return image[y1:y2, x1:x2, :]
