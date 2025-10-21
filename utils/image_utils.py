from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at path: {image_path}")
    return image


def preprocess_image(image: np.ndarray, size: Tuple[int, int] | None = None) -> np.ndarray:
    if size is not None:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def compute_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    x_min, y_min, x_max, y_max = bbox
    return max(0, x_max - x_min) * max(0, y_max - y_min)


def bbox_to_dimensions(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x_min, y_min, x_max, y_max = bbox
    return max(0, x_max - x_min), max(0, y_max - y_min)
