from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from utils.logging_utils import get_logger


logger = get_logger(__name__)

try:
    import torch
except ImportError as exc:  # pragma: no cover
    torch = None
    logger.warning("Torch is required for depth estimation: %s", exc)


class DepthEstimator:
    def __init__(self, model_type: str = "DPT_Large", device: str = "auto") -> None:
        if torch is None:
            raise ImportError("PyTorch is required for depth estimation")
        self.model_type = model_type
        self.device = device
        self._model = None
        self._transform = None

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading MiDaS depth model %s", self.model_type)
        self._model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self._model.eval()
        if self.device != "auto":
            self._model.to(self.device)
        else:
            preferred = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(preferred)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type in {"DPT_Large", "DPT_Hybrid"}:
            self._transform = midas_transforms.dpt_transform
        else:
            self._transform = midas_transforms.small_transform

    def infer(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            self._load()
        except Exception as exc:
            logger.error("Could not load depth model: %s", exc)
            return None

        if self._model is None or self._transform is None:
            return None

        input_batch = self._transform(image).to(next(self._model.parameters()).device)
        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        return depth_map
