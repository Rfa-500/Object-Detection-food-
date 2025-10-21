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
    """Thin wrapper around MiDaS depth estimation with mobile optimisations."""

    def __init__(
        self,
        model_type: str = "DPT_Large",
        device: str = "auto",
        optimize_for_mobile: bool = False,
        max_input_size: Optional[int] = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for depth estimation")
        self.model_type = model_type
        self.device = device
        self.optimize_for_mobile = optimize_for_mobile
        self.max_input_size = int(max_input_size) if max_input_size else None
        self._model = None
        self._transform = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.optimize_for_mobile and self.model_type == "DPT_Large":
            # Automatically prefer the lightest MiDaS model when optimising for
            # resource constrained devices such as phones or edge boards.
            self.model_type = "MiDaS_small"

        logger.info("Loading MiDaS depth model %s", self.model_type)
        self._model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self._model.eval()
        if self.device != "auto":
            self._model.to(self.device)
        else:
            preferred = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(preferred)
        if self.optimize_for_mobile and not torch.cuda.is_available():
            # Dynamic quantisation keeps accuracy high while reducing CPU load.
            try:
                self._model = torch.quantization.quantize_dynamic(
                    self._model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as exc:  # pragma: no cover - quantisation is optional
                logger.warning("Mobile depth optimisation skipped: %s", exc)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type in {"DPT_Large", "DPT_Hybrid"}:
            self._transform = midas_transforms.dpt_transform
        else:
            self._transform = midas_transforms.small_transform

    def infer(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Return a normalised depth map or ``None`` if inference fails."""
        try:
            self._load()
        except Exception as exc:
            logger.error("Could not load depth model: %s", exc)
            return None

        if self._model is None or self._transform is None:
            return None

        if self.max_input_size:
            image = _resize_long_edge(image, self.max_input_size)

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


def _resize_long_edge(image: np.ndarray, max_size: int) -> np.ndarray:
    """Resize image preserving aspect ratio if its long edge exceeds ``max_size``."""

    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_size:
        return image

    scale = max_size / float(long_edge)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
