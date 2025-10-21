from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.logging_utils import get_logger


logger = get_logger(__name__)

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - dependency missing in tests
    YOLO = None  # type: ignore
    logger.warning("Ultralytics YOLO not available: %s", exc)


@dataclass
class Detection:
    """Container for a single model detection.

    Attributes
    ----------
    label:
        Canonical class label for the detection. The label is remapped
        according to the configuration provided to :class:`YOLODetector`.
    confidence:
        Model confidence score in the ``[0, 1]`` range.
    bbox:
        Bounding box coordinates expressed as a dictionary with ``x_min``,
        ``y_min``, ``x_max`` and ``y_max`` keys.
    """

    label: str
    confidence: float
    bbox: Dict[str, int]


class YOLODetector:
    """Wrapper around the Ultralytics YOLO interface used in this project."""
    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        max_detections: int = 150,
        class_map: Optional[Dict[str, str]] = None,
        device: str = "auto",
    ) -> None:
        if YOLO is None:
            raise ImportError("Ultralytics is required for YOLO detection. Please install ultralytics.")

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.class_map = class_map or {}
        self.device = device

        logger.info("Loading YOLO weights from %s", weights_path)
        try:
            self.model = YOLO(weights_path)
            if device != "auto":
                self.model.to(device)
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)
            raise

        self.model.fuse()

    def detect(self, image: np.ndarray, class_filter: Optional[List[str]] = None) -> List[Detection]:
        """Run inference on the provided ``image`` and return filtered detections."""
        logger.debug("Running YOLO inference")
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=max(image.shape[:2]),
            device=self.device,
            verbose=False,
            max_det=self.max_detections,
        )
        detections: List[Detection] = []
        for result in results:
            if not hasattr(result, "boxes"):
                continue
            boxes = result.boxes.cpu()
            for box in boxes:
                conf = float(box.conf.item())
                cls_idx = int(box.cls.item())
                label = self.model.names.get(cls_idx, str(cls_idx))
                mapped_label = self._remap_label(label)
                if class_filter and mapped_label not in class_filter:
                    continue
                if conf < self.confidence_threshold:
                    continue
                xyxy = box.xyxy.numpy().reshape(-1)
                x_min, y_min, x_max, y_max = xyxy
                detections.append(
                    Detection(
                        label=mapped_label,
                        confidence=conf,
                        bbox={
                            "x_min": int(x_min),
                            "y_min": int(y_min),
                            "x_max": int(x_max),
                            "y_max": int(y_max),
                        },
                    )
                )
        detections = self._non_max_suppression(detections)
        return detections

    def _remap_label(self, label: str) -> str:
        """Map raw model labels to project labels using the configured map."""
        for target, source in self.class_map.items():
            if source.lower() == label.lower():
                return target
        return label.replace(" ", "_").lower()

    def _non_max_suppression(self, detections: List[Detection]) -> List[Detection]:
        """Apply class-aware non max suppression to the detections."""
        if not detections:
            return []

        grouped: Dict[str, List[Detection]] = {}
        for det in detections:
            grouped.setdefault(det.label, []).append(det)

        filtered: List[Detection] = []
        for label, dets in grouped.items():
            boxes = np.array([[d.bbox["x_min"], d.bbox["y_min"], d.bbox["x_max"], d.bbox["y_max"]] for d in dets])
            scores = np.array([d.confidence for d in dets])
            order = scores.argsort()[::-1]
            keep: List[int] = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                if order.size == 1:
                    break
                ious = self._iou(boxes[i], boxes[order[1:]])
                mask = ious <= self.iou_threshold
                order = order[1:][mask]
            filtered.extend(dets[k] for k in keep)
        return filtered

    @staticmethod
    def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
        """Compute IoU between a reference ``box`` and an array of ``others``."""
        if others.size == 0:
            return np.array([])
        x1 = np.maximum(box[0], others[:, 0])
        y1 = np.maximum(box[1], others[:, 1])
        x2 = np.minimum(box[2], others[:, 2])
        y2 = np.minimum(box[3], others[:, 3])

        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        others_area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
        union = box_area + others_area - inter_area
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, inter_area / union, 0.0)
        return iou
