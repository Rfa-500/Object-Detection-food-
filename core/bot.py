from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from core.config_loader import AppConfig
from estimation.portion_estimator import PortionEstimate, PortionEstimator
from models.depth import DepthEstimator
from models.detector import Detection, YOLODetector
from nutrition.database import NutritionDatabase
from utils.image_utils import load_image
from utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class DetectionResult:
    """Bundle the detection, portion estimate, nutrition and occlusion context."""

    detection: Detection
    portion: PortionEstimate
    nutrition: Optional[Dict[str, float]]
    detection_reliability: float
    occlusion_ratio: float
    is_occluded: bool


class FoodDetectionBot:
    """High level orchestrator combining detection, portioning and nutrition."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.nutrition_db = NutritionDatabase()
        self.detector = YOLODetector(
            weights_path=self.config.yolo.get("weights", "yolov8n.pt"),
            confidence_threshold=float(self.config.yolo.get("confidence_threshold", 0.4)),
            iou_threshold=float(self.config.yolo.get("iou_threshold", 0.5)),
            max_detections=int(self.config.yolo.get("max_detections", 150)),
            class_map=self.config.yolo.get("class_map", {}),
            device=self.config.yolo.get("device", "auto"),
        )
        self.reference_classes = {
            key: {k: float(v) for k, v in value.items()}
            for key, value in self.config.reference_classes.items()
        }
        self.portion_estimator = PortionEstimator(
            reference_objects=self.reference_classes,
            density_overrides={k: float(v) for k, v in self.config.portion.get("class_density_overrides", {}).items()},
            default_density=float(self.config.portion.get("default_density_g_per_cm3", 0.65)),
            fallback_mass_g=float(self.config.portion.get("fallback_mass_g", 150)),
            base_uncertainty_pct=float(self.config.portion.get("uncertainty_pct", 25)),
        )
        self.depth_estimator: Optional[DepthEstimator]
        try:
            self.depth_estimator = DepthEstimator(
                model_type=self.config.depth.get("model_type", "DPT_Large"),
                device=self.config.depth.get("device", "auto"),
                optimize_for_mobile=bool(self.config.depth.get("optimize_for_mobile", False)),
                max_input_size=self.config.depth.get("max_input_size"),
            )
        except Exception as exc:
            logger.warning("Depth estimator disabled: %s", exc)
            self.depth_estimator = None

        self.food_classes = list(self.config.yolo.get("class_map", {}).keys())
        self.runtime_opts = self.config.runtime
        self.occlusion_threshold = float(self.runtime_opts.get("occlusion_threshold", 0.35))
        self.occlusion_penalty = float(self.runtime_opts.get("occlusion_penalty", 0.6))
        self.depth_occlusion_margin = float(self.runtime_opts.get("depth_occlusion_margin", 0.05))

    def process_image(
        self,
        image_path: str,
        export_csv: bool = False,
        output_path: Optional[str] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        """Run the full analysis pipeline on a still image."""

        logger.info("Processing image %s", image_path)
        self._notify(progress_cb, "loading_image")
        image = load_image(image_path)

        self._notify(progress_cb, "running_detection")
        target_classes = list({*self.food_classes, *self.reference_classes.keys()})
        detections = self.detector.detect(image, class_filter=target_classes)
        if not detections:
            raise RuntimeError("No detections found in the image")

        reference_dets = [
            (det.label, (det.bbox["x_min"], det.bbox["y_min"], det.bbox["x_max"], det.bbox["y_max"]), det.confidence)
            for det in detections
            if det.label in self.reference_classes
        ]
        food_dets = [det for det in detections if det.label in self.food_classes]

        if not food_dets:
            raise RuntimeError("No food items detected in the image")

        depth_map = None
        self._notify(progress_cb, "estimating_depth")
        if self.depth_estimator is not None:
            depth_map = self.depth_estimator.infer(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        detection_results: List[DetectionResult] = []
        totals = {
            "calories": 0.0,
            "protein_g": 0.0,
            "carbs_g": 0.0,
            "fat_g": 0.0,
            "fiber_g": 0.0,
            "sugar_g": 0.0,
            "sodium_mg": 0.0,
        }

        self._notify(progress_cb, "estimating_portions")
        for det in food_dets:
            bbox = (det.bbox["x_min"], det.bbox["y_min"], det.bbox["x_max"], det.bbox["y_max"])
            portion_estimate = self.portion_estimator.estimate(det.label, bbox, reference_dets, depth_map)
            nutrition = self.nutrition_db.normalize_quantity(det.label, portion_estimate.mass_g)
            occlusion_ratio = self._estimate_occlusion(
                det,
                detections,
                image.shape[:2],
                depth_map,
            )
            is_occluded = occlusion_ratio >= self.occlusion_threshold
            reliability = self._compute_reliability(
                det.confidence, portion_estimate.reliability, nutrition, occlusion_ratio
            )
            if nutrition:
                for key in totals:
                    totals[key] += nutrition.get(key, 0.0)
            detection_results.append(
                DetectionResult(
                    detection=det,
                    portion=portion_estimate,
                    nutrition=nutrition,
                    detection_reliability=reliability,
                    occlusion_ratio=occlusion_ratio,
                    is_occluded=is_occluded,
                )
            )

        overall_reliability = (
            sum(result.detection_reliability for result in detection_results) / len(detection_results)
        )

        output = {
            "run_id": str(uuid.uuid4()),
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "image_path": image_path,
            "detections": [
                {
                    "id": str(uuid.uuid4()),
                    "label": result.detection.label,
                    "confidence": result.detection.confidence,
                    "bbox": result.detection.bbox,
                    "portion_mass_g": result.portion.mass_g,
                    "portion_uncertainty_pct": result.portion.uncertainty_pct,
                    "reference_object": result.portion.reference_object,
                    "scale_cm_per_px": result.portion.scale_cm_per_px,
                    "nutrition": result.nutrition,
                    "detection_reliability": result.detection_reliability,
                    "occlusion_ratio": result.occlusion_ratio,
                    "occluded": result.is_occluded,
                }
                for result in detection_results
            ],
            "totals": totals,
            "overall_reliability": overall_reliability,
        }

        if export_csv and output_path:
            self._export_csv(output, Path(output_path))

        if output_path:
            self._write_json(output, Path(output_path))

        self._notify(progress_cb, "completed")

        return output

    def _compute_reliability(
        self,
        confidence: float,
        portion_reliability: float,
        nutrition: Optional[Dict[str, float]],
        occlusion_ratio: float,
    ) -> float:
        """Combine different quality indicators into a single reliability value."""

        nutrition_factor = 1.0 if nutrition else 0.6
        reliability = (confidence + portion_reliability + nutrition_factor) / 3.0
        occlusion_factor = max(0.0, 1.0 - self.occlusion_penalty * occlusion_ratio)
        return min(0.99, reliability * occlusion_factor)

    def _write_json(self, payload: Dict, output_path: Path) -> None:
        logger.info("Writing JSON output to %s", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _export_csv(self, payload: Dict, output_path: Path) -> None:
        logger.info("Exporting CSV summary to %s", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            "label,confidence,portion_mass_g,uncertainty_pct,calories,protein_g,carbs_g,fat_g,fiber_g,sugar_g,sodium_mg,occlusion_ratio,occluded"
        ]
        for det in payload.get("detections", []):
            nutrition = det.get("nutrition") or {}
            rows.append(
                ",".join(
                    [
                        det.get("label", ""),
                        f"{det.get('confidence', 0.0):.3f}",
                        f"{det.get('portion_mass_g', 0.0):.1f}",
                        f"{det.get('portion_uncertainty_pct', 0.0):.1f}",
                        f"{nutrition.get('calories', 0.0):.1f}",
                        f"{nutrition.get('protein_g', 0.0):.1f}",
                        f"{nutrition.get('carbs_g', 0.0):.1f}",
                        f"{nutrition.get('fat_g', 0.0):.1f}",
                        f"{nutrition.get('fiber_g', 0.0):.1f}",
                        f"{nutrition.get('sugar_g', 0.0):.1f}",
                        f"{nutrition.get('sodium_mg', 0.0):.1f}",
                        f"{det.get('occlusion_ratio', 0.0):.2f}",
                        str(det.get('occluded', False)),
                    ]
                )
            )
        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(rows))

    def _notify(self, progress_cb: Optional[Callable[[str], None]], stage: str) -> None:
        if progress_cb:
            try:
                progress_cb(stage)
            except Exception as exc:
                logger.debug("Progress callback raised %s", exc)

    def _estimate_occlusion(
        self,
        target: Detection,
        all_detections: List[Detection],
        image_shape: tuple[int, int],
        depth_map: Optional[np.ndarray],
    ) -> float:
        """Estimate the portion of ``target`` that is occluded by peers or borders."""

        target_box = target.bbox
        target_area = self._box_area(target_box)
        if target_area <= 0:
            return 0.0

        max_overlap = 0.0
        depth_occlusion = 0.0
        target_depth = self._average_depth(depth_map, target_box)
        for other in all_detections:
            if other is target:
                continue
            overlap_area = self._intersection_area(target_box, other.bbox)
            if overlap_area <= 0:
                continue
            max_overlap = max(max_overlap, overlap_area / target_area)
            if depth_map is None or target_depth is None:
                continue
            other_depth = self._average_depth(depth_map, other.bbox)
            if other_depth is None:
                continue
            # A noticeably smaller depth implies that ``other`` sits closer to the
            # camera and therefore likely occludes ``target`` within the overlap.
            if other_depth + self.depth_occlusion_margin < target_depth:
                depth_occlusion = max(depth_occlusion, overlap_area / target_area)

        boundary_ratio = self._boundary_occlusion_ratio(target_box, image_shape)
        return max(max_overlap, boundary_ratio, depth_occlusion)

    @staticmethod
    def _box_area(bbox: Dict[str, int]) -> float:
        width = max(0, bbox["x_max"] - bbox["x_min"])
        height = max(0, bbox["y_max"] - bbox["y_min"])
        return float(width * height)

    @staticmethod
    def _intersection_area(box_a: Dict[str, int], box_b: Dict[str, int]) -> float:
        x_min = max(box_a["x_min"], box_b["x_min"])
        y_min = max(box_a["y_min"], box_b["y_min"])
        x_max = min(box_a["x_max"], box_b["x_max"])
        y_max = min(box_a["y_max"], box_b["y_max"])
        width = max(0, x_max - x_min)
        height = max(0, y_max - y_min)
        return float(width * height)

    @staticmethod
    def _boundary_occlusion_ratio(bbox: Dict[str, int], image_shape: tuple[int, int]) -> float:
        """Return a heuristic occlusion score if the box touches image borders."""

        height, width = image_shape
        margin = 4
        touches_edge = (
            bbox["x_min"] <= margin
            or bbox["y_min"] <= margin
            or bbox["x_max"] >= width - margin
            or bbox["y_max"] >= height - margin
        )
        if not touches_edge:
            return 0.0

        horizontal_clip = 0.0
        vertical_clip = 0.0
        if bbox["x_min"] <= margin:
            horizontal_clip += margin
        if bbox["x_max"] >= width - margin:
            horizontal_clip += margin
        if bbox["y_min"] <= margin:
            vertical_clip += margin
        if bbox["y_max"] >= height - margin:
            vertical_clip += margin
        area = FoodDetectionBot._box_area(bbox)
        if area == 0:
            return 0.0
        estimated_loss = (horizontal_clip * (bbox["y_max"] - bbox["y_min"])) + (
            vertical_clip * (bbox["x_max"] - bbox["x_min"])
        )
        return min(0.9, estimated_loss / max(area, 1.0))

    @staticmethod
    def _average_depth(depth_map: Optional[np.ndarray], bbox: Dict[str, int]) -> Optional[float]:
        """Return the mean depth in ``bbox`` or ``None`` if unavailable."""

        if depth_map is None:
            return None
        x_min, y_min, x_max, y_max = (
            max(0, bbox["x_min"]),
            max(0, bbox["y_min"]),
            max(0, bbox["x_max"]),
            max(0, bbox["y_max"]),
        )
        if x_max <= x_min or y_max <= y_min:
            return None
        region = depth_map[y_min:y_max, x_min:x_max]
        if region.size == 0:
            return None
        # Guard against NaNs from upstream models.
        with np.errstate(invalid="ignore"):
            mean_val = float(np.nanmean(region))
        if np.isnan(mean_val):
            return None
        return mean_val
