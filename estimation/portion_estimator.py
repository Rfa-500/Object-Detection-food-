from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.image_utils import bbox_to_dimensions
from utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class PortionEstimate:
    mass_g: float
    uncertainty_pct: float
    reference_object: Optional[str]
    scale_cm_per_px: Optional[float]
    reliability: float


class PortionEstimator:
    def __init__(
        self,
        reference_objects: Dict[str, Dict[str, float]],
        density_overrides: Dict[str, float],
        default_density: float,
        fallback_mass_g: float,
        base_uncertainty_pct: float,
    ) -> None:
        self.reference_objects = reference_objects
        self.density_overrides = density_overrides
        self.default_density = default_density
        self.fallback_mass_g = fallback_mass_g
        self.base_uncertainty_pct = base_uncertainty_pct
        self.special_thickness: Dict[str, float] = {
            "pizza": 3.0,
            "pizza_slice": 3.0,
            "burger": 5.0,
            "sandwich": 5.0,
            "taco": 3.0,
            "burrito": 6.0,
            "sushi": 3.5,
            "steak": 4.0,
            "salmon": 3.5,
            "cake_slice": 4.0,
            "pancake": 1.5,
            "waffle": 2.0,
            "pasta": 2.5,
            "rice": 2.0,
            "soup": 4.0,
            "salad": 6.0,
        }

    def estimate(
        self,
        label: str,
        bbox: Tuple[int, int, int, int],
        reference_detections: List[Tuple[str, Tuple[int, int, int, int], float]],
        depth_map: Optional[np.ndarray] = None,
    ) -> PortionEstimate:
        width_px, height_px = bbox_to_dimensions(bbox)
        if width_px == 0 or height_px == 0:
            logger.warning("Invalid bounding box dimensions for %s: %s", label, bbox)
            return PortionEstimate(
                mass_g=self.fallback_mass_g,
                uncertainty_pct=100.0,
                reference_object=None,
                scale_cm_per_px=None,
                reliability=0.1,
            )

        reference_scale, reference_name, ref_conf = self._compute_scale(reference_detections, width_px, height_px)
        scale_cm_per_px = reference_scale if reference_scale else None

        if reference_scale:
            width_cm = width_px * reference_scale
            height_cm = height_px * reference_scale
        else:
            width_cm, height_cm = self._heuristic_dimensions(width_px, height_px)

        thickness_cm = self._estimate_thickness(label, width_cm, height_cm, depth_map, bbox)
        volume_cm3 = max(width_cm * height_cm * max(thickness_cm, 0.1), 1.0)

        density = self.density_overrides.get(label, self.default_density)
        mass_g = volume_cm3 * density

        uncertainty = self.base_uncertainty_pct
        reliability = 0.5
        if reference_scale:
            uncertainty *= 0.6
            reliability += 0.3 * min(1.0, ref_conf)
        if depth_map is not None:
            uncertainty *= 0.85
            reliability += 0.1

        reliability = min(0.95, reliability)
        uncertainty = max(10.0, min(uncertainty, 80.0))

        return PortionEstimate(
            mass_g=mass_g,
            uncertainty_pct=uncertainty,
            reference_object=reference_name,
            scale_cm_per_px=scale_cm_per_px,
            reliability=reliability,
        )

    def _compute_scale(
        self,
        reference_detections: List[Tuple[str, Tuple[int, int, int, int], float]],
        width_px: int,
        height_px: int,
    ) -> Tuple[Optional[float], Optional[str], float]:
        best_scale: Optional[float] = None
        best_ref: Optional[str] = None
        best_conf = 0.0
        for label, ref_bbox, confidence in reference_detections:
            if label not in self.reference_objects:
                continue
            ref_dims = self.reference_objects[label]
            ref_width_px, ref_height_px = bbox_to_dimensions(ref_bbox)
            scale_candidates = []
            if "width_cm" in ref_dims and ref_width_px > 0:
                scale_candidates.append(ref_dims["width_cm"] / ref_width_px)
            if "height_cm" in ref_dims and ref_height_px > 0:
                scale_candidates.append(ref_dims["height_cm"] / ref_height_px)
            if "diameter_cm" in ref_dims:
                avg_px = (ref_width_px + ref_height_px) / 2 if (ref_width_px + ref_height_px) > 0 else 0
                if avg_px > 0:
                    scale_candidates.append(ref_dims["diameter_cm"] / avg_px)
            if not scale_candidates:
                continue
            scale = float(np.mean(scale_candidates))
            if confidence > best_conf:
                best_scale = scale
                best_ref = label
                best_conf = confidence
        return best_scale, best_ref, best_conf

    def _heuristic_dimensions(self, width_px: int, height_px: int) -> Tuple[float, float]:
        # Assume a default scale of 0.2 cm per pixel for fallback
        scale = 0.2
        return width_px * scale, height_px * scale

    def _estimate_thickness(
        self,
        label: str,
        width_cm: float,
        height_cm: float,
        depth_map: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
    ) -> float:
        base = self.special_thickness.get(label, max(1.5, min(width_cm, height_cm) * 0.35))
        if depth_map is None:
            return base
        x_min, y_min, x_max, y_max = bbox
        depth_region = depth_map[y_min:y_max, x_min:x_max]
        if depth_region.size == 0:
            return base
        depth_norm = depth_region.astype(np.float32)
        depth_norm = depth_norm - depth_norm.min()
        range_val = depth_norm.max() - depth_norm.min()
        if range_val <= 1e-6:
            return base
        variability = range_val / (depth_norm.mean() + 1e-6)
        adjustment = np.clip(variability, 0.2, 2.5)
        return base * (1.0 + 0.25 * adjustment)
