import numpy as np

from estimation.portion_estimator import PortionEstimator


REFERENCE_OBJECTS = {
    "credit_card": {"width_cm": 8.6, "height_cm": 5.4}
}


def test_portion_estimator_uses_reference_scale() -> None:
    estimator = PortionEstimator(
        reference_objects=REFERENCE_OBJECTS,
        density_overrides={"sandwich": 0.7},
        default_density=0.65,
        fallback_mass_g=150.0,
        base_uncertainty_pct=25.0,
    )
    bbox = (0, 0, 200, 120)
    references = [("credit_card", (0, 0, 86, 54), 0.9)]
    depth_map = np.ones((200, 200), dtype=np.float32)
    estimate = estimator.estimate("sandwich", bbox, references, depth_map)
    assert estimate.reference_object == "credit_card"
    assert estimate.mass_g > 0
    assert 0 < estimate.uncertainty_pct <= 80
