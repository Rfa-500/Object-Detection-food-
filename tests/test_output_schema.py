import json
from pathlib import Path

import jsonschema


def test_output_schema_validates() -> None:
    schema_path = Path("data/output_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    sample = {
        "run_id": "123",
        "processed_at": "2024-01-01T00:00:00Z",
        "image_path": "sample.jpg",
        "overall_reliability": 0.8,
        "totals": {
            "calories": 500,
            "protein_g": 25,
            "carbs_g": 60,
            "fat_g": 20,
            "fiber_g": 10,
            "sugar_g": 30,
            "sodium_mg": 800
        },
        "detections": [
            {
                "id": "det-1",
                "label": "apple",
                "confidence": 0.9,
                "bbox": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 120},
                "portion_mass_g": 150.0,
                "portion_uncertainty_pct": 20.0,
                "reference_object": "credit_card",
                "scale_cm_per_px": 0.2,
                "nutrition": {
                    "calories": 78,
                    "protein_g": 0.5,
                    "carbs_g": 20,
                    "fat_g": 0.3,
                    "fiber_g": 4.0,
                    "sugar_g": 15,
                    "sodium_mg": 5
                },
                "detection_reliability": 0.85
            }
        ]
    }
    jsonschema.validate(instance=sample, schema=schema)
