from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class NutritionRecord:
    food_name: str
    category: str
    serving_size_g: float
    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    sugar_g: float
    sodium_mg: float


class NutritionDatabase:
    def __init__(self, csv_path: Path | str = Path("data/nutrition_database.csv")) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Nutrition database file not found: {self.csv_path}")
        self.records: Dict[str, NutritionRecord] = {}
        self._load()

    def _load(self) -> None:
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_fields = {
                "food_name",
                "category",
                "serving_size_g",
                "calories_kcal",
                "protein_g",
                "carbs_g",
                "fat_g",
                "fiber_g",
                "sugar_g",
                "sodium_mg",
            }
            if set(reader.fieldnames or []) != required_fields:
                missing = required_fields - set(reader.fieldnames or [])
                extra = set(reader.fieldnames or []) - required_fields
                raise ValueError(
                    f"Invalid nutrition database schema. Missing: {missing or 'None'}, Extra: {extra or 'None'}"
                )
            for row in reader:
                try:
                    record = NutritionRecord(
                        food_name=row["food_name"].strip().lower(),
                        category=row["category"].strip(),
                        serving_size_g=float(row["serving_size_g"]),
                        calories_kcal=float(row["calories_kcal"]),
                        protein_g=float(row["protein_g"]),
                        carbs_g=float(row["carbs_g"]),
                        fat_g=float(row["fat_g"]),
                        fiber_g=float(row["fiber_g"]),
                        sugar_g=float(row["sugar_g"]),
                        sodium_mg=float(row["sodium_mg"]),
                    )
                except (TypeError, ValueError) as exc:
                    logger.warning("Skipping invalid nutrition row %s due to %s", row, exc)
                    continue
                self.records[record.food_name] = record
        logger.info("Loaded %d nutrition records", len(self.records))

    def get(self, food_name: str) -> Optional[NutritionRecord]:
        return self.records.get(food_name.lower())

    def normalize_quantity(self, food_name: str, mass_g: float) -> Optional[Dict[str, float]]:
        record = self.get(food_name)
        if not record:
            return None
        factor = mass_g / record.serving_size_g
        return {
            "calories": record.calories_kcal * factor,
            "protein_g": record.protein_g * factor,
            "carbs_g": record.carbs_g * factor,
            "fat_g": record.fat_g * factor,
            "fiber_g": record.fiber_g * factor,
            "sugar_g": record.sugar_g * factor,
            "sodium_mg": record.sodium_mg * factor,
        }
