from nutrition.database import NutritionDatabase


def test_nutrition_database_loads() -> None:
    db = NutritionDatabase()
    assert len(db.records) >= 100
    apple = db.get("apple")
    assert apple is not None
    normalized = db.normalize_quantity("apple", 150)
    assert normalized is not None
    assert normalized["calories"] > 0
    assert normalized["sugar_g"] > 0
