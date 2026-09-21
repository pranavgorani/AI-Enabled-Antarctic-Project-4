"""Tests for Sea-Ice Forecasting ML model."""
import pytest
import numpy as np
from app.ml.sea_ice_model import sea_ice_forecaster, SeaIceCategoryConfig


def test_sea_ice_category_classification():
    assert SeaIceCategoryConfig.classify(10.0) == "Open Water"
    assert SeaIceCategoryConfig.classify(35.0) == "Low Ice"
    assert SeaIceCategoryConfig.classify(55.0) == "Moderate Ice"
    assert SeaIceCategoryConfig.classify(85.0) == "High Ice"
    assert SeaIceCategoryConfig.classify(95.0) == "Very High Ice"


def test_sea_ice_forecaster_predictions():
    sample_grid = [
        {"latitude": -62.0, "longitude": 10.0, "concentration_pct": 25.0},
        {"latitude": -66.0, "longitude": 11.0, "concentration_pct": 60.0},
        {"latitude": -70.0, "longitude": 12.0, "concentration_pct": 85.0}
    ]

    for lead_h in [24, 48, 72]:
        result = sea_ice_forecaster.forecast_grid(sample_grid, lead_hours=lead_h)
        assert result["lead_hours"] == lead_h
        assert 0.7 <= result["confidence"] <= 1.0
        assert len(result["forecast"]) == len(sample_grid)
        for item in result["forecast"]:
            assert 0.0 <= item["predicted_concentration_pct"] <= 100.0
            assert item["category"] in ["Open Water", "Low Ice", "Moderate Ice", "High Ice", "Very High Ice"]
