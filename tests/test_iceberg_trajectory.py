"""Tests for Iceberg Intelligence & Trajectory Predictor."""
import pytest
from app.ml.iceberg_model import iceberg_processor
from app.ml.trajectory_model import trajectory_predictor


def test_iceberg_hazard_classification():
    score, level = iceberg_processor.classify_hazard("Mega Tabular", velocity_knots=1.5, length_m=3500)
    assert score >= 80.0
    assert level == "CRITICAL HAZARD"

    score_low, level_low = iceberg_processor.classify_hazard("Growler / Bergy Bit", velocity_knots=0.2, length_m=50)
    assert score_low < 50.0


def test_trajectory_prediction_physics():
    iceberg = {
        "id": "ICE-TEST",
        "latitude": -64.0,
        "longitude": 10.0,
        "velocity_knots": 1.0,
        "direction_deg": 270.0
    }
    ocean = {"current_speed_knots": 0.8, "current_direction_deg": 270.0}
    wind = {"wind_speed_knots": 25.0, "wind_direction_deg": 240.0}

    traj = trajectory_predictor.predict_trajectory(
        iceberg=iceberg,
        ocean_current=ocean,
        wind=wind,
        sea_ice_concentration=35.0,
        lead_hours_list=[6, 12, 24, 48, 72]
    )

    assert traj["iceberg_id"] == "ICE-TEST"
    assert len(traj["waypoints"]) == 6  # 0h + 5 lead steps
    # Verify expanding uncertainty radius
    radii = [wp["uncertainty_radius_km"] for wp in traj["waypoints"][1:]]
    assert sorted(radii) == radii
    # Verify GeoJSON structure
    assert traj["geojson"]["geometry"]["type"] == "LineString"
    assert len(traj["geojson"]["geometry"]["coordinates"]) == 6
