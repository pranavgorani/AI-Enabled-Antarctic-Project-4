"""Tests for Environmental Risk Engine."""
import pytest
from app.risk.risk_engine import risk_engine, RiskWeights


def test_risk_bounds_and_components():
    res = risk_engine.evaluate_point_risk(
        concentration_pct=50.0,
        min_iceberg_dist_km=15.0,
        wind_speed_knots=30.0,
        wave_height_m=2.5
    )

    assert 0.0 <= res["overall_risk"] <= 100.0
    for k, v in res["components"].items():
        assert 0.0 <= v <= 100.0

    # Ensure contribution percentages sum to approximately 100
    contrib_sum = sum(res["contributions_pct"].values())
    assert 99.0 <= contrib_sum <= 101.0
    assert "explanation" in res
    assert "dominant_factor" in res


def test_iceberg_proximity_escalation():
    far_res = risk_engine.evaluate_point_risk(min_iceberg_dist_km=50.0)
    near_res = risk_engine.evaluate_point_risk(min_iceberg_dist_km=3.0)

    assert near_res["components"]["iceberg_risk"] > far_res["components"]["iceberg_risk"]
    assert near_res["overall_risk"] > far_res["overall_risk"]


def test_custom_weights_normalization():
    weights = RiskWeights(
        sea_ice_weight=0.5,
        iceberg_weight=0.5,
        weather_weight=0.0,
        wave_weight=0.0,
        current_weight=0.0
    )
    res = risk_engine.evaluate_point_risk(custom_weights=weights)
    assert res["weights_used"]["sea_ice"] == 0.5
    assert res["weights_used"]["iceberg"] == 0.5
    assert res["weights_used"]["weather"] == 0.0
