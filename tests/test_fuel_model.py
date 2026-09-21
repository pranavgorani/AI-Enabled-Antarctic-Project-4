"""Tests for Fuel Model and Transit Duration."""
import pytest
from app.routing.fuel_model import fuel_model


def test_ice_factor_scaling():
    # Open water factor must be 1.0
    assert fuel_model.calculate_ice_factor(10.0) == 1.0
    # Moderate ice
    factor_mod = fuel_model.calculate_ice_factor(45.0)
    assert factor_mod > 1.0
    # Heavy pack ice
    factor_heavy = fuel_model.calculate_ice_factor(85.0)
    assert factor_heavy > factor_mod


def test_weather_and_current_factors():
    calm = fuel_model.calculate_weather_factor(10.0, 1.0)
    storm = fuel_model.calculate_weather_factor(45.0, 5.0)
    assert storm > calm

    fav_current = fuel_model.calculate_current_factor(1.5, 180.0, 180.0)
    opp_current = fuel_model.calculate_current_factor(1.5, 0.0, 180.0)
    assert fav_current < opp_current


def test_estimate_segment():
    res = fuel_model.estimate_segment(
        distance_nm=100.0,
        cruising_speed_knots=12.0,
        concentration_pct=40.0
    )
    assert res["distance_nm"] == 100.0
    assert res["distance_km"] == 185.2
    assert res["adjusted_fuel_mt"] >= res["base_fuel_mt"]
    assert res["transit_hours"] > 0.0
