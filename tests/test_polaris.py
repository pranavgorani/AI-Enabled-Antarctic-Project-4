"""
Unit Tests for POLARIS Risk Assessment Engine (IMO MSC.1/Circ.1519).
"""

import pytest
from app.risk.polaris_engine import PolarisEngine


def test_polaris_pc1_all_positive():
    """PC1 icebreaker should have positive RIO across thick first year ice regimes."""
    regime = [
        {"ice_type": "thick_first_year_ice_second_stage", "concentration_tenths": 8},
        {"ice_type": "medium_first_year_ice", "concentration_tenths": 2}
    ]
    res = PolarisEngine.calculate_rio(ice_regime=regime, ice_class="PC1")
    assert res["status"] == "Normal Operation"
    assert res["rio"] > 0
    assert res["ice_class"] == "PC1"


def test_polaris_pc7_negative_thick_ice():
    """PC7 low ice class should face negative RIO and Special Consideration in thick ice."""
    regime = [
        {"ice_type": "thick_first_year_ice_second_stage", "concentration_tenths": 7},
        {"ice_type": "medium_first_year_ice", "concentration_tenths": 3}
    ]
    res = PolarisEngine.calculate_rio(ice_regime=regime, ice_class="PC7")
    assert res["rio"] < 0
    assert res["status"] in ["Elevated Operational Risk", "Special Consideration Required"]


def test_polaris_open_water_zero_ice():
    """Open water regime should yield positive RIO (+30)."""
    regime = [
        {"ice_type": "open_water", "concentration_tenths": 10}
    ]
    res = PolarisEngine.calculate_rio(ice_regime=regime, ice_class="PC4")
    assert res["rio"] == 30
    assert res["status"] == "Normal Operation"


def test_polaris_unknown_class_fallback():
    """Unknown ice class defaults safely to PC4."""
    regime = [{"ice_type": "medium_first_year_ice", "concentration_tenths": 5}]
    res = PolarisEngine.calculate_rio(ice_regime=regime, ice_class="UNKNOWN_CLASS")
    assert res["ice_class"] == "PC4"
    assert "rio" in res


def test_polaris_vessel_profile():
    """Validates pre-configured vessel profiles."""
    prof = PolarisEngine.get_vessel_profile("RV Bharati Explorer")
    assert prof["ice_class"] == "PC4"
    assert prof["length_m"] == 125.0
