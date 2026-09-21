"""
Comprehensive Edge-Case and Stress Testing Suite.
Validates 17 critical boundary conditions and failure modes.
"""

import pytest
import math
from app.geospatial.land_mask import antarctic_land_mask
from app.routing.time_dependent_astar import TimeDependentAStar
from app.risk.polaris_engine import PolarisEngine
from app.fuel.ice_resistance import LindqvistFuelModel
from app.routing.pareto import ParetoFrontier
from app.routing.sensitivity import RouteSensitivityAnalyzer
from app.routing.rtz_export import MaritimeRouteExporter
from app.core.security import RateLimiter


# 1. Vessel initialized on land/ice shelf
def test_edge_case_start_on_land():
    # Interior of East Antarctica (Dome C / South Pole vicinity)
    assert antarctic_land_mask.is_land(-80.0, 10.0) is True
    # Path finder should detect collision
    path = TimeDependentAStar.find_path(
        start=(-80.0, 10.0),
        goal=(-64.0, 10.0),
        departure_hour=0
    )
    assert path["status"] in ["blocked_start_on_land", "blocked_start_or_goal_on_land"]
    assert len(path["waypoints"]) == 0


# 2. Destination point on land
def test_edge_case_destination_on_land():
    path = TimeDependentAStar.find_path(
        start=(-62.0, 10.0),
        goal=(-82.0, 15.0),
        departure_hour=0
    )
    assert path["status"] in ["blocked_goal_on_land", "blocked_start_or_goal_on_land"]
    assert len(path["waypoints"]) == 0


# 3. Start and end points identical
def test_edge_case_identical_start_and_goal():
    path = TimeDependentAStar.find_path(
        start=(-62.0, 10.0),
        goal=(-62.0, 10.0),
        departure_hour=0
    )
    assert path["status"] == "identical_start_and_goal"
    assert len(path["waypoints"]) == 1
    assert path["total_distance_nm"] == 0.0


# 4. Coordinates out of range
def test_edge_case_coordinates_out_of_range():
    # Outside high-latitude Antarctic operational bounds
    path = TimeDependentAStar.find_path(
        start=(10.0, 10.0),  # Northern hemisphere!
        goal=(-64.0, 10.0),
        departure_hour=0
    )
    assert path["status"] == "out_of_antarctic_bounds"


# 5. Zero ice concentration handling in fuel model
def test_edge_case_zero_ice_fuel():
    res = LindqvistFuelModel.estimate_fuel_burn(
        distance_nm=100.0,
        speed_knots=12.0,
        ice_concentration=0.0,
        ice_thickness_m=0.0
    )
    assert res["fuel_burn_tonnes"] > 0
    assert res["ice_resistance_breakdown_kN"]["R_ice_kN"] == 0.0


# 6. 100% solid sea ice handling in fuel model
def test_edge_case_solid_ice_fuel():
    res = LindqvistFuelModel.estimate_fuel_burn(
        distance_nm=50.0,
        speed_knots=12.0,
        ice_concentration=1.0,
        ice_thickness_m=2.5
    )
    assert res["actual_speed_knots"] < 12.0
    assert res["effective_resistance_kN"] > res["open_water_resistance_kN"]


# 7. Extreme hurricane-force katabatic winds (>100 knots)
def test_edge_case_extreme_wind_risk():
    from app.risk.risk_engine import risk_engine
    risk = risk_engine.evaluate_point_risk(concentration_pct=20.0, wind_speed_knots=120.0)
    assert risk["components"]["weather_risk"] >= 70.0
    assert risk["risk_score"] > 25.0


# 8. Extreme sub-zero temperatures (-60°C)
def test_edge_case_extreme_subzero_temp():
    from app.risk.risk_engine import risk_engine
    risk = risk_engine.evaluate_point_risk(concentration_pct=10.0, air_temp_c=-60.0)
    assert risk["components"]["weather_risk"] >= 20.0
    assert risk["risk_score"] > 15.0


# 9. Missing / empty ice regime in POLARIS
def test_edge_case_empty_polaris_regime():
    res = PolarisEngine.calculate_rio(ice_regime=[], ice_class="PC4")
    assert res["rio"] == 0
    assert res["status"] == "Normal Operation"


# 10. Single-point waypoint trajectory export to RTZ/GPX
def test_edge_case_single_waypoint_rtz():
    single_wp = [(-64.0, 10.0)]
    rtz_xml = MaritimeRouteExporter.export_rtz("SoloWP", single_wp)
    assert "<waypoint" in rtz_xml
    assert "<route" in rtz_xml
    gpx_xml = MaritimeRouteExporter.export_gpx("SoloWP", single_wp)
    assert "<rtept" in gpx_xml


# 11. Large waypoint count route smoothing
def test_edge_case_route_smoothing_stress():
    pts = [(-60.0 - (i * 0.1), 10.0 + math.sin(i) * 0.2) for i in range(100)]
    smoothed = TimeDependentAStar.smooth_path(pts)
    assert len(smoothed) <= len(pts)
    assert smoothed[0] == pts[0]
    assert smoothed[-1] == pts[-1]


# 12. Zero distance fuel calculation
def test_edge_case_zero_distance_fuel():
    res = LindqvistFuelModel.estimate_fuel_burn(
        distance_nm=0.0,
        speed_knots=12.0,
        ice_concentration=0.5,
        ice_thickness_m=1.0
    )
    assert res["fuel_burn_tonnes"] == 0.0
    assert res["transit_time_hours"] == 0.0


# 13. Negative ice thickness input handling
def test_edge_case_negative_ice_thickness():
    res = LindqvistFuelModel.calculate_lindqvist_ice_resistance(
        ice_thickness_m=-0.5,
        vessel_speed_ms=6.0,
        vessel_beam_m=20.0,
        vessel_length_m=100.0,
        vessel_draft_m=7.0
    )
    assert res["R_ice_kN"] == 0.0


# 14. Completely blocked path when destination surrounded by land
def test_edge_case_blocked_path():
    path = TimeDependentAStar.find_path(
        start=(-62.0, 10.0),
        goal=(-85.0, 0.0),  # Deep inland Antarctic plateau
        departure_hour=0
    )
    assert path["status"] != "optimal_found"
    assert path["total_distance_nm"] == 0.0


# 15. Empty route candidate list in Pareto analysis
def test_edge_case_empty_pareto_candidates():
    res = ParetoFrontier.compute_pareto_front([])
    assert res["count_pareto"] == 0
    assert res["recommended_balanced_route"] is None


# 16. Zero weight perturbation in sensitivity analysis
def test_edge_case_zero_perturbation_sensitivity():
    routes = [
        {"route_id": "R1", "risk_score": 30.0, "fuel_tonnes": 50.0, "transit_time_hours": 40.0},
        {"route_id": "R2", "risk_score": 60.0, "fuel_tonnes": 30.0, "transit_time_hours": 35.0}
    ]
    res = RouteSensitivityAnalyzer.evaluate_sensitivity(routes, perturbation_pct=0.0)
    assert res["stability_score"] == 1.0


# 17. Rate limiter denial under spam conditions
def test_edge_case_rate_limiter_stress():
    limiter = RateLimiter(max_requests=5, window_seconds=10)
    client_ip = "192.168.1.100"

    # First 5 should succeed
    for _ in range(5):
        assert limiter.is_allowed(client_ip) is True

    # 6th should be rejected
    assert limiter.is_allowed(client_ip) is False
