"""Tests for Multi-Objective Route Optimizer and Replanner."""
import pytest
from app.routing.optimizer import route_optimizer
from app.routing.astar import AStarRouter
from app.routing.dijkstra import DijkstraRouter


def test_astar_and_dijkstra_basic_graph():
    graph = {
        "A": [("B", 10.0), ("C", 15.0)],
        "B": [("D", 12.0)],
        "C": [("D", 10.0)],
        "D": []
    }
    coords = {
        "A": (-60.0, 10.0),
        "B": (-62.0, 10.0),
        "C": (-61.0, 12.0),
        "D": (-64.0, 11.0)
    }

    path_astar = AStarRouter.find_path("A", "D", graph, coords)
    path_dijkstra = DijkstraRouter.find_path("A", "D", graph)

    assert path_astar == ["A", "B", "D"]
    assert path_dijkstra == ["A", "B", "D"]


def test_generate_all_routes():
    routes = route_optimizer.generate_all_routes()

    for key in ["recommended", "fastest", "fuel_efficient", "alternative"]:
        assert key in routes
        r = routes[key]
        assert r["total_distance_km"] > 500.0
        assert r["estimated_duration_hours"] > 20.0
        assert r["estimated_fuel_mt"] > 10.0
        assert 0.0 <= r["average_risk_score"] <= 100.0
        assert len(r["waypoints"]) >= 2
        assert r["geojson"]["geometry"]["type"] == "LineString"

    assert len(routes["comparison_summary"]) == 4


def test_dynamic_replanning_trigger():
    routes = route_optimizer.generate_all_routes()
    rec = routes["recommended"]
    wp_mid = rec["waypoints"][len(rec["waypoints"]) // 2]

    # Iceberg placed right on waypoint
    hazard_ib = {"id": "ICE-HAZARD", "name": "Test Hazard"}
    replan = route_optimizer.check_dynamic_replanning(
        current_route=rec,
        hazard_iceberg=hazard_ib,
        hazard_predicted_lat=wp_mid["latitude"],
        hazard_predicted_lon=wp_mid["longitude"],
        proximity_threshold_km=15.0
    )

    assert replan["reassessment_required"] is True
    assert "ROUTE REASSESSMENT REQUIRED" in replan["status_banner"]
    assert replan["alternative_route"]["risk_score"] < replan["previous_route"]["risk_score"]
