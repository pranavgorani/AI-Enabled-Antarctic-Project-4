"""Tests for FastAPI endpoints using TestClient."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_sea_ice_endpoints():
    r_curr = client.get("/api/sea-ice/current")
    assert r_curr.status_code == 200
    assert r_curr.json()["count"] > 0

    r_fc = client.get("/api/sea-ice/forecast?lead_hours=24")
    assert r_fc.status_code == 200
    assert r_fc.json()["lead_hours"] == 24
    assert len(r_fc.json()["forecast"]) > 0


def test_iceberg_endpoints():
    r_all = client.get("/api/icebergs")
    assert r_all.status_code == 200
    bergs = r_all.json()["icebergs"]
    assert len(bergs) > 0

    first_id = bergs[0]["id"]
    r_single = client.get(f"/api/icebergs/{first_id}")
    assert r_single.status_code == 200
    assert r_single.json()["iceberg"]["id"] == first_id

    r_traj = client.get(f"/api/icebergs/{first_id}/trajectory")
    assert r_traj.status_code == 200
    assert "waypoints" in r_traj.json()


def test_weather_and_ocean_endpoints():
    r_wx = client.get("/api/weather")
    assert r_wx.status_code == 200
    assert r_wx.json()["count"] > 0

    r_oc = client.get("/api/ocean")
    assert r_oc.status_code == 200
    assert r_oc.json()["count"] > 0


def test_risk_and_alerts():
    r_risk = client.get("/api/risk")
    assert r_risk.status_code == 200
    assert 0 <= r_risk.json()["overall_risk"] <= 100

    r_alerts = client.get("/api/alerts")
    assert r_alerts.status_code == 200
    assert len(r_alerts.json()["alerts"]) >= 1


def test_routes_endpoints():
    r_opt = client.post("/api/routes/optimize", json={
        "start_lat": -58.5,
        "start_lon": 10.5,
        "dest_lat": -70.767,
        "dest_lon": 11.731
    })
    assert r_opt.status_code == 200
    data = r_opt.json()
    assert "recommended" in data
    assert "fastest" in data

    rec_waypoints = data["recommended"]["waypoints"]
    target_wp = rec_waypoints[len(rec_waypoints) // 2]

    r_replan = client.post("/api/routes/recalculate", json={
        "hazard_iceberg_id": "ICE-042",
        "hazard_predicted_lat": target_wp["latitude"],
        "hazard_predicted_lon": target_wp["longitude"],
        "proximity_threshold_km": 20.0
    })
    assert r_replan.status_code == 200
    assert r_replan.json()["reassessment_required"] is True


def test_simulation_and_ai_endpoints():
    r_sim = client.post("/api/simulation/run", json={
        "sea_ice_delta_pct": 15.0,
        "wind_speed_knots": 35.0,
        "wave_height_m": 4.0,
        "current_speed_knots": 1.2,
        "iceberg_count_multiplier": 1.5
    })
    assert r_sim.status_code == 200
    assert "deltas" in r_sim.json()

    r_query = client.post("/api/ai/query", json={"query": "Why was this route selected?"})
    assert r_query.status_code == 200
    assert "Recommended" in r_query.json()["response"]

    r_briefing = client.post("/api/ai/briefing", json={
        "mission_name": "Antarctic Mission Alpha",
        "vessel_name": "RV Explorer",
        "destination": "Maitri",
        "forecast_window": "72 Hours"
    })
    assert r_briefing.status_code == 200
    assert "ANTARCTIC NAVIGATION BRIEFING" in r_briefing.json()["briefing"]
