"""Tests for What-If Simulation Engine."""
import pytest
from app.simulation.simulator import simulator, SimulationParameters


def test_simulation_execution():
    params = SimulationParameters(
        sea_ice_delta_pct=25.0,
        wind_speed_knots=45.0,
        wave_height_m=5.0,
        current_speed_knots=1.5,
        iceberg_count_multiplier=2.0
    )
    res = simulator.run_simulation(params=params)

    assert "baseline" in res
    assert "perturbed" in res
    assert "deltas" in res
    assert res["deltas"]["risk_score_delta"] > 0
    assert res["deltas"]["fuel_pct_delta"] > 0
    assert "simulated" in res["explanation"].lower()
