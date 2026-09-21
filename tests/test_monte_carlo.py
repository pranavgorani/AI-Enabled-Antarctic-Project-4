"""
Unit Tests for Monte Carlo Iceberg Trajectory Ensemble and Backtesting.
"""

import pytest
from app.ml.trajectory_model import iceberg_trajectory_model


def test_monte_carlo_ensemble_generation():
    """Generates 120 perturbation simulations with 50% and 90% envelopes."""
    res = iceberg_trajectory_model.run_monte_carlo_ensemble(
        iceberg_id="ICE-042",
        num_simulations=120,
        hours=48
    )

    assert res["iceberg_id"] == "ICE-042"
    assert res["num_simulations"] == 120
    assert len(res["mean_trajectory"]) == 49  # 0 to 48 hours inclusive

    # Check corridors
    assert len(res["corridor_50_pct"]["polygon"]) > 0
    assert len(res["corridor_90_pct"]["polygon"]) > 0

    # Summary statistics
    summary = res["summary"]
    assert summary["mean_drift_km"] > 0
    assert summary["p90_dispersion_km"] >= summary["p50_dispersion_km"]


def test_iceberg_backtest():
    """Validates historical backtesting against simulated actual track."""
    backtest = iceberg_trajectory_model.backtest_trajectory("ICE-042", hours=48)
    assert "mean_absolute_displacement_error_km" in backtest
    assert "root_mean_squared_error_km" in backtest
    assert backtest["hours_evaluated"] == 48
    assert backtest["root_mean_squared_error_km"] >= backtest["mean_absolute_displacement_error_km"]


def test_monte_carlo_empty_iceberg_fallback():
    """Handles non-existent iceberg ID gracefully without crashing."""
    res = iceberg_trajectory_model.run_monte_carlo_ensemble(
        iceberg_id="NON_EXISTENT_ICEBERG_999",
        num_simulations=10,
        hours=24
    )
    assert res["iceberg_id"] == "NON_EXISTENT_ICEBERG_999"
    assert len(res["mean_trajectory"]) > 0
