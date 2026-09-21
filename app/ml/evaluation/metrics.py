"""
Scientific Evaluation Metrics for Polar Sea-Ice Forecasting
Calculates:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Integrated Ice Edge Error (IIEE) in km² (Goessling et al., 2016)
"""

import numpy as np
import math


class CryosphericMetrics:
    """Calculates scientifically rigorous sea-ice evaluation metrics."""

    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error in concentration percentage."""
        return round(float(np.mean(np.abs(y_true - y_pred))), 3)

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error in concentration percentage."""
        return round(float(math.sqrt(np.mean((y_true - y_pred) ** 2))), 3)

    @staticmethod
    def calculate_iiee(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold_pct: float = 15.0,
        grid_cell_area_km2: float = 625.0
    ) -> dict:
        """
        Calculates Integrated Ice Edge Error (IIEE).
        IIEE = Area of Overestimation (A+) + Area of Underestimation (A-)
        Standard metric published in Goessling et al. (2016) for ice edge displacement.
        """
        obs_ice = y_true >= threshold_pct
        pred_ice = y_pred >= threshold_pct

        # Overestimation: model has ice, observation does not
        overestimation_cells = int(np.sum(pred_ice & ~obs_ice))
        # Underestimation: observation has ice, model does not
        underestimation_cells = int(np.sum(~pred_ice & obs_ice))

        a_plus = round(overestimation_cells * grid_cell_area_km2, 1)
        a_minus = round(underestimation_cells * grid_cell_area_km2, 1)
        total_iiee = round(a_plus + a_minus, 1)

        return {
            "iiee_total_km2": total_iiee,
            "overestimation_area_km2": a_plus,
            "underestimation_area_km2": a_minus,
            "threshold_pct": threshold_pct,
            "cells_misaligned": overestimation_cells + underestimation_cells
        }

    @classmethod
    def evaluate_model(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "ML-Model",
        lead_hours: int = 24
    ) -> dict:
        """Computes comprehensive metric report."""
        mae = cls.calculate_mae(y_true, y_pred)
        rmse = cls.calculate_rmse(y_true, y_pred)
        iiee = cls.calculate_iiee(y_true, y_pred)

        return {
            "model": model_name,
            "lead_hours": lead_hours,
            "mae": mae,
            "rmse": rmse,
            "iiee_km2": iiee["iiee_total_km2"],
            "iiee_details": iiee
        }
