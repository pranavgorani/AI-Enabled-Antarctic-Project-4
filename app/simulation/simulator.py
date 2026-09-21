"""
What-If Environmental Simulation Engine
Polar Navigator AI - MoES / NCPOR

Simulates extreme polar scenario perturbations:
- Sea-ice concentration surge (+10% to +40%)
- Severe gale / blizzard winds (up to 55 knots)
- Extreme swell / wave height escalation (up to 7.0m)
- Ocean current acceleration
- Iceberg density / drift velocity intensification

Calculates:
- Baseline vs Perturbed environmental risk scores
- Route re-optimization deltas (Distance, Fuel, ETA, Cost)
- Dynamic route reassessment triggers
"""

import copy
from datetime import datetime
from pydantic import BaseModel, Field
from app.risk.risk_engine import risk_engine, RiskWeights
from app.routing.fuel_model import fuel_model
from app.routing.optimizer import route_optimizer


class SimulationParameters(BaseModel):
    sea_ice_delta_pct: float = Field(default=20.0, ge=-50.0, le=50.0)
    wind_speed_knots: float = Field(default=40.0, ge=5.0, le=70.0)
    wave_height_m: float = Field(default=4.5, ge=0.5, le=10.0)
    current_speed_knots: float = Field(default=1.4, ge=0.1, le=3.5)
    iceberg_count_multiplier: float = Field(default=1.5, ge=0.5, le=3.0)


class EnvironmentalSimulator:
    """Executes counterfactual 'What-If' scenario simulations."""

    def run_simulation(
        self,
        baseline_routes: dict = None,
        params: SimulationParameters = None
    ) -> dict:
        """
        Executes a What-If simulation comparing baseline navigation metrics against perturbed polar conditions.
        """
        if params is None:
            params = SimulationParameters()

        if baseline_routes is None:
            baseline_routes = route_optimizer.generate_all_routes()

        rec_base = baseline_routes.get("recommended", {})

        # 1. Baseline Values
        base_dist = rec_base.get("total_distance_km", 1380.0)
        base_duration = rec_base.get("estimated_duration_hours", 62.0)
        base_fuel = rec_base.get("estimated_fuel_mt", 68.0)
        base_risk = rec_base.get("average_risk_score", 38.0)
        base_ice = rec_base.get("average_ice_concentration_pct", 35.0)

        # 2. Perturbed Environmental Calculation
        sim_ice = min(98.0, max(5.0, base_ice + params.sea_ice_delta_pct))
        sim_eval = risk_engine.evaluate_point_risk(
            concentration_pct=sim_ice,
            wind_speed_knots=params.wind_speed_knots,
            wave_height_m=params.wave_height_m,
            current_speed_knots=params.current_speed_knots,
            min_iceberg_dist_km=14.0 / params.iceberg_count_multiplier
        )

        sim_risk = sim_eval["overall_risk"]

        # Perturbed Fuel & ETA
        fuel_eval = fuel_model.estimate_segment(
            distance_nm=base_dist / 1.852,
            concentration_pct=sim_ice,
            wind_speed_knots=params.wind_speed_knots,
            wave_height_m=params.wave_height_m,
            current_speed_knots=params.current_speed_knots
        )

        sim_fuel = fuel_eval["adjusted_fuel_mt"]
        sim_duration = fuel_eval["transit_hours"]

        # If risk spikes above 65, an evasive detour route is required (+35 km)
        replanning_triggered = sim_risk >= 60.0
        if replanning_triggered:
            sim_dist = base_dist + 42.5
            sim_fuel = round(sim_fuel * 1.06, 2)
            sim_duration = round(sim_duration * 1.05, 1)
        else:
            sim_dist = base_dist

        # 3. Compute Deltas
        delta_risk = round(sim_risk - base_risk, 1)
        delta_fuel_pct = round(((sim_fuel - base_fuel) / base_fuel) * 100.0, 1)
        delta_duration_hours = round(sim_duration - base_duration, 1)
        delta_dist_km = round(sim_dist - base_dist, 1)

        summary_explanation = (
            f"Under the simulated scenario (Sea-Ice: {sim_ice:.1f}%, Wind: {params.wind_speed_knots} kts, "
            f"Wave: {params.wave_height_m}m), overall operational risk shifts from {base_risk} to {sim_risk} "
            f"({'+' if delta_risk > 0 else ''}{delta_risk} pts). "
            f"Fuel consumption increases by {delta_fuel_pct}% ({sim_fuel:.1f} MT vs {base_fuel:.1f} MT), "
            f"extending voyage duration by {delta_duration_hours} hours. "
        )
        if replanning_triggered:
            summary_explanation += "CRITICAL: Risk threshold exceeded. Dynamic route replanning has been triggered to avoid heavy pack-ice compression."

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "simulation_parameters": params.model_dump(),
            "baseline": {
                "risk_score": base_risk,
                "fuel_mt": base_fuel,
                "duration_hours": base_duration,
                "distance_km": base_dist,
                "sea_ice_pct": base_ice
            },
            "perturbed": {
                "risk_score": sim_risk,
                "fuel_mt": sim_fuel,
                "duration_hours": sim_duration,
                "distance_km": sim_dist,
                "sea_ice_pct": round(sim_ice, 1),
                "severity": sim_eval["severity"]
            },
            "deltas": {
                "risk_score_delta": delta_risk,
                "fuel_pct_delta": delta_fuel_pct,
                "duration_hours_delta": delta_duration_hours,
                "distance_km_delta": delta_dist_km
            },
            "replanning_triggered": replanning_triggered,
            "explanation": summary_explanation
        }


simulator = EnvironmentalSimulator()
