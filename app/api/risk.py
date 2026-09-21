"""Environmental risk evaluation endpoints."""
from fastapi import APIRouter
from pydantic import BaseModel
from app.risk.risk_engine import risk_engine, RiskWeights

router = APIRouter(prefix="/api/risk", tags=["Risk Analysis"])


class RiskEvaluationRequest(BaseModel):
    concentration_pct: float = 40.0
    ice_thickness_m: float = 0.6
    vessel_ice_class: str = "PC3"
    min_iceberg_dist_km: float = 20.0
    wind_speed_knots: float = 25.0
    air_temp_c: float = -6.0
    visibility_km: float = 18.0
    wave_height_m: float = 2.4
    current_speed_knots: float = 0.8
    vessel_speed_knots: float = 12.0
    weights: RiskWeights = None


@router.get("")
def get_default_risk():
    """Evaluates default operational point risk."""
    return risk_engine.evaluate_point_risk()


@router.post("/evaluate")
def evaluate_custom_risk(req: RiskEvaluationRequest):
    """Calculates custom point risk score and Explainable AI breakdown."""
    return risk_engine.evaluate_point_risk(
        concentration_pct=req.concentration_pct,
        ice_thickness_m=req.ice_thickness_m,
        vessel_ice_class=req.vessel_ice_class,
        min_iceberg_dist_km=req.min_iceberg_dist_km,
        wind_speed_knots=req.wind_speed_knots,
        air_temp_c=req.air_temp_c,
        visibility_km=req.visibility_km,
        wave_height_m=req.wave_height_m,
        current_speed_knots=req.current_speed_knots,
        vessel_speed_knots=req.vessel_speed_knots,
        custom_weights=req.weights
    )
