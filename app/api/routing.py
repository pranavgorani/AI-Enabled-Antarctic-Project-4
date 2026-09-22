"""
Route optimization and dynamic replanning endpoints.
Vercel-compatible with both structured nested requests and flat parameters.
"""

import os
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.routing.optimizer import route_optimizer
from app.services.iceberg_service import iceberg_service

router = APIRouter(prefix="/api/routes", tags=["Routing"])


class Coordinates(BaseModel):
    lat: float
    lon: float


class VesselInfo(BaseModel):
    ice_class: str = "PC4"
    speed: float = 12.0


class RouteOptimizationRequest(BaseModel):
    start: Optional[Coordinates] = None
    destination: Optional[Coordinates] = None
    vessel: Optional[VesselInfo] = None
    # Backward compatible flat parameters
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None
    dest_lat: Optional[float] = None
    dest_lon: Optional[float] = None
    vessel_speed_knots: Optional[float] = None


class RecalculateRequest(BaseModel):
    hazard_iceberg_id: str = "ICE-042"
    hazard_predicted_lat: float = -64.6
    hazard_predicted_lon: float = 10.7
    proximity_threshold_km: float = 15.0


@router.post("/optimize")
def optimize_routes(req: RouteOptimizationRequest = None):
    """
    Computes 4 evaluated navigation routes (Recommended Lower-Risk, Fastest, Fuel-Efficient, Alternative).
    Supports both nested (start: {lat, lon}) and flat (start_lat) parameters.
    """
    if req is None:
        req = RouteOptimizationRequest()

    s_lat = req.start.lat if req.start else (req.start_lat if req.start_lat is not None else -58.5)
    s_lon = req.start.lon if req.start else (req.start_lon if req.start_lon is not None else 10.5)
    d_lat = req.destination.lat if req.destination else (req.dest_lat if req.dest_lat is not None else -70.767)
    d_lon = req.destination.lon if req.destination else (req.dest_lon if req.dest_lon is not None else 11.731)
    spd = req.vessel.speed if req.vessel else (req.vessel_speed_knots if req.vessel_speed_knots is not None else 12.0)

    routes = route_optimizer.generate_all_routes(
        start_lat=s_lat,
        start_lon=s_lon,
        dest_lat=d_lat,
        dest_lon=d_lon,
        vessel_speed_knots=spd
    )

    rec = routes["recommended"]
    data_mode = os.getenv("DATA_MODE", "DEMO").upper()

    return {
        "success": True,
        "data_mode": data_mode,
        "route": rec,
        "routes": routes,
        "recommended": routes.get("recommended"),
        "fastest": routes.get("fastest"),
        "fuel_efficient": routes.get("fuel_efficient"),
        "alternative": routes.get("alternative"),
        "distance": rec.get("total_distance_nm", 0.0),
        "eta": rec.get("estimated_duration_hours", 0.0),
        "fuel": rec.get("estimated_fuel_mt", 0.0),
        "risk": rec.get("average_risk_score", 0.0),
        "confidence": 0.94,
        "disclaimer": "DEMO DATA — NOT FOR REAL-WORLD NAVIGATION"
    }


@router.post("/recalculate")
def recalculate_route(req: RecalculateRequest):
    """
    Triggers dynamic route reassessment if a hazard iceberg approaches within threshold.
    """
    routes = route_optimizer.generate_all_routes()
    rec_route = routes["recommended"]

    ib = iceberg_service.get_iceberg_by_id(req.hazard_iceberg_id) or {
        "id": req.hazard_iceberg_id,
        "name": "Simulated Hazard Candidate"
    }

    result = route_optimizer.check_dynamic_replanning(
        current_route=rec_route,
        hazard_iceberg=ib,
        hazard_predicted_lat=req.hazard_predicted_lat,
        hazard_predicted_lon=req.hazard_predicted_lon,
        proximity_threshold_km=req.proximity_threshold_km
    )

    res = dict(result)
    res["success"] = True
    res["data_mode"] = os.getenv("DATA_MODE", "DEMO").upper()
    return res


@router.get("/{route_id}")
def get_route_by_id(route_id: str):
    """Retrieves specific evaluated route by route identifier or category."""
    routes = route_optimizer.generate_all_routes()
    key = route_id.lower().replace("-", "_")
    if key in routes:
        return {"status": "success", "success": True, "route_id": key, "route": routes[key]}
    for k, r in routes.items():
        if r.get("route_id") == route_id or r.get("route_name", "").lower() == route_id.lower():
            return {"status": "success", "success": True, "route_id": k, "route": r}
    return {"status": "success", "success": True, "route_id": "recommended", "route": routes["recommended"]}
