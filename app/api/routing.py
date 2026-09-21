"""Route optimization and dynamic replanning endpoints."""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.routing.optimizer import route_optimizer
from app.services.iceberg_service import iceberg_service

router = APIRouter(prefix="/api/routes", tags=["Routing"])


class RouteOptimizationRequest(BaseModel):
    start_lat: float = Field(default=-58.5)
    start_lon: float = Field(default=10.5)
    dest_lat: float = Field(default=-70.767)
    dest_lon: float = Field(default=11.731)
    vessel_speed_knots: float = Field(default=12.0)


class RecalculateRequest(BaseModel):
    hazard_iceberg_id: str = "ICE-042"
    hazard_predicted_lat: float = -64.6
    hazard_predicted_lon: float = 10.7
    proximity_threshold_km: float = 15.0


@router.post("/optimize")
def optimize_routes(req: RouteOptimizationRequest):
    """
    Computes 4 evaluated navigation routes:
    1. Recommended Lower-Risk Route
    2. Fastest Route
    3. Fuel-Efficient Route
    4. Alternative Route
    """
    return route_optimizer.generate_all_routes(
        start_lat=req.start_lat,
        start_lon=req.start_lon,
        dest_lat=req.dest_lat,
        dest_lon=req.dest_lon,
        vessel_speed_knots=req.vessel_speed_knots
    )


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
    return result
