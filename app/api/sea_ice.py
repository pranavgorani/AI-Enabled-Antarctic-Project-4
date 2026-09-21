"""Sea-Ice API endpoints."""
from fastapi import APIRouter, Query
from app.services.satellite_service import satellite_service
from app.ml.sea_ice_model import sea_ice_forecaster

router = APIRouter(prefix="/api/sea-ice", tags=["Sea Ice"])


@router.get("/current")
def get_current_sea_ice():
    """Returns the current observed sea-ice concentration grid."""
    grid = satellite_service.get_sea_ice_observations()
    return {
        "status": "success",
        "data_mode": "DEMO DATA — NOT FOR REAL-WORLD NAVIGATION",
        "count": len(grid),
        "grid": grid
    }


@router.get("/forecast")
def get_sea_ice_forecast(lead_hours: int = Query(default=24, enum=[24, 48, 72])):
    """Returns AI/ML sea-ice concentration forecast for 24h, 48h, or 72h lead times."""
    grid = satellite_service.get_sea_ice_observations()
    forecast = sea_ice_forecaster.forecast_grid(grid, lead_hours=lead_hours)
    forecast["data_mode"] = "DEMO DATA — NOT FOR REAL-WORLD NAVIGATION"
    return forecast
