"""Meteorological data API endpoints."""
from fastapi import APIRouter, Query
from app.services.weather_service import weather_service

router = APIRouter(prefix="/api/weather", tags=["Weather"])


@router.get("")
def get_weather(
    lat: float = Query(default=None),
    lon: float = Query(default=None)
):
    """Returns full weather grid or point weather observation."""
    if lat is not None and lon is not None:
        return weather_service.get_point_weather(lat, lon)
    grid = weather_service.get_weather_grid()
    return {
        "status": "success",
        "data_mode": "DEMO DATA — NOT FOR REAL-WORLD NAVIGATION",
        "count": len(grid),
        "grid": grid
    }
