"""Ocean dynamics API endpoints."""
from fastapi import APIRouter, Query
from app.services.ocean_service import ocean_service

router = APIRouter(prefix="/api/ocean", tags=["Ocean"])


@router.get("")
def get_ocean(
    lat: float = Query(default=None),
    lon: float = Query(default=None)
):
    """Returns full ocean current grid or point ocean condition."""
    if lat is not None and lon is not None:
        return ocean_service.get_point_ocean(lat, lon)
    grid = ocean_service.get_ocean_grid()
    return {
        "status": "success",
        "data_mode": "DEMO DATA — NOT FOR REAL-WORLD NAVIGATION",
        "count": len(grid),
        "grid": grid
    }
