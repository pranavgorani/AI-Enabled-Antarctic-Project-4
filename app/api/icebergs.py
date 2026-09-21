"""Iceberg intelligence and trajectory endpoints."""
from fastapi import APIRouter, HTTPException
from app.services.iceberg_service import iceberg_service

router = APIRouter(prefix="/api/icebergs", tags=["Icebergs"])


@router.get("")
def get_icebergs():
    """Lists all active tracked icebergs with observations and hazard rankings."""
    icebergs = iceberg_service.get_all_icebergs()
    return {
        "status": "success",
        "data_mode": "DEMO DATA — NOT FOR REAL-WORLD NAVIGATION",
        "count": len(icebergs),
        "icebergs": icebergs
    }


@router.get("/{iceberg_id}")
def get_iceberg(iceberg_id: str):
    """Retrieves specific iceberg observation details."""
    ib = iceberg_service.get_iceberg_by_id(iceberg_id)
    if not ib:
        raise HTTPException(status_code=404, detail=f"Iceberg '{iceberg_id}' not found")
    return {"status": "success", "iceberg": ib}


@router.get("/{iceberg_id}/trajectory")
def get_iceberg_trajectory(iceberg_id: str):
    """Computes physics-informed 6h-72h drift trajectory and uncertainty corridors."""
    traj = iceberg_service.get_iceberg_trajectory(iceberg_id)
    if not traj:
        raise HTTPException(status_code=404, detail=f"Iceberg '{iceberg_id}' not found")
    return traj
