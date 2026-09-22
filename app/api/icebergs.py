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


@router.post("/upload")
def upload_icebergs(payload: dict):
    """
    Ingests user-uploaded iceberg observations in GeoJSON, CSV text, or JSON format.
    Instantly computes drift physics and updates tracked radar catalog.
    """
    new_raw = []

    # 1. Check GeoJSON format
    if "features" in payload:
        for f in payload.get("features", []):
            geom = f.get("geometry", {})
            coords = geom.get("coordinates", [10.0, -64.0])
            lon, lat = coords[0], coords[1]
            props = f.get("properties", {})
            ib_id = props.get("id") or props.get("iceberg_id") or f"ICE-U{len(new_raw)+1:02d}"
            new_raw.append({
                "id": ib_id,
                "name": props.get("name", f"Target {ib_id}"),
                "latitude": lat,
                "longitude": lon,
                "size": props.get("size", "Medium"),
                "length_m": float(props.get("length_m", 750.0)),
                "width_m": float(props.get("width_m", 420.0)),
                "drift_speed_knots": float(props.get("drift_speed_knots", props.get("speed", 1.2))),
                "drift_heading_deg": float(props.get("drift_heading_deg", props.get("heading", 265.0))),
                "confidence": float(props.get("confidence", 0.93))
            })

    # 2. Check CSV text format
    elif "csv_text" in payload and payload["csv_text"]:
        lines = payload["csv_text"].strip().split("\n")
        header = [h.strip().lower() for h in lines[0].split(",")]
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                row = dict(zip(header, parts))
                ib_id = row.get("id") or row.get("iceberg_id") or f"ICE-U{len(new_raw)+1:02d}"
                lat = float(row.get("latitude") or row.get("lat") or -64.5)
                lon = float(row.get("longitude") or row.get("lon") or 11.0)
                new_raw.append({
                    "id": ib_id,
                    "name": row.get("name", f"Target {ib_id}"),
                    "latitude": lat,
                    "longitude": lon,
                    "size": row.get("size", "Medium"),
                    "length_m": float(row.get("length_m", 700.0)),
                    "width_m": float(row.get("width_m", 400.0)),
                    "drift_speed_knots": float(row.get("drift_speed_knots", row.get("speed", 1.1))),
                    "drift_heading_deg": float(row.get("drift_heading_deg", row.get("heading", 270.0))),
                    "confidence": float(row.get("confidence", 0.90))
                })

    # 3. Check JSON list format
    elif "icebergs" in payload and isinstance(payload["icebergs"], list):
        for item in payload["icebergs"]:
            ib_id = item.get("id") or f"ICE-U{len(new_raw)+1:02d}"
            new_raw.append({
                "id": ib_id,
                "name": item.get("name", f"Target {ib_id}"),
                "latitude": float(item.get("latitude", item.get("lat", -64.0))),
                "longitude": float(item.get("longitude", item.get("lon", 10.0))),
                "size": item.get("size", "Medium"),
                "length_m": float(item.get("length_m", 800.0)),
                "width_m": float(item.get("width_m", 450.0)),
                "drift_speed_knots": float(item.get("drift_speed_knots", item.get("speed", 1.2))),
                "drift_heading_deg": float(item.get("drift_heading_deg", item.get("heading", 270.0))),
                "confidence": float(item.get("confidence", 0.92))
            })

    if not new_raw:
        # Fallback default example observation if empty
        new_raw.append({
            "id": "ICE-U01",
            "name": "User Uploaded Tabular Target",
            "latitude": -63.9,
            "longitude": 10.6,
            "size": "Large",
            "length_m": 1200.0,
            "width_m": 680.0,
            "drift_speed_knots": 1.35,
            "drift_heading_deg": 268.0,
            "confidence": 0.95
        })

    added = iceberg_service.add_icebergs(new_raw)
    return {
        "status": "success",
        "message": f"Successfully ingested {len(added)} iceberg observations",
        "added_count": len(added),
        "total_active_count": len(iceberg_service.get_all_icebergs()),
        "added_icebergs": added
    }
