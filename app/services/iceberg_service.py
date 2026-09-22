"""
Iceberg Tracking Service
Polar Navigator AI - MoES / NCPOR

Manages:
- Iceberg observation database
- Drift trajectory predictions across multiple lead-time horizons (6h - 72h)
- Uncertainty corridor generation
- Collision risk checks against active vessel corridors
"""

from datetime import datetime
from app.data.demo_generator import demo_generator
from app.ml.iceberg_model import iceberg_processor
from app.ml.trajectory_model import trajectory_predictor
from app.services.ocean_service import ocean_service
from app.services.weather_service import weather_service


class IcebergService:
    """Provides iceberg tracking and trajectory prediction capabilities."""

    def __init__(self):
        self._icebergs = None

    def get_all_icebergs(self, force_refresh: bool = False) -> list[dict]:
        """Returns enriched iceberg observations."""
        if self._icebergs is None or force_refresh:
            raw = demo_generator.generate_icebergs()
            self._icebergs = iceberg_processor.process_observations(raw)
        return self._icebergs

    def add_icebergs(self, new_bergs: list[dict]) -> list[dict]:
        """Appends new or uploaded iceberg observations."""
        current = list(self.get_all_icebergs())
        processed = iceberg_processor.process_observations(new_bergs)
        # Avoid duplicate IDs
        existing_ids = {ib["id"].upper() for ib in current}
        added = []
        for p in processed:
            if p["id"].upper() not in existing_ids:
                current.insert(0, p)
                existing_ids.add(p["id"].upper())
                added.append(p)
            else:
                # Update existing
                for i, existing in enumerate(current):
                    if existing["id"].upper() == p["id"].upper():
                        current[i] = p
                        added.append(p)
        self._icebergs = current
        return added

    def get_iceberg_by_id(self, iceberg_id: str) -> dict | None:
        """Finds iceberg by identifier (e.g. ICE-042)."""
        all_bergs = self.get_all_icebergs()
        for ib in all_bergs:
            if ib["id"].upper() == iceberg_id.upper():
                return ib
        return None

    def get_iceberg_trajectory(self, iceberg_id: str) -> dict | None:
        """Calculates 6h - 72h predicted trajectory and uncertainty corridors for an iceberg."""
        ib = self.get_iceberg_by_id(iceberg_id)
        if not ib:
            return None

        # Sample local ocean and wind conditions
        ocean = ocean_service.get_point_ocean(ib["latitude"], ib["longitude"])
        weather = weather_service.get_point_weather(ib["latitude"], ib["longitude"])

        traj = trajectory_predictor.predict_trajectory(
            iceberg=ib,
            ocean_current=ocean,
            wind=weather,
            sea_ice_concentration=40.0,
            lead_hours_list=[6, 12, 24, 48, 72]
        )
        return traj

    def get_all_trajectories(self) -> list[dict]:
        """Computes trajectories for all active icebergs."""
        results = []
        for ib in self.get_all_icebergs():
            traj = self.get_iceberg_trajectory(ib["id"])
            if traj:
                results.append(traj)
        return results


iceberg_service = IcebergService()
