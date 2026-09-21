"""
Demo Data Generator for Polar Navigator AI
Generates high-fidelity realistic Antarctic environmental datasets:
- Sea-ice concentration grid
- Iceberg observations & metadata
- Iceberg drift vectors
- Weather grid (wind, waves, temperature, pressure)
- Ocean current grid & sea surface temperature (SST)
- Antarctic research stations (Indian & International)
- Research vessel position (RV Explorer)
"""

import os
import json
import math
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

DATA_DISCLAIMER = "DEMO DATA — NOT FOR REAL-WORLD NAVIGATION"

# Antarctic Indian and International Research Stations
ANTARCTIC_STATIONS = [
    {
        "name": "Maitri Research Station",
        "country": "India",
        "operator": "NCPOR / MoES",
        "latitude": -70.767,
        "longitude": 11.731,
        "elevation_m": 117,
        "type": "Permanent",
        "status": "Active"
    },
    {
        "name": "Bharati Research Station",
        "country": "India",
        "operator": "NCPOR / MoES",
        "latitude": -69.407,
        "longitude": 76.195,
        "elevation_m": 35,
        "type": "Permanent",
        "status": "Active"
    },
    {
        "name": "Dakshin Gangotri (Historic)",
        "country": "India",
        "operator": "NCPOR / MoES",
        "latitude": -70.083,
        "longitude": 12.000,
        "elevation_m": 0,
        "type": "Supply Base / Historical",
        "status": "Decommissioned Base"
    },
    {
        "name": "Neumayer Station III",
        "country": "Germany",
        "operator": "AWI",
        "latitude": -70.674,
        "longitude": -8.274,
        "elevation_m": 40,
        "type": "Permanent",
        "status": "Active"
    },
    {
        "name": "Troll Station",
        "country": "Norway",
        "operator": "Norwegian Polar Institute",
        "latitude": -72.011,
        "longitude": 2.535,
        "elevation_m": 1275,
        "type": "Permanent",
        "status": "Active"
    },
    {
        "name": "Halley VI Research Station",
        "country": "United Kingdom",
        "operator": "BAS",
        "latitude": -75.583,
        "longitude": -26.200,
        "elevation_m": 30,
        "type": "Permanent",
        "status": "Active"
    },
    {
        "name": "Syowa Station",
        "country": "Japan",
        "operator": "NIPR",
        "latitude": -69.006,
        "longitude": 39.584,
        "elevation_m": 29,
        "type": "Permanent",
        "status": "Active"
    },
    {
        "name": "Princess Elisabeth",
        "country": "Belgium",
        "operator": "International Polar Foundation",
        "latitude": -71.949,
        "longitude": 23.348,
        "elevation_m": 1400,
        "type": "Zero-Emission Summer",
        "status": "Active"
    }
]

DEFAULT_VESSEL = {
    "name": "RV Explorer",
    "call_sign": "VLEXP-IND",
    "operator": "NCPOR / MoES",
    "ice_class": "PC3",  # Polar Class 3 (Year-round operation in second-year ice)
    "latitude": -58.5,
    "longitude": 10.5,
    "heading_deg": 178.0,
    "speed_knots": 12.5,
    "cruising_speed_knots": 12.0,
    "max_speed_knots": 15.5,
    "fuel_capacity_mt": 2500.0,
    "fuel_remaining_mt": 2180.0,
    "base_fuel_rate_mt_per_nm": 0.035,
    "destination": "Maitri Research Station",
    "dest_lat": -70.767,
    "dest_lon": 11.731
}

DEFAULT_ICEBERGS = [
    {
        "id": "ICE-042",
        "name": "Iceberg ICE-042 (Hazard Candidate)",
        "latitude": -64.4,
        "longitude": 10.9,
        "size_category": "Large Tabular",
        "length_m": 950.0,
        "width_m": 580.0,
        "height_m": 38.0,
        "mass_mt": 2.1e6,
        "velocity_knots": 1.15,
        "direction_deg": 272.0,
        "confidence": 0.94,
        "source": "DEMO DATA",
        "status": "Active Tracking"
    },
    {
        "id": "ICE-A23a",
        "name": "ICE-A23a Giant Fragment",
        "latitude": -62.8,
        "longitude": 6.8,
        "size_category": "Mega Tabular",
        "length_m": 3500.0,
        "width_m": 2200.0,
        "height_m": 55.0,
        "mass_mt": 35.0e6,
        "velocity_knots": 1.4,
        "direction_deg": 285.0,
        "confidence": 0.98,
        "source": "DEMO DATA",
        "status": "Active Tracking"
    },
    {
        "id": "ICE-089",
        "name": "Iceberg ICE-089",
        "latitude": -66.8,
        "longitude": 13.6,
        "size_category": "Medium Pinnacle",
        "length_m": 420.0,
        "width_m": 260.0,
        "height_m": 28.0,
        "mass_mt": 0.8e6,
        "velocity_knots": 0.65,
        "direction_deg": 260.0,
        "confidence": 0.89,
        "source": "DEMO DATA",
        "status": "Active Tracking"
    },
    {
        "id": "ICE-104",
        "name": "Iceberg ICE-104 (Bergy Cluster)",
        "latitude": -67.5,
        "longitude": 9.8,
        "size_category": "Cluster",
        "length_m": 280.0,
        "width_m": 190.0,
        "height_m": 18.0,
        "mass_mt": 0.35e6,
        "velocity_knots": 0.72,
        "direction_deg": 270.0,
        "confidence": 0.91,
        "source": "DEMO DATA",
        "status": "Active Tracking"
    },
    {
        "id": "ICE-055",
        "name": "Iceberg ICE-055",
        "latitude": -65.3,
        "longitude": 12.8,
        "size_category": "Tabular",
        "length_m": 620.0,
        "width_m": 340.0,
        "height_m": 32.0,
        "mass_mt": 1.2e6,
        "velocity_knots": 0.85,
        "direction_deg": 275.0,
        "confidence": 0.93,
        "source": "DEMO DATA",
        "status": "Active Tracking"
    }
]


class DemoDataGenerator:
    """Generates synthetic Antarctic navigation and environmental datasets."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.demo_dir = os.path.join(base_dir, "data", "demo")
        self.geojson_dir = os.path.join(base_dir, "data", "geojson")
        self.csv_dir = os.path.join(base_dir, "data", "csv")
        self._ensure_directories()

    def _ensure_directories(self):
        for d in [self.demo_dir, self.geojson_dir, self.csv_dir]:
            os.makedirs(d, exist_ok=True)

    def generate_sea_ice_grid(self, lat_min=-72.0, lat_max=-56.0, lon_min=0.0, lon_max=22.0, step=0.5) -> list[dict]:
        """
        Generates realistic sea-ice grid points with physical gradients:
        - Open water (0-15%) north of -61°S
        - Marginal ice zone (15-40%) between -61°S and -64.5°S
        - Pack ice (40-75%) between -64.5°S and -68.5°S
        - Fast ice / heavy pack (75-95%) south of -68.5°S near Queen Maud Land
        - Simulated leads and coastal polynyas near 11°E-13°E
        """
        grid = []
        now = datetime.utcnow()
        lats = np.arange(lat_min, lat_max + step, step)
        lons = np.arange(lon_min, lon_max + step, step)

        for lat in lats:
            for lon in lons:
                # Base concentration by latitude
                # normalized distance from -58 to -70
                t = ((-lat) - 58.0) / (70.0 - 58.0)
                t = max(0.0, min(1.0, t))

                # Sigmoidal transition into pack ice
                raw_conc = 100.0 / (1.0 + math.exp(-6.0 * (t - 0.45)))

                # Add longitude dependent variations (polynya near 11.5E, ridge near 6E)
                polynya_effect = -18.0 * math.exp(-((lon - 11.5) ** 2 + (lat + 69.5) ** 2) / 1.5)
                ridge_effect = 12.0 * math.exp(-((lon - 6.0) ** 2 + (lat + 65.0) ** 2) / 3.0)

                # Random micro-scale patchiness
                noise = np.random.normal(0, 3.5)

                conc = max(0.0, min(100.0, raw_conc + polynya_effect + ridge_effect + noise))
                thickness = 0.0 if conc < 10 else round(0.2 + (conc / 100.0) * 1.8, 2)

                # Sea-ice classification
                if conc < 15.0:
                    category = "Open Water"
                elif conc < 40.0:
                    category = "Low Ice"
                elif conc < 70.0:
                    category = "Moderate Ice"
                elif conc < 90.0:
                    category = "High Ice"
                else:
                    category = "Very High Ice"

                grid.append({
                    "latitude": round(float(lat), 3),
                    "longitude": round(float(lon), 3),
                    "concentration_pct": round(float(conc), 1),
                    "thickness_m": thickness,
                    "category": category,
                    "source": DATA_DISCLAIMER,
                    "timestamp": now.isoformat()
                })
        return grid

    def generate_weather_grid(self, lat_min=-72.0, lat_max=-56.0, lon_min=0.0, lon_max=22.0, step=1.0) -> list[dict]:
        """
        Generates realistic Southern Ocean / Antarctic weather grid:
        - Strong westerlies north of -64°S (Roaring Forties / Furious Fifties)
        - Katabatic outflow from Antarctic ice sheet in coastal zone
        - Low pressure systems typical of circumpolar trough
        """
        grid = []
        now = datetime.utcnow()
        for lat in np.arange(lat_min, lat_max + step, step):
            for lon in np.arange(lon_min, lon_max + step, step):
                # Latitude-based temperature gradient (-2°C at -56°S to -22°C at -72°S)
                air_temp = -2.0 - ((-lat - 56.0) * 1.3) + np.random.normal(0, 1.2)

                # Wind speed: strong westerlies at -58°S (~28 kts), katabatic near -70°S (~32 kts)
                if lat > -63.0:
                    # Westerlies
                    wind_speed = 26.0 + 8.0 * math.sin(lon * 0.2) + np.random.normal(0, 3.0)
                    wind_dir = 260.0 + np.random.normal(0, 15.0)
                elif lat > -68.0:
                    # Trough zone
                    wind_speed = 18.0 + np.random.normal(0, 4.0)
                    wind_dir = 210.0 + np.random.normal(0, 20.0)
                else:
                    # Katabatic / Coastal easterlies
                    wind_speed = 28.0 + np.random.normal(0, 5.0)
                    wind_dir = 135.0 + np.random.normal(0, 15.0)

                wind_speed = max(5.0, round(float(wind_speed), 1))
                wind_dir = round(float(wind_dir % 360), 1)

                # Wave height: driven by wind in open water, heavily damped in ice pack
                base_wave = 1.2 + (wind_speed / 10.0) * 0.8
                ice_damping = max(0.15, 1.0 - max(0.0, (-lat - 62.0) / 7.0))
                wave_height = round(float(max(0.4, base_wave * ice_damping)), 1)
                wave_period = round(float(7.0 + wave_height * 1.2), 1)

                pressure = round(float(982.0 + 10.0 * math.cos(lat * 0.3) + np.random.normal(0, 3)), 1)
                visibility = 25.0 if air_temp > -10 else 12.0
                if wind_speed > 35:
                    visibility = 4.0  # Blowing snow

                grid.append({
                    "latitude": round(float(lat), 3),
                    "longitude": round(float(lon), 3),
                    "air_temp_c": round(float(air_temp), 1),
                    "wind_speed_knots": wind_speed,
                    "wind_direction_deg": wind_dir,
                    "wave_height_m": wave_height,
                    "wave_period_s": wave_period,
                    "pressure_hpa": pressure,
                    "visibility_km": visibility,
                    "precipitation_mm": 0.0 if visibility > 10 else 1.5,
                    "source": DATA_DISCLAIMER,
                    "timestamp": now.isoformat()
                })
        return grid

    def generate_ocean_grid(self, lat_min=-72.0, lat_max=-56.0, lon_min=0.0, lon_max=22.0, step=1.0) -> list[dict]:
        """
        Generates Antarctic Circumpolar Current (ACC) eastward drift (north)
        and Antarctic Coastal Counter-Current westward drift (south).
        """
        grid = []
        now = datetime.utcnow()
        for lat in np.arange(lat_min, lat_max + step, step):
            for lon in np.arange(lon_min, lon_max + step, step):
                if lat > -64.0:
                    # ACC: Eastward current (~80-100 deg), speed 0.7 - 1.3 knots
                    speed = 0.9 + 0.3 * math.sin(lat * 0.5) + np.random.normal(0, 0.1)
                    direction = 92.0 + np.random.normal(0, 10)
                    sst = 1.5 - ((-lat - 56.0) * 0.35)
                else:
                    # Coastal Counter Current: Westward (~260-280 deg), speed 0.4 - 0.9 knots
                    speed = 0.65 + np.random.normal(0, 0.1)
                    direction = 275.0 + np.random.normal(0, 12)
                    sst = -1.6 - ((-lat - 64.0) * 0.05)

                speed = max(0.1, round(float(speed), 2))
                direction = round(float(direction % 360), 1)
                sst = round(float(max(-1.9, sst)), 2)

                grid.append({
                    "latitude": round(float(lat), 3),
                    "longitude": round(float(lon), 3),
                    "current_speed_knots": speed,
                    "current_direction_deg": direction,
                    "sea_surface_temp_c": sst,
                    "salinity_psu": round(float(34.2 + np.random.normal(0, 0.15)), 2),
                    "source": DATA_DISCLAIMER,
                    "timestamp": now.isoformat()
                })
        return grid

    def generate_icebergs(self) -> list[dict]:
        """Returns iceberg observations with physical attributes."""
        now = datetime.utcnow()
        icebergs = []
        for ib in DEFAULT_ICEBERGS:
            item = dict(ib)
            item["last_observed"] = now.isoformat()
            item["disclaimer"] = DATA_DISCLAIMER
            icebergs.append(item)
        return icebergs

    def generate_all_and_save(self) -> dict:
        """Generates all datasets and writes them to disk in CSV, JSON, and GeoJSON."""
        sea_ice = self.generate_sea_ice_grid()
        weather = self.generate_weather_grid()
        ocean = self.generate_ocean_grid()
        icebergs = self.generate_icebergs()
        stations = ANTARCTIC_STATIONS
        vessel = DEFAULT_VESSEL

        # 1. Save JSON
        with open(os.path.join(self.demo_dir, "sea_ice_grid.json"), "w") as f:
            json.dump(sea_ice, f, indent=2)
        with open(os.path.join(self.demo_dir, "weather_grid.json"), "w") as f:
            json.dump(weather, f, indent=2)
        with open(os.path.join(self.demo_dir, "ocean_grid.json"), "w") as f:
            json.dump(ocean, f, indent=2)
        with open(os.path.join(self.demo_dir, "icebergs.json"), "w") as f:
            json.dump(icebergs, f, indent=2)
        with open(os.path.join(self.demo_dir, "stations.json"), "w") as f:
            json.dump(stations, f, indent=2)
        with open(os.path.join(self.demo_dir, "vessel.json"), "w") as f:
            json.dump(vessel, f, indent=2)

        # 2. Save CSV
        pd.DataFrame(sea_ice).to_csv(os.path.join(self.csv_dir, "sea_ice_grid.csv"), index=False)
        pd.DataFrame(weather).to_csv(os.path.join(self.csv_dir, "weather_grid.csv"), index=False)
        pd.DataFrame(ocean).to_csv(os.path.join(self.csv_dir, "ocean_grid.csv"), index=False)
        pd.DataFrame(icebergs).to_csv(os.path.join(self.csv_dir, "icebergs.csv"), index=False)
        pd.DataFrame(stations).to_csv(os.path.join(self.csv_dir, "stations.csv"), index=False)

        # 3. Save GeoJSON
        iceberg_features = []
        for ib in icebergs:
            iceberg_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [ib["longitude"], ib["latitude"]]
                },
                "properties": ib
            })
        iceberg_geojson = {"type": "FeatureCollection", "features": iceberg_features}
        with open(os.path.join(self.geojson_dir, "icebergs.geojson"), "w") as f:
            json.dump(iceberg_geojson, f, indent=2)

        station_features = []
        for st in stations:
            station_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [st["longitude"], st["latitude"]]
                },
                "properties": st
            })
        station_geojson = {"type": "FeatureCollection", "features": station_features}
        with open(os.path.join(self.geojson_dir, "stations.geojson"), "w") as f:
            json.dump(station_geojson, f, indent=2)

        return {
            "status": "success",
            "sea_ice_count": len(sea_ice),
            "weather_count": len(weather),
            "ocean_count": len(ocean),
            "iceberg_count": len(icebergs),
            "station_count": len(stations),
            "disclaimer": DATA_DISCLAIMER
        }


# Global instance
demo_generator = DemoDataGenerator()

if __name__ == "__main__":
    res = demo_generator.generate_all_and_save()
    print("Demo Data Generation:", res)
