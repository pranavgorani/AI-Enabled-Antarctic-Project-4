"""
Iceberg Intelligence and Observation Processing Module
Polar Navigator AI - MoES / NCPOR

Supports:
- CSV, JSON, and GeoJSON ingestion & export
- Physical classification (Pinnacle, Tabular, Bergy bit, Mega-tabular)
- Hazard scoring based on mass, drift velocity, and maritime proximity
- Detection confidence tracking
"""

import json
import math
from datetime import datetime
import pandas as pd
import numpy as np


class IcebergProcessor:
    """Processes iceberg telemetry, observations, and spatial geometries."""

    @staticmethod
    def classify_hazard(size_category: str, velocity_knots: float, length_m: float) -> tuple[float, str]:
        """
        Calculates normalized hazard index (0-100) and danger category.
        High mass + rapid drift velocity = elevated navigational hazard.
        """
        base_scores = {
            "Growler / Bergy Bit": 25.0,
            "Small Pinnacle": 45.0,
            "Medium Pinnacle": 60.0,
            "Tabular": 75.0,
            "Large Tabular": 88.0,
            "Mega Tabular": 96.0,
            "Cluster": 80.0
        }
        score = base_scores.get(size_category, 50.0)

        # Kinetic energy / velocity amplifier
        velocity_penalty = min(20.0, velocity_knots * 12.0)
        total_score = min(100.0, score + velocity_penalty)

        if total_score >= 80:
            level = "CRITICAL HAZARD"
        elif total_score >= 60:
            level = "HIGH HAZARD"
        elif total_score >= 40:
            level = "MODERATE HAZARD"
        else:
            level = "LOW HAZARD"

        return round(total_score, 1), level

    @staticmethod
    def load_from_json(json_path_or_str: str) -> list[dict]:
        """Loads icebergs from JSON file or JSON string."""
        if json_path_or_str.endswith(".json"):
            with open(json_path_or_str, "r") as f:
                data = json.load(f)
        else:
            data = json.loads(json_path_or_str)

        if isinstance(data, dict) and "features" in data:
            # GeoJSON format
            icebergs = []
            for f in data["features"]:
                coords = f["geometry"]["coordinates"]
                props = f.get("properties", {})
                props["longitude"] = coords[0]
                props["latitude"] = coords[1]
                icebergs.append(props)
            return icebergs
        return data

    @staticmethod
    def load_from_csv(csv_path: str) -> list[dict]:
        """Loads icebergs from CSV file."""
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")

    @classmethod
    def process_observations(cls, raw_icebergs: list[dict]) -> list[dict]:
        """Enriches raw iceberg observations with hazard rankings and geometric properties."""
        enriched = []
        for ib in raw_icebergs:
            item = dict(ib)
            hazard_score, hazard_level = cls.classify_hazard(
                item.get("size_category", "Tabular"),
                item.get("velocity_knots", 0.8),
                item.get("length_m", 500.0)
            )
            item["hazard_score"] = hazard_score
            item["hazard_level"] = hazard_level

            # Calculate estimated submerged draft (Antarctic tabular icebergs: ~85-90% underwater)
            freeboard = item.get("height_m", 30.0)
            item["estimated_draft_m"] = round(freeboard * 5.5, 1)

            # Radar cross-section estimate
            length = item.get("length_m", 500.0)
            width = item.get("width_m", 300.0)
            item["area_sq_km"] = round((length * width) / 1e6, 3)

            enriched.append(item)

        # Sort by hazard score descending
        enriched.sort(key=lambda x: x["hazard_score"], reverse=True)
        return enriched

    @staticmethod
    def to_geojson(icebergs: list[dict]) -> dict:
        """Converts iceberg records to GeoJSON FeatureCollection."""
        features = []
        for ib in icebergs:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [ib["longitude"], ib["latitude"]]
                },
                "properties": {
                    "id": ib.get("id", "ICE-UNKNOWN"),
                    "name": ib.get("name", "Iceberg"),
                    "size_category": ib.get("size_category", "Tabular"),
                    "velocity_knots": ib.get("velocity_knots", 0.0),
                    "direction_deg": ib.get("direction_deg", 0.0),
                    "hazard_score": ib.get("hazard_score", 50.0),
                    "hazard_level": ib.get("hazard_level", "MODERATE HAZARD"),
                    "confidence": ib.get("confidence", 0.9),
                    "source": ib.get("source", "DEMO DATA")
                }
            })
        return {
            "type": "FeatureCollection",
            "features": features
        }


iceberg_processor = IcebergProcessor()
