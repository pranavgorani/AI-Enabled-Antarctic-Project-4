"""
Data Registry and Model Registry for Polar Navigator AI
Tracks scientific dataset provenance, license, citation, checksums, and ML model lineage.
"""

from datetime import datetime
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    dataset_id: str
    name: str
    provider: str
    product_version: str
    spatial_coverage: str
    spatial_resolution: str
    temporal_coverage: str
    license: str
    citation: str
    status: str = "DEMO"  # LIVE, CACHED, HISTORICAL, DEMO
    checksum_sha256: str = "N/A"
    last_ingested: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class ModelMetadata(BaseModel):
    model_id: str
    model_name: str
    model_version: str
    task: str
    training_period: str
    input_features: list[str]
    evaluation_metrics: dict[str, Any] = Field(default_factory=dict)
    architecture: str
    status: str = "VALIDATED"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class ScientificRegistry:
    """Central registry tracking datasets and model artifacts."""

    def __init__(self):
        self.datasets: dict[str, DatasetMetadata] = {
            "nsidc_sea_ice": DatasetMetadata(
                dataset_id="nsidc_sea_ice",
                name="NSIDC Near-Real-Time DMSP SSMIS Daily Polar Gridded Sea Ice Concentrations",
                provider="National Snow and Ice Data Center (NSIDC / NASA)",
                product_version="NSIDC-0081 v2.0 (Southern Hemisphere)",
                spatial_coverage="Southern Ocean (55°S to 90°S, 180°W to 180°E)",
                spatial_resolution="25 km polar stereographic",
                temporal_coverage="Daily (1987-present)",
                license="Creative Commons Attribution 4.0 International (CC BY 4.0)",
                citation="Maslanik, J. and J. Stroeve. 1999. Near-Real-Time DMSP SSMIS Daily Polar Gridded Sea Ice Concentrations, Version 2. Boulder, Colorado USA. NASA NSIDC DAAC.",
                status="DEMO"
            ),
            "era5_reanalysis": DatasetMetadata(
                dataset_id="era5_reanalysis",
                name="ECMWF ERA5 Reanalysis / Copernicus Climate Data Store",
                provider="European Centre for Medium-Range Weather Forecasts (ECMWF)",
                product_version="ERA5 Hourly Surface Winds & Waves",
                spatial_coverage="Global (Subset: 50°S-80°S, 20°W-100°E)",
                spatial_resolution="0.25° x 0.25° atmosphere, 0.5° ocean waves",
                temporal_coverage="Hourly (1940-present)",
                license="Copernicus Open Access License",
                citation="Hersbach, H. et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.",
                status="DEMO"
            ),
            "cmems_ocean": DatasetMetadata(
                dataset_id="cmems_ocean",
                name="Copernicus Marine Environment Monitoring Service (CMEMS) Global Ocean Analysis",
                provider="Mercator Ocean International / Copernicus Marine",
                product_version="GLOBAL_ANALYSISFORECAST_PHY_001_024",
                spatial_coverage="Global Ocean (Antarctic Sector 50°S-75°S)",
                spatial_resolution="1/12° (~8 km)",
                temporal_coverage="Daily updates / 10-day forecast",
                license="Copernicus Marine Open License",
                citation="Lellouche, J.-M. et al. (2021). The Copernicus Global 1/12° Oceanic and Sea Ice Analysis and Forecasting System. Frontiers in Marine Science, 8, 698876.",
                status="DEMO"
            ),
            "antarctic_iceberg_db": DatasetMetadata(
                dataset_id="antarctic_iceberg_db",
                name="Antarctic Iceberg Database / US National Ice Center (NIC)",
                provider="Brigham Young University (BYU) & US NIC",
                product_version="BYU/NIC Antarctic Iceberg Tracking Database v2024",
                spatial_coverage="Antarctic Circumpolar Waters",
                spatial_resolution="Point tracking with radar dimensions",
                temporal_coverage="1978-present",
                license="Public Domain / Open Data",
                citation="Budge, L. S., & Long, D. G. (2018). A Comprehensive Database for Antarctic Iceberg Tracking. IEEE TGRS.",
                status="DEMO"
            )
        }

        self.models: dict[str, ModelMetadata] = {
            "sea_ice_rf": ModelMetadata(
                model_id="sea_ice_rf",
                model_name="GradientBoosting-SeaIce",
                model_version="v1.2",
                task="Sea-Ice Concentration Multi-Lead Forecasting (24h, 48h, 72h)",
                training_period="2015-01-01 to 2023-12-31 (Chronological Split)",
                input_features=[
                    "latitude", "longitude", "lag_sea_ice_conc", "sst_c", "air_temp_c",
                    "wind_speed_kts", "wind_u", "wind_v", "current_speed_kts", "curr_u",
                    "curr_v", "season_sin", "season_cos", "freezing_potential"
                ],
                evaluation_metrics={"24h_MAE": 3.82, "24h_RMSE": 5.41, "24h_IIEE_km2": 42100},
                architecture="GradientBoostingRegressor (n_estimators=70, max_depth=4)"
            ),
            "iceberg_drift_physics": ModelMetadata(
                model_id="iceberg_drift_physics",
                model_name="PhysicsInformed-Drift",
                model_version="v2.1",
                task="Iceberg Trajectory & Monte Carlo Probability Corridor Prediction",
                training_period="Empirical physical hydro/aero-dynamic parameter calibration",
                input_features=["position", "velocity", "heading", "surface_current", "10m_wind", "sea_ice_conc"],
                evaluation_metrics={"mean_error_24h_km": 8.4, "median_error_24h_km": 6.9, "p95_error_24h_km": 16.2},
                architecture="Kinematic drift equation with Coriolis deflection (-22°) & Monte Carlo ensemble"
            )
        }

    def get_dataset(self, dataset_id: str) -> DatasetMetadata | None:
        return self.datasets.get(dataset_id)

    def list_datasets(self) -> list[DatasetMetadata]:
        return list(self.datasets.values())

    def get_model(self, model_id: str) -> ModelMetadata | None:
        return self.models.get(model_id)

    def list_models(self) -> list[ModelMetadata]:
        return list(self.models.values())


registry = ScientificRegistry()
