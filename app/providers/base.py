"""
Provider Abstraction for Antarctic Environmental Datasets
Follows the architecture:
DataProvider
    ├── LiveProvider
    ├── CachedProvider
    └── DemoProvider
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class DataResponse(BaseModel):
    """Standardized response envelope with explicit scientific provenance metadata."""
    data: Any
    source: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    data_status: str  # LIVE, CACHED, HISTORICAL, MODELLED, DEMO / SIMULATED
    model_version: Optional[str] = None
    confidence: Optional[float] = None
    freshness_seconds: Optional[int] = None
    spatial_coverage: Optional[str] = None
    disclaimer: Optional[str] = None


class DataProvider(ABC):
    """Abstract base class for all environmental providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        pass


class SeaIceProvider(DataProvider):
    """Interface for Sea-Ice Concentration providers."""

    @abstractmethod
    def get_current(self, bbox: Optional[dict] = None) -> DataResponse:
        pass

    @abstractmethod
    def get_historical(self, start_date: str, end_date: str) -> DataResponse:
        pass

    @abstractmethod
    def get_forecast(self, lead_hours: int = 24) -> DataResponse:
        pass


class WeatherProvider(DataProvider):
    """Interface for Meteorological providers (winds, temperatures, pressure)."""

    @abstractmethod
    def get_current(self, lat: Optional[float] = None, lon: Optional[float] = None) -> DataResponse:
        pass

    @abstractmethod
    def get_forecast(self, lead_hours: int = 24) -> DataResponse:
        pass


class OceanProvider(DataProvider):
    """Interface for Oceanographic providers (currents, SST, waves)."""

    @abstractmethod
    def get_currents(self, lat: Optional[float] = None, lon: Optional[float] = None) -> DataResponse:
        pass

    @abstractmethod
    def get_temperature(self, lat: Optional[float] = None, lon: Optional[float] = None) -> DataResponse:
        pass

    @abstractmethod
    def get_waves(self, lat: Optional[float] = None, lon: Optional[float] = None) -> DataResponse:
        pass


class IcebergProvider(DataProvider):
    """Interface for Iceberg observations and trajectory tracking."""

    @abstractmethod
    def get_observations(self) -> DataResponse:
        pass

    @abstractmethod
    def get_tracks(self, iceberg_id: str) -> DataResponse:
        pass
