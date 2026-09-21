"""
SQLAlchemy Database Models for Polar Navigator AI
Supports both SQLite (local demo mode) and PostgreSQL/PostGIS (production).
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    role = Column(String(50), default="polar_navigator")  # navigator, scientist, admin
    created_at = Column(DateTime, default=datetime.utcnow)

    missions = relationship("Mission", back_populates="creator")


class Vessel(Base):
    __tablename__ = "vessels"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    call_sign = Column(String(20), unique=True, index=True)
    ice_class = Column(String(20), default="PC3")  # Polar Class PC1-PC7, 1A Super, etc.
    length_m = Column(Float, default=130.0)
    beam_m = Column(Float, default=24.0)
    draft_m = Column(Float, default=8.5)
    max_speed_knots = Column(Float, default=16.0)
    cruising_speed_knots = Column(Float, default=12.0)
    fuel_capacity_mt = Column(Float, default=2500.0)
    base_fuel_rate_mt_per_nm = Column(Float, default=0.035)  # Metric tons per nautical mile
    created_at = Column(DateTime, default=datetime.utcnow)

    missions = relationship("Mission", back_populates="vessel")
    positions = relationship("VesselPosition", back_populates="vessel")


class VesselPosition(Base):
    __tablename__ = "vessel_positions"

    id = Column(Integer, primary_key=True, index=True)
    vessel_id = Column(Integer, ForeignKey("vessels.id"))
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    heading_deg = Column(Float, default=0.0)
    speed_knots = Column(Float, default=0.0)
    recorded_at = Column(DateTime, default=datetime.utcnow)

    vessel = relationship("Vessel", back_populates="positions")


class Mission(Base):
    __tablename__ = "missions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), nullable=False)
    description = Column(Text, nullable=True)
    vessel_id = Column(Integer, ForeignKey("vessels.id"), nullable=True)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    start_port = Column(String(100), default="Cape Town / Sub-Antarctic Gateway")
    start_lat = Column(Float, default=-55.0)
    start_lon = Column(Float, default=10.0)
    destination_name = Column(String(100), default="Maitri Research Station")
    dest_lat = Column(Float, default=-70.767)
    dest_lon = Column(Float, default=11.731)
    status = Column(String(50), default="active")  # planning, active, completed
    departure_time = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    creator = relationship("User", back_populates="missions")
    vessel = relationship("Vessel", back_populates="missions")
    routes = relationship("Route", back_populates="mission")
    risk_assessments = relationship("RiskAssessment", back_populates="mission")
    alerts = relationship("Alert", back_populates="mission")


class SeaIceObservation(Base):
    __tablename__ = "sea_ice_observations"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    concentration_pct = Column(Float, nullable=False)  # 0.0 - 100.0%
    thickness_m = Column(Float, default=0.0)
    source = Column(String(50), default="DEMO_DATA")  # DEMO_DATA, SENTINEL_1, AMSR2
    observed_at = Column(DateTime, default=datetime.utcnow)


class SeaIceForecast(Base):
    __tablename__ = "sea_ice_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    lead_hours = Column(Integer, nullable=False)  # 24, 48, 72
    predicted_concentration_pct = Column(Float, nullable=False)
    confidence_score = Column(Float, default=0.85)
    model_version = Column(String(50), default="RF-IceForecast-v1.0")
    created_at = Column(DateTime, default=datetime.utcnow)


class Iceberg(Base):
    __tablename__ = "icebergs"

    id = Column(String(50), primary_key=True, index=True)  # e.g., ICE-042
    name = Column(String(100), nullable=True)
    length_m = Column(Float, default=500.0)
    width_m = Column(Float, default=300.0)
    freeboard_height_m = Column(Float, default=30.0)
    mass_mt = Column(Float, default=1.5e6)  # Estimated mass in metric tons
    status = Column(String(50), default="tracked")  # tracked, grounded, disintegrated
    created_at = Column(DateTime, default=datetime.utcnow)

    observations = relationship("IcebergObservation", back_populates="iceberg")
    trajectories = relationship("IcebergTrajectory", back_populates="iceberg")


class IcebergObservation(Base):
    __tablename__ = "iceberg_observations"

    id = Column(Integer, primary_key=True, index=True)
    iceberg_id = Column(String(50), ForeignKey("icebergs.id"), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    velocity_knots = Column(Float, default=0.5)
    direction_deg = Column(Float, default=270.0)
    confidence = Column(Float, default=0.92)
    source = Column(String(50), default="DEMO_DATA")  # DEMO_DATA, SAR, ALTIMETRY
    observed_at = Column(DateTime, default=datetime.utcnow)

    iceberg = relationship("Iceberg", back_populates="observations")


class IcebergTrajectory(Base):
    __tablename__ = "iceberg_trajectories"

    id = Column(Integer, primary_key=True, index=True)
    iceberg_id = Column(String(50), ForeignKey("icebergs.id"), nullable=False)
    lead_hours = Column(Integer, nullable=False)  # 6, 12, 24, 48, 72
    predicted_lat = Column(Float, nullable=False)
    predicted_lon = Column(Float, nullable=False)
    drift_speed_knots = Column(Float, default=0.5)
    drift_heading_deg = Column(Float, default=270.0)
    uncertainty_radius_km = Column(Float, default=5.0)
    geojson_polygon = Column(JSON, nullable=True)  # Uncertainty corridor boundary
    created_at = Column(DateTime, default=datetime.utcnow)

    iceberg = relationship("Iceberg", back_populates="trajectories")


class WeatherObservation(Base):
    __tablename__ = "weather_observations"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    air_temp_c = Column(Float, default=-5.0)
    wind_speed_knots = Column(Float, default=20.0)
    wind_direction_deg = Column(Float, default=220.0)
    pressure_hpa = Column(Float, default=985.0)
    visibility_km = Column(Float, default=15.0)
    precipitation_mm = Column(Float, default=0.0)
    wave_height_m = Column(Float, default=2.5)
    wave_period_s = Column(Float, default=8.0)
    source = Column(String(50), default="DEMO_DATA")
    recorded_at = Column(DateTime, default=datetime.utcnow)


class OceanCondition(Base):
    __tablename__ = "ocean_conditions"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    current_speed_knots = Column(Float, default=0.8)
    current_direction_deg = Column(Float, default=90.0)
    sea_surface_temp_c = Column(Float, default=-1.2)
    salinity_psu = Column(Float, default=34.1)
    source = Column(String(50), default="DEMO_DATA")
    recorded_at = Column(DateTime, default=datetime.utcnow)


class Route(Base):
    __tablename__ = "routes"

    id = Column(Integer, primary_key=True, index=True)
    mission_id = Column(Integer, ForeignKey("missions.id"), nullable=True)
    route_name = Column(String(100), nullable=False)  # Recommended, Fastest, Fuel-Efficient, Alternative
    route_type = Column(String(50), default="recommended")
    total_distance_nm = Column(Float, default=0.0)
    total_distance_km = Column(Float, default=0.0)
    estimated_duration_hours = Column(Float, default=0.0)
    estimated_fuel_mt = Column(Float, default=0.0)
    overall_risk_score = Column(Float, default=0.0)
    ice_risk_score = Column(Float, default=0.0)
    iceberg_risk_score = Column(Float, default=0.0)
    weather_risk_score = Column(Float, default=0.0)
    waypoints_geojson = Column(JSON, nullable=True)
    explanation = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    mission = relationship("Mission", back_populates="routes")
    waypoints = relationship("RouteWaypoint", back_populates="route", cascade="all, delete-orphan")


class RouteWaypoint(Base):
    __tablename__ = "route_waypoints"

    id = Column(Integer, primary_key=True, index=True)
    route_id = Column(Integer, ForeignKey("routes.id"), nullable=False)
    seq = Column(Integer, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    ice_concentration_pct = Column(Float, default=0.0)
    risk_at_waypoint = Column(Float, default=0.0)
    eta_hours = Column(Float, default=0.0)

    route = relationship("Route", back_populates="waypoints")


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, index=True)
    mission_id = Column(Integer, ForeignKey("missions.id"), nullable=True)
    overall_risk = Column(Float, nullable=False)
    sea_ice_risk = Column(Float, default=0.0)
    iceberg_risk = Column(Float, default=0.0)
    weather_risk = Column(Float, default=0.0)
    wave_risk = Column(Float, default=0.0)
    current_risk = Column(Float, default=0.0)
    weights_json = Column(JSON, nullable=True)
    explanation = Column(Text, nullable=True)
    evaluated_at = Column(DateTime, default=datetime.utcnow)

    mission = relationship("Mission", back_populates="risk_assessments")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    mission_id = Column(Integer, ForeignKey("missions.id"), nullable=True)
    priority = Column(String(20), default="HIGH")  # CRITICAL, HIGH, MEDIUM, LOW
    alert_type = Column(String(50), default="ICEBERG")  # ICEBERG, SEA_ICE, WEATHER, ROUTE
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=True)
    is_resolved = Column(Boolean, default=False)
    triggered_at = Column(DateTime, default=datetime.utcnow)

    mission = relationship("Mission", back_populates="alerts")


class DataSource(Base):
    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, index=True)
    source_name = Column(String(100), nullable=False)
    data_type = Column(String(50), nullable=False)  # SEA_ICE, WEATHER, OCEAN, ICEBERGS
    provider = Column(String(100), default="NCPOR / MoES Simulated Network")
    status = Column(String(20), default="DEMO")  # LIVE, HISTORICAL, MODELLED, DEMO
    freshness_minutes = Column(Integer, default=15)
    confidence_score = Column(Float, default=0.95)
    last_sync = Column(DateTime, default=datetime.utcnow)
