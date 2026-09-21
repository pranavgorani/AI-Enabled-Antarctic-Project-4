"""
Maritime Fuel Optimization and ETA Calculation Model
Polar Navigator AI - MoES / NCPOR

Calculates:
- Base Fuel = Distance * Base Fuel Rate
- Ice Factor: Exponential resistance increase with sea-ice concentration & thickness
- Weather Factor: Aerodynamic drag penalty from adverse wind & waves
- Current Factor: Drift assistance or resistance
- Total Adjusted Fuel Consumption (Metric Tons)
- ETA & Transit Duration
- Estimated Voyage Fuel Cost ($ USD / MT)
"""

import math
from datetime import datetime, timedelta


class PolarFuelModel:
    """Calculates fuel consumption, emissions, and transit duration under polar conditions."""

    FUEL_COST_USD_PER_MT = 780.0  # Typical marine gas oil / low-sulfur polar fuel cost

    @classmethod
    def calculate_ice_factor(cls, concentration_pct: float, thickness_m: float = 0.5, ice_class: str = "PC3") -> float:
        """
        Calculates ice resistance factor.
        In open water (< 15%), factor is 1.0.
        In heavy pack ice (> 75%), factor can exceed 2.5 - 3.2 depending on hull ice class.
        """
        if concentration_pct <= 15.0:
            return 1.0

        # Class multiplier: PC1 cuts ice easier than PC5
        class_efficiency = {
            "PC1": 0.65, "PC2": 0.75, "PC3": 0.85, "PC4": 0.95,
            "PC5": 1.10, "PC6": 1.25, "PC7": 1.45, "OpenWater": 2.20
        }.get(ice_class, 0.9)

        # Ice breaking resistance non-linear scaling
        conc_ratio = (concentration_pct - 15.0) / 85.0
        thickness_factor = max(1.0, 1.0 + (thickness_m - 0.5) * 0.8)

        penalty = (conc_ratio ** 1.8) * 1.8 * class_efficiency * thickness_factor
        return round(1.0 + penalty, 3)

    @classmethod
    def calculate_weather_factor(cls, wind_speed_knots: float, wave_height_m: float) -> float:
        """
        Calculates aerodynamic & hydrodynamic sea-state penalty.
        Calm waters = 1.0; Storm force winds + 5m waves = 1.45+.
        """
        wind_penalty = max(0.0, (wind_speed_knots - 15.0) * 0.007)
        wave_penalty = max(0.0, (wave_height_m - 1.5) * 0.045)
        return round(1.0 + wind_penalty + wave_penalty, 3)

    @classmethod
    def calculate_current_factor(
        cls,
        current_speed_knots: float,
        current_dir_deg: float,
        vessel_heading_deg: float,
        vessel_speed_knots: float = 12.0
    ) -> float:
        """
        Calculates current assistance/resistance along the vessel track.
        Favorable current reduces fuel; opposing current increases fuel.
        """
        # Angle between current vector and ship track
        angle_diff = math.radians(abs((current_dir_deg - vessel_heading_deg + 180) % 360 - 180))
        # Effective along-track current component
        effective_current = current_speed_knots * math.cos(angle_diff)

        # Ratio of effective speed alteration
        speed_ratio = effective_current / max(5.0, vessel_speed_knots)
        factor = 1.0 - (speed_ratio * 0.75)
        return round(max(0.75, min(1.35, factor)), 3)

    @classmethod
    def estimate_segment(
        cls,
        distance_nm: float,
        cruising_speed_knots: float = 12.0,
        base_rate_mt_per_nm: float = 0.035,
        concentration_pct: float = 30.0,
        ice_thickness_m: float = 0.5,
        ice_class: str = "PC3",
        wind_speed_knots: float = 22.0,
        wave_height_m: float = 2.0,
        current_speed_knots: float = 0.7,
        current_dir_deg: float = 270.0,
        vessel_heading_deg: float = 180.0
    ) -> dict:
        """
        Calculates fuel and time for an individual route leg or whole route.
        """
        ice_factor = cls.calculate_ice_factor(concentration_pct, ice_thickness_m, ice_class)
        weather_factor = cls.calculate_weather_factor(wind_speed_knots, wave_height_m)
        current_factor = cls.calculate_current_factor(current_speed_knots, current_dir_deg, vessel_heading_deg, cruising_speed_knots)

        # In ice pack, speed is reduced
        effective_speed = cruising_speed_knots
        if concentration_pct > 60.0:
            effective_speed = max(4.5, cruising_speed_knots * (1.0 - (concentration_pct - 60.0) / 100.0 * 0.7))
        elif concentration_pct > 30.0:
            effective_speed = max(8.0, cruising_speed_knots * 0.88)

        transit_hours = distance_nm / max(2.0, effective_speed)

        base_fuel = distance_nm * base_rate_mt_per_nm
        adjusted_fuel = base_fuel * ice_factor * weather_factor * current_factor
        fuel_cost_usd = adjusted_fuel * cls.FUEL_COST_USD_PER_MT

        return {
            "distance_nm": round(distance_nm, 1),
            "distance_km": round(distance_nm * 1.852, 1),
            "transit_hours": round(transit_hours, 1),
            "effective_speed_knots": round(effective_speed, 1),
            "base_fuel_mt": round(base_fuel, 2),
            "adjusted_fuel_mt": round(adjusted_fuel, 2),
            "fuel_cost_usd": round(fuel_cost_usd, 0),
            "ice_factor": ice_factor,
            "weather_factor": weather_factor,
            "current_factor": current_factor,
            "fuel_penalty_pct": round(((adjusted_fuel / max(0.01, base_fuel)) - 1.0) * 100, 1)
        }


fuel_model = PolarFuelModel()
