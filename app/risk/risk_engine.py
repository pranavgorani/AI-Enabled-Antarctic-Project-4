"""
Environmental Risk Engine
Polar Navigator AI - MoES / NCPOR

Evaluates maritime operational risk (0 - 100 index) across five environmental pillars:
1. Sea-Ice Risk (concentration, thickness, compression pressure)
2. Iceberg Proximity & Trajectory Intersection Risk
3. Weather Risk (wind velocity, blowing snow, sub-zero chill, visibility)
4. Wave / Swell Risk (significant wave height, wave steepness, ice-damping)
5. Ocean Current Risk (shear, opposing currents, grounding drift)

Provides Explainable AI (XAI) feature attributions and navigational advisories.
"""

import math
from pydantic import BaseModel, Field


class RiskWeights(BaseModel):
    sea_ice_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    iceberg_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    weather_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    wave_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    current_weight: float = Field(default=0.15, ge=0.0, le=1.0)

    def normalized(self) -> "RiskWeights":
        total = (
            self.sea_ice_weight +
            self.iceberg_weight +
            self.weather_weight +
            self.wave_weight +
            self.current_weight
        )
        if total <= 0:
            return RiskWeights()
        return RiskWeights(
            sea_ice_weight=self.sea_ice_weight / total,
            iceberg_weight=self.iceberg_weight / total,
            weather_weight=self.weather_weight / total,
            wave_weight=self.wave_weight / total,
            current_weight=self.current_weight / total,
        )


class RiskEngine:
    """Calculates multi-criteria environmental risk scores and explanations."""

    def __init__(self, weights: RiskWeights = None):
        self.weights = weights.normalized() if weights else RiskWeights().normalized()

    @staticmethod
    def calculate_sea_ice_risk(concentration_pct: float, thickness_m: float = 0.5, vessel_ice_class: str = "PC3") -> float:
        """
        Calculates sea-ice risk score (0-100).
        Dependent on ice concentration and vessel polar ice classification capability.
        """
        # Baseline capability thresholds by Ice Class
        # PC1/PC2: Heavy Icebreakers; PC3/PC4: Medium Polar vessels; PC6/PC7: Light; Non-polar: zero tolerance
        class_tolerance = {
            "PC1": 95.0,
            "PC2": 90.0,
            "PC3": 80.0,
            "PC4": 70.0,
            "PC5": 60.0,
            "PC6": 50.0,
            "PC7": 40.0,
            "OpenWater": 15.0
        }
        tolerance = class_tolerance.get(vessel_ice_class, 75.0)

        # Non-linear risk curve: low below 20%, escalates rapidly above tolerance
        if concentration_pct <= 15.0:
            risk = concentration_pct * 0.6
        elif concentration_pct <= tolerance:
            risk = 9.0 + ((concentration_pct - 15.0) / (tolerance - 15.0)) * 45.0
        else:
            excess = concentration_pct - tolerance
            risk = 54.0 + (excess / (100.0 - tolerance)) * 46.0

        # Thickness modifier
        if thickness_m > 1.2:
            risk = min(100.0, risk * 1.25)

        return round(float(min(100.0, max(0.0, risk))), 1)

    @staticmethod
    def calculate_iceberg_risk(
        min_distance_km: float,
        iceberg_size: str = "Large Tabular",
        iceberg_velocity_kts: float = 1.0,
        approaching_route: bool = False
    ) -> float:
        """
        Calculates iceberg proximity and collision envelope risk (0-100).
        Distance < 5 km is critical; > 30 km is low.
        """
        if min_distance_km > 35.0:
            base_risk = max(5.0, 25.0 - (min_distance_km - 35.0) * 0.5)
        elif min_distance_km > 15.0:
            # Moderate proximity
            base_risk = 25.0 + ((35.0 - min_distance_km) / 20.0) * 35.0
        elif min_distance_km > 5.0:
            # High proximity warning zone
            base_risk = 60.0 + ((15.0 - min_distance_km) / 10.0) * 25.0
        else:
            # Critical immediate danger corridor
            base_risk = 85.0 + (5.0 - min_distance_km) * 3.0

        if approaching_route:
            base_risk = min(100.0, base_risk + 18.0)

        if "Mega" in iceberg_size or "Large" in iceberg_size:
            base_risk = min(100.0, base_risk + 8.0)

        return round(float(min(100.0, max(0.0, base_risk))), 1)

    @staticmethod
    def calculate_weather_risk(
        wind_speed_knots: float,
        air_temp_c: float,
        visibility_km: float
    ) -> float:
        """Calculates atmospheric / storm risk (0-100)."""
        # Wind risk (Gale > 34 kts, Storm > 48 kts, Hurricane > 64 kts)
        if wind_speed_knots < 20.0:
            wind_risk = wind_speed_knots * 1.0
        elif wind_speed_knots < 35.0:
            wind_risk = 20.0 + (wind_speed_knots - 20.0) * 2.2
        elif wind_speed_knots < 50.0:
            wind_risk = 53.0 + (wind_speed_knots - 35.0) * 2.4
        else:
            wind_risk = min(100.0, 89.0 + (wind_speed_knots - 50.0) * 1.5)

        # Cold chill / superstructure icing risk
        temp_risk = 0.0
        if air_temp_c < -10.0:
            temp_risk = min(25.0, (-air_temp_c - 10.0) * 2.0)

        # Poor visibility risk (fog / whiteout / blowing snow)
        vis_risk = 0.0
        if visibility_km < 5.0:
            vis_risk = min(30.0, (5.0 - visibility_km) * 6.0)

        total_weather = wind_risk * 0.7 + temp_risk * 0.15 + vis_risk * 0.15
        return round(float(min(100.0, max(0.0, total_weather))), 1)

    @staticmethod
    def calculate_wave_risk(wave_height_m: float, sea_ice_pct: float = 0.0) -> float:
        """
        Calculates sea state / wave risk (0-100).
        Ice damping dramatically suppresses wave orbital motion.
        """
        # Effective wave height accounting for wave-ice attenuation
        attenuation = max(0.1, 1.0 - (sea_ice_pct / 100.0) * 0.85)
        eff_wave = wave_height_m * attenuation

        if eff_wave < 1.5:
            wave_risk = eff_wave * 12.0
        elif eff_wave < 3.5:
            wave_risk = 18.0 + (eff_wave - 1.5) * 18.0
        elif eff_wave < 6.0:
            wave_risk = 54.0 + (eff_wave - 3.5) * 14.0
        else:
            wave_risk = min(100.0, 89.0 + (eff_wave - 6.0) * 5.0)

        return round(float(min(100.0, max(0.0, wave_risk))), 1)

    @staticmethod
    def calculate_current_risk(
        current_speed_knots: float,
        vessel_speed_knots: float = 12.0,
        is_opposing: bool = False
    ) -> float:
        """Calculates ocean current risk (0-100) due to drift and velocity opposition."""
        shear = current_speed_knots * 20.0
        if is_opposing:
            shear += current_speed_knots * 15.0
        return round(float(min(100.0, max(0.0, shear))), 1)

    def evaluate_point_risk(
        self,
        concentration_pct: float = 35.0,
        ice_thickness_m: float = 0.6,
        vessel_ice_class: str = "PC3",
        min_iceberg_dist_km: float = 22.0,
        iceberg_size: str = "Tabular",
        iceberg_approaching: bool = False,
        wind_speed_knots: float = 24.0,
        air_temp_c: float = -6.0,
        visibility_km: float = 18.0,
        wave_height_m: float = 2.4,
        current_speed_knots: float = 0.8,
        vessel_speed_knots: float = 12.0,
        is_opposing_current: bool = False,
        custom_weights: RiskWeights = None
    ) -> dict:
        """
        Evaluates full multi-criteria risk score (0-100) and produces XAI breakdown.
        """
        w = custom_weights.normalized() if custom_weights else self.weights

        r_ice = self.calculate_sea_ice_risk(concentration_pct, ice_thickness_m, vessel_ice_class)
        r_iceberg = self.calculate_iceberg_risk(min_iceberg_dist_km, iceberg_size, 1.0, iceberg_approaching)
        r_weather = self.calculate_weather_risk(wind_speed_knots, air_temp_c, visibility_km)
        r_wave = self.calculate_wave_risk(wave_height_m, concentration_pct)
        r_current = self.calculate_current_risk(current_speed_knots, vessel_speed_knots, is_opposing_current)

        overall = (
            w.sea_ice_weight * r_ice +
            w.iceberg_weight * r_iceberg +
            w.weather_weight * r_weather +
            w.wave_weight * r_wave +
            w.current_weight * r_current
        )
        overall = round(float(overall), 1)

        # Relative contribution percentage for Explainable AI
        contrib_sum = (
            w.sea_ice_weight * r_ice +
            w.iceberg_weight * r_iceberg +
            w.weather_weight * r_weather +
            w.wave_weight * r_wave +
            w.current_weight * r_current
        )
        if contrib_sum > 0:
            c_ice = round((w.sea_ice_weight * r_ice / contrib_sum) * 100, 1)
            c_iceberg = round((w.iceberg_weight * r_iceberg / contrib_sum) * 100, 1)
            c_weather = round((w.weather_weight * r_weather / contrib_sum) * 100, 1)
            c_wave = round((w.wave_weight * r_wave / contrib_sum) * 100, 1)
            c_current = round((w.current_weight * r_current / contrib_sum) * 100, 1)
        else:
            c_ice = c_iceberg = c_weather = c_wave = c_current = 20.0

        # Risk Classification
        if overall >= 75:
            severity = "CRITICAL RISK"
            advisory = "Navigation hazard critical. Divert or wait for conditions to ease."
        elif overall >= 55:
            severity = "HIGH RISK"
            advisory = "Elevated risk corridor. Reduce speed, maintain 24h radar watch, consider routing alternatives."
        elif overall >= 35:
            severity = "MODERATE RISK"
            advisory = "Normal polar transit conditions. Standard ice-pilot watch active."
        else:
            severity = "LOW RISK"
            advisory = "Favorable transit conditions within safe operational margins."

        # Dominant Factor identification
        factors = [
            ("Sea Ice", r_ice, c_ice),
            ("Iceberg Hazard", r_iceberg, c_iceberg),
            ("Severe Weather", r_weather, c_weather),
            ("Wave/Sea State", r_wave, c_wave),
            ("Ocean Current", r_current, c_current)
        ]
        factors.sort(key=lambda x: x[1], reverse=True)
        dominant_factor = factors[0][0]

        explanation = (
            f"Overall environmental risk assessed at {overall}/100 ({severity}). "
            f"Primary risk driver is {dominant_factor} ({factors[0][1]}/100, contributing {factors[0][2]}% of total cost). "
            f"Advisory: {advisory}"
        )

        return {
            "overall_risk": overall,
            "risk_score": overall,
            "severity": severity,
            "category": severity,
            "dominant_factor": dominant_factor,
            "components": {
                "sea_ice_risk": r_ice,
                "iceberg_risk": r_iceberg,
                "weather_risk": r_weather,
                "wave_risk": r_wave,
                "current_risk": r_current
            },
            "contributions_pct": {
                "sea_ice": c_ice,
                "iceberg": c_iceberg,
                "weather": c_weather,
                "wave": c_wave,
                "current": c_current
            },
            "weights_used": {
                "sea_ice": w.sea_ice_weight,
                "iceberg": w.iceberg_weight,
                "weather": w.weather_weight,
                "wave": w.wave_weight,
                "current": w.current_weight
            },
            "advisory": advisory,
            "explanation": explanation
        }


risk_engine = RiskEngine()
