"""
Ship Resistance and Fuel Consumption in Antarctic Ice Regimes.

Implements the Lindqvist (1989) semi-empirical formulation for level ice resistance:
R_ice = R_crushing + R_breaking + R_submersion

Reference:
- Lindqvist, G. (1989). A straightforward method for calculation of ice resistance
  of ships. Proceedings of the 10th International Conference on Port and Ocean
  Engineering under Arctic Conditions (POAC '89), Lulea, Sweden, Vol. 2, pp. 722-735.

Notes on Calibration:
- Crushing, breaking, and submersion parameters use standard empirical constants
  from Lindqvist (1989).
- Open-water resistance uses standard Holtrop & Mennen simplified naval approximation.
- Specific Fuel Oil Consumption (SFOC) is modeled at 185 g/kWh (assumed for medium-speed marine diesel).
- Clearly flagged where parameters are assumed coefficients.
"""

from typing import Dict, Any
import numpy as np


class LindqvistFuelModel:
    """
    Computes ice resistance, open-water resistance, propulsion power, and fuel burn.
    """

    # Gravitational acceleration (m/s^2)
    G = 9.81
    # Water density (kg/m^3)
    RHO_W = 1025.0
    # Sea-ice density (kg/m^3)
    RHO_ICE = 917.0
    # Standard assumed Specific Fuel Oil Consumption (SFOC) in g/kWh
    DEFAULT_SFOC_G_KWH = 185.0  # Assumed coefficient for modern 4-stroke marine diesel

    @classmethod
    def calculate_lindqvist_ice_resistance(
        cls,
        ice_thickness_m: float,
        vessel_speed_ms: float,
        vessel_beam_m: float,
        vessel_length_m: float,
        vessel_draft_m: float,
        stem_angle_deg: float = 30.0,
        waterline_angle_deg: float = 30.0,
        flexural_strength_kpa: float = 500.0,
        friction_coef: float = 0.15
    ) -> Dict[str, float]:
        """
        Calculate total ice resistance (R_ice) in Newtons using Lindqvist (1989).

        Components:
        1. R_c: Crushing resistance at bow
        2. R_b: Breaking (bending) resistance
        3. R_s: Submersion resistance (buoyancy and friction of submerged ice blocks)

        Returns:
            Dict containing R_c, R_b, R_s, and total R_ice in kN.
        """
        if ice_thickness_m <= 0.001 or vessel_speed_ms <= 0.01:
            return {"R_c_kN": 0.0, "R_b_kN": 0.0, "R_s_kN": 0.0, "R_ice_kN": 0.0}

        B = vessel_beam_m
        L = vessel_length_m
        T = vessel_draft_m
        h = ice_thickness_m
        v = vessel_speed_ms

        phi = np.radians(stem_angle_deg)        # Stem angle
        alpha = np.radians(waterline_angle_deg) # Waterline entrance angle
        sigma_b = flexural_strength_kpa * 1000.0 # Convert kPa to Pa (N/m^2)
        mu = friction_coef                     # Assumed ice-hull friction coefficient (0.15 for clean coated hull)

        delta_rho = cls.RHO_W - cls.RHO_ICE

        # 1. Crushing Resistance (R_c)
        # R_c = 0.5 * sigma_b * h^2 * (tan(phi) + mu * cos(phi) / cos(alpha)) / (1 - mu * sin(phi) / cos(alpha))
        denom_c = max(0.1, 1.0 - mu * np.sin(phi) / np.cos(alpha))
        R_c = 0.5 * sigma_b * (h ** 2) * (np.tan(phi) + mu * np.cos(phi) / np.cos(alpha)) / denom_c

        # 2. Breaking / Bending Resistance (R_b)
        # R_b = 0.0316 * sigma_b * B * h * (h / B)^0.3 * (1 + 1.4 * v / sqrt(g * h))
        v_rel = v / np.sqrt(cls.G * max(0.05, h))
        R_b = 0.0316 * sigma_b * B * h * ((h / B) ** 0.3) * (1.0 + 1.4 * v_rel)

        # 3. Submersion Resistance (R_s)
        # R_s = delta_rho * g * h * B * (T * (B + T) / (B + 2T) + mu * (0.7 * L - T / tan(phi) - B / (4 * tan(alpha)) + T * cos(phi) * cos(alpha) / sin(phi)))
        geom_term1 = T * (B + T) / (B + 2.0 * T)
        geom_term2 = mu * max(
            0.0,
            0.7 * L - T / np.tan(phi) - B / (4.0 * np.tan(alpha)) + (T * np.cos(phi) * np.cos(alpha) / np.sin(phi))
        )
        R_s = delta_rho * cls.G * h * B * (geom_term1 + geom_term2) * (1.0 + 1.4 * v_rel)

        total_R_ice = R_c + R_b + R_s

        return {
            "R_c_kN": round(R_c / 1000.0, 2),
            "R_b_kN": round(R_b / 1000.0, 2),
            "R_s_kN": round(R_s / 1000.0, 2),
            "R_ice_kN": round(total_R_ice / 1000.0, 2)
        }

    @classmethod
    def calculate_open_water_resistance(
        cls,
        vessel_speed_ms: float,
        vessel_length_m: float,
        vessel_beam_m: float,
        vessel_draft_m: float,
        block_coefficient: float = 0.65
    ) -> float:
        """
        Simplified open-water resistance estimation (kN) using standard hydrodynamic scaling.
        R_ow = 0.5 * rho_w * v^2 * S * C_t
        """
        if vessel_speed_ms <= 0.01:
            return 0.0

        # Wetted surface area approximation (Denny-Mumford formula)
        # S = 1.7 * L * T + C_b * B * L
        S = 1.7 * vessel_length_m * vessel_draft_m + block_coefficient * vessel_beam_m * vessel_length_m

        # Total resistance coefficient C_t (frictional + residual)
        # Assumed coefficient: ~0.0035 for displacement research vessels at moderate Froude numbers
        C_t = 0.0038

        R_ow_N = 0.5 * cls.RHO_W * (vessel_speed_ms ** 2) * S * C_t
        return round(R_ow_N / 1000.0, 2)

    @classmethod
    def estimate_fuel_burn(
        cls,
        distance_nm: float,
        speed_knots: float,
        ice_concentration: float,  # 0.0 to 1.0
        ice_thickness_m: float,
        vessel_profile: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Estimate transit time, required propulsion power (kW), and fuel consumed (metric tonnes).

        Args:
            distance_nm: Nautical miles along the leg.
            speed_knots: Desired cruising speed in open water.
            ice_concentration: Fractional sea-ice coverage (0.0 to 1.0).
            ice_thickness_m: Mean ice thickness in meters.
            vessel_profile: Vessel dimensions, class, and propulsion parameters.

        Returns:
            Dict containing power_kw, fuel_burn_tonnes, transit_time_hours, and resistance components.
        """
        if vessel_profile is None:
            # Default to RV Bharati Explorer (PC4-ish Polar Research Vessel)
            vessel_profile = {
                "name": "RV Bharati Explorer",
                "length_m": 125.0,
                "beam_m": 22.0,
                "draft_m": 7.5,
                "propulsion_efficiency": 0.65,  # Quasi-propulsive coefficient (assumed)
                "max_mcr_kw": 12000.0,          # Maximum Continuous Rating
                "hotel_load_kw": 800.0,         # Auxiliary generator electrical load
                "sfoc_g_kwh": cls.DEFAULT_SFOC_G_KWH
            }

        # Adjust actual speed based on ice concentration
        # Vessels must slow down in dense ice
        speed_factor = max(0.25, 1.0 - 0.70 * ice_concentration)
        actual_speed_knots = max(2.5, speed_knots * speed_factor)
        v_ms = actual_speed_knots * 0.514444

        # Calculate open-water and ice resistances
        r_ow_kn = cls.calculate_open_water_resistance(
            vessel_speed_ms=v_ms,
            vessel_length_m=vessel_profile.get("length_m", 125.0),
            vessel_beam_m=vessel_profile.get("beam_m", 22.0),
            vessel_draft_m=vessel_profile.get("draft_m", 7.5)
        )

        r_ice_components = cls.calculate_lindqvist_ice_resistance(
            ice_thickness_m=ice_thickness_m,
            vessel_speed_ms=v_ms,
            vessel_beam_m=vessel_profile.get("beam_m", 22.0),
            vessel_length_m=vessel_profile.get("length_m", 125.0),
            vessel_draft_m=vessel_profile.get("draft_m", 7.5)
        )

        # Weighted combined resistance:
        # In partial ice concentration C, effective resistance is (1-C)*R_ow + C*(R_ow + R_ice)
        effective_r_kn = (1.0 - ice_concentration) * r_ow_kn + ice_concentration * (r_ow_kn + r_ice_components["R_ice_kN"])

        # Effective Towing Power P_E = R * v (kW)
        # Delivered Propulsion Power P_D = P_E / eta_prop
        eta_prop = vessel_profile.get("propulsion_efficiency", 0.65)
        propulsion_power_kw = (effective_r_kn * v_ms) / max(0.1, eta_prop)

        # Cap at Maximum Continuous Rating (MCR)
        max_mcr = vessel_profile.get("max_mcr_kw", 12000.0)
        power_kw = min(max_mcr, propulsion_power_kw)

        # Total power including auxiliary hotel load
        hotel_load = vessel_profile.get("hotel_load_kw", 800.0)
        total_power_kw = power_kw + hotel_load

        # Transit time (hours)
        transit_time_hours = distance_nm / actual_speed_knots if actual_speed_knots > 0 else 0.0

        # Fuel burn: Fuel (tonnes) = Total_Power (kW) * Time (h) * SFOC (g/kWh) / 1,000,000
        sfoc = vessel_profile.get("sfoc_g_kwh", cls.DEFAULT_SFOC_G_KWH)
        fuel_burn_tonnes = (total_power_kw * transit_time_hours * sfoc) / 1_000_000.0

        return {
            "distance_nm": round(distance_nm, 1),
            "planned_speed_knots": round(speed_knots, 1),
            "actual_speed_knots": round(actual_speed_knots, 1),
            "transit_time_hours": round(transit_time_hours, 2),
            "ice_concentration": round(ice_concentration, 2),
            "ice_thickness_m": round(ice_thickness_m, 2),
            "effective_resistance_kN": round(effective_r_kn, 2),
            "open_water_resistance_kN": round(r_ow_kn, 2),
            "ice_resistance_breakdown_kN": r_ice_components,
            "propulsion_power_kW": round(power_kw, 1),
            "total_power_kW": round(total_power_kw, 1),
            "fuel_burn_tonnes": round(fuel_burn_tonnes, 3),
            "assumptions": {
                "method": "Lindqvist (1989) Level Ice Semi-Empirical Formulation",
                "sfoc_g_kwh": sfoc,
                "propulsion_efficiency": eta_prop,
                "note": "Parameters include assumed empirical friction and hull geometry coefficients"
            }
        }
