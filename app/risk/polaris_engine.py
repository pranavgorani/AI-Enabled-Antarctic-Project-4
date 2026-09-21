"""
POLARIS-Style Polar Ice Risk Engine
Reference: IMO MSC.1/Circ.1519 (2016) Polar Operational Limit Assessment Risk Indexing System
Research Implementation — Not Regulatory Approval

Calculates:
- Risk Index Outcome (RIO) for vessel polar class and ice regime
  RIO = Sum(C_i * RIV_i)
- Operational Risk Criteria:
  RIO >= 0: Normal Operation
  -10 <= RIO < 0: Elevated Operational Risk (Speed limitation & enhanced monitoring)
  RIO < -10: Special Consideration Required / Subject to Special Conditions
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import BaseModel, Field


class VesselProfile(BaseModel):
    name: str = "RV Explorer"
    ice_class: str = "PC3"  # PC1 - PC7, OpenWaterNonPolar
    polar_category: str = "Category A (High Polar)"
    length_m: float = 130.0
    beam_m: float = 24.0
    draft_m: float = 8.5
    displacement_mt: float = 12500.0
    engine_power_kw: float = 14500.0
    cruising_speed_knots: float = 12.0
    max_speed_knots: float = 15.5
    fuel_rate_mt_per_nm: float = 0.035


class PolarisIceRegime(BaseModel):
    """Ice concentration breakdown by development stage (tenths or percentages)."""
    multi_year_ice_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    second_year_ice_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    thick_first_year_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    medium_first_year_pct: float = Field(default=25.0, ge=0.0, le=100.0)
    thin_first_year_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    grey_white_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    grey_ice_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    open_water_pct: float = Field(default=35.0, ge=0.0, le=100.0)


class PolarisEngine:
    """Calculates POLARIS-style Risk Index Outcome (RIO) from versioned IMO tables."""

    def __init__(self, config_path: str = "config/polaris_riv.yaml"):
        self.config_path = config_path
        self.riv_tables = self._load_riv_tables()

    def _load_riv_tables(self) -> dict:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                    return cfg.get("riv_tables", {})
            except Exception:
                pass
        # Fallback embedded RIV table from IMO MSC.1/Circ.1519
        return {
            "PC1": {"multi_year_ice": 2, "second_year_ice": 2, "thick_first_year_ice": 3, "medium_first_year_ice": 3, "thin_first_year_ice": 3, "grey_white_ice": 3, "grey_ice": 3, "open_water": 3},
            "PC3": {"multi_year_ice": 0, "second_year_ice": 1, "thick_first_year_ice": 2, "medium_first_year_ice": 2, "thin_first_year_ice": 3, "grey_white_ice": 3, "grey_ice": 3, "open_water": 3},
            "PC5": {"multi_year_ice": -2, "second_year_ice": -1, "thick_first_year_ice": 0, "medium_first_year_ice": 1, "thin_first_year_ice": 2, "grey_white_ice": 2, "grey_ice": 3, "open_water": 3},
            "PC7": {"multi_year_ice": -3, "second_year_ice": -3, "thick_first_year_ice": -2, "medium_first_year_ice": -1, "thin_first_year_ice": 0, "grey_white_ice": 1, "grey_ice": 2, "open_water": 3},
            "OpenWaterNonPolar": {"multi_year_ice": -3, "second_year_ice": -3, "thick_first_year_ice": -3, "medium_first_year_ice": -3, "thin_first_year_ice": -2, "grey_white_ice": -1, "grey_ice": 0, "open_water": 3}
        }

    @classmethod
    def get_vessel_profile(cls, name: str = "RV Bharati Explorer") -> dict:
        """Returns standard polar research vessel profile."""
        profiles = {
            "RV Bharati Explorer": {
                "name": "RV Bharati Explorer",
                "ice_class": "PC4",
                "polar_category": "Category A",
                "length_m": 125.0,
                "beam_m": 22.0,
                "draft_m": 7.5,
                "displacement_mt": 11500.0,
                "engine_power_kw": 12000.0,
                "cruising_speed_knots": 12.0
            },
            "RV Explorer": {
                "name": "RV Explorer",
                "ice_class": "PC3",
                "polar_category": "Category A",
                "length_m": 130.0,
                "beam_m": 24.0,
                "draft_m": 8.5,
                "displacement_mt": 12500.0,
                "engine_power_kw": 14500.0,
                "cruising_speed_knots": 12.5
            }
        }
        return profiles.get(name, profiles["RV Bharati Explorer"])

    @classmethod
    def calculate_rio(
        cls,
        vessel: Any = None,
        regime: Any = None,
        ice_regime: Any = None,
        ice_class: str = "PC4"
    ) -> dict:
        inst = cls() if isinstance(cls, type) else cls
        return inst._do_calculate_rio(vessel=vessel, regime=regime, ice_regime=ice_regime, ice_class=ice_class)

    def _do_calculate_rio(
        self,
        vessel: Any = None,
        regime: Any = None,
        ice_regime: Any = None,
        ice_class: str = "PC4"
    ) -> dict:
        """
        Computes RIO = Sum(C_i * RIV_i) where C_i is concentration in tenths (0 to 10).
        Accepts either (vessel, regime) objects or (ice_regime list/dict, ice_class str).
        """
        v_name = "RV Bharati Explorer"
        if vessel is not None:
            if isinstance(vessel, VesselProfile):
                ice_class = vessel.ice_class
                v_name = vessel.name
            elif isinstance(vessel, dict):
                ice_class = vessel.get("ice_class", ice_class)
                v_name = vessel.get("name", v_name)
            elif isinstance(vessel, str):
                ice_class = vessel

        if ice_class not in self.riv_tables:
            # Safe default fallback
            ice_class = "PC4" if "PC4" in self.riv_tables else "PC3"

        riv_map = self.riv_tables.get(ice_class, self.riv_tables.get("PC3", {}))

        # Build tenths mapping
        tenths = {}
        if regime is not None and isinstance(regime, PolarisIceRegime):
            tenths = {
                "multi_year_ice": regime.multi_year_ice_pct / 10.0,
                "second_year_ice": regime.second_year_ice_pct / 10.0,
                "thick_first_year_ice": regime.thick_first_year_pct / 10.0,
                "medium_first_year_ice": regime.medium_first_year_pct / 10.0,
                "thin_first_year_ice": regime.thin_first_year_pct / 10.0,
                "grey_white_ice": regime.grey_white_pct / 10.0,
                "grey_ice": regime.grey_ice_pct / 10.0,
                "open_water": regime.open_water_pct / 10.0
            }
        elif ice_regime is not None:
            if isinstance(ice_regime, list):
                for item in ice_regime:
                    itype = item.get("ice_type", "open_water")
                    # Map second stage alias if present
                    if "thick_first_year_ice" in itype:
                        itype = "thick_first_year_ice"
                    elif "medium_first_year_ice" in itype:
                        itype = "medium_first_year_ice"
                    elif "thin_first_year_ice" in itype:
                        itype = "thin_first_year_ice"

                    c_tenths = item.get("concentration_tenths", item.get("concentration_pct", 0) / 10.0)
                    tenths[itype] = tenths.get(itype, 0.0) + float(c_tenths)
            elif isinstance(ice_regime, dict):
                for k, v in ice_regime.items():
                    tenths[k] = float(v) / 10.0 if float(v) > 10.0 else float(v)

        # Calculate sum of products
        rio = 0.0
        contributions = {}
        for ice_type, c_tenths in tenths.items():
            riv_val = riv_map.get(ice_type, riv_map.get("open_water", 3))
            contrib = c_tenths * riv_val
            rio += contrib
            contributions[ice_type] = {
                "concentration_pct": round(c_tenths * 10.0, 1),
                "concentration_tenths": round(c_tenths, 2),
                "riv": riv_val,
                "rio_contribution": round(contrib, 2)
            }

        rio = int(round(rio))

        # Operational Category Determination
        if rio >= 0:
            category = "Normal Operation"
            color = "#10B981"  # Emerald Green
            advisory = "Ice conditions are within standard operational envelope for vessel ice class."
        elif rio >= -10:
            category = "Elevated Operational Risk"
            color = "#F59E0B"  # Amber
            advisory = "Operation requires caution, speed limitation, and active ice-lead monitoring."
        else:
            category = "Special Consideration Required"
            color = "#EF4444"  # Red
            advisory = "Ice severity exceeds standard design capability. Operation subject to special conditions."

        return {
            "vessel_name": v_name,
            "ice_class": ice_class,
            "rio": rio,
            "status": category,
            "operational_category": category,
            "color": color,
            "advisory": advisory,
            "methodology": "POLARIS-style research implementation (IMO MSC.1/Circ.1519)",
            "disclaimer": "POLARIS-style research implementation — not regulatory approval.",
            "type_contributions": contributions
        }

    def estimate_regime_from_concentration(self, total_conc_pct: float) -> PolarisIceRegime:
        """
        Synthesizes stage-of-development partition when only total concentration is available.
        """
        ow = max(0.0, 100.0 - total_conc_pct)
        if total_conc_pct <= 15.0:
            return PolarisIceRegime(open_water_pct=100.0)
        elif total_conc_pct <= 45.0:
            return PolarisIceRegime(
                open_water_pct=ow,
                thin_first_year_pct=total_conc_pct * 0.6,
                medium_first_year_pct=total_conc_pct * 0.4
            )
        elif total_conc_pct <= 75.0:
            return PolarisIceRegime(
                open_water_pct=ow,
                medium_first_year_pct=total_conc_pct * 0.45,
                thick_first_year_pct=total_conc_pct * 0.40,
                second_year_ice_pct=total_conc_pct * 0.15
            )
        else:
            return PolarisIceRegime(
                open_water_pct=ow,
                thick_first_year_pct=total_conc_pct * 0.45,
                second_year_ice_pct=total_conc_pct * 0.35,
                multi_year_ice_pct=total_conc_pct * 0.20
            )


polaris_engine = PolarisEngine()
