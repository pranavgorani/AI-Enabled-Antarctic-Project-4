"""
PDF Mission Report Generator
Polar Navigator AI - MoES / NCPOR

Generates comprehensive Antarctic Navigation Decision Support Mission PDF reports:
- Mission Summary
- Environmental Conditions
- Sea-Ice Forecast
- Iceberg Analysis & Trajectory Predictions
- Multi-Objective Route Comparison
- Risk Analysis & Explainable AI
- Fuel Analysis & Cost Estimation
- Navigation Briefing
- Data Sources & Model Provenance
"""

import os
from datetime import datetime
from fpdf import FPDF
from app.services.ai_assistant import ai_assistant
from app.routing.optimizer import route_optimizer


class PolarMissionPDF(FPDF):
    """Custom PDF document styling for Polar Navigator AI."""

    def header(self):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(10, 25, 47)  # Navy
        self.cell(0, 8, "POLAR NAVIGATOR AI - MoES / NCPOR", ln=1, align="L")
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "Antarctic Sea-Ice, Iceberg Trajectory & Navigation Decision Support System", ln=1, align="L")
        self.line(10, 25, 200, 25)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | RESEARCH PROTOTYPE — NOT FOR REAL-WORLD NAVIGATION", align="C")


def generate_mission_pdf(output_path: str = "mission_report.pdf") -> str:
    """Generates the mission report PDF and saves it to output_path."""
    pdf = PolarMissionPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    routes = route_optimizer.generate_all_routes()
    rec = routes["recommended"]
    icebergs = ai_assistant.get_icebergs()
    weather = ai_assistant.get_weather(-64.5, 11.0)
    sea_ice_data = ai_assistant.get_sea_ice()

    # Title Block
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(14, 116, 144)  # Deep Cyan
    pdf.cell(0, 10, "ANTARCTIC MISSION NAVIGATION & RISK REPORT", ln=1, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 6, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | Data Mode: DEMO / SIMULATED", ln=1, align="C")
    pdf.ln(5)

    # Section 1: Mission Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(10, 25, 47)
    pdf.cell(0, 7, "1. MISSION SUMMARY & VESSEL PROFILE", ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, (
        "Mission: Antarctic Expedition Alpha (2025-2026 Season)\n"
        "Vessel: RV Explorer (Call Sign: VLEXP-IND)\n"
        "Ice Class: Polar Class PC3 (Year-round operation in second-year ice)\n"
        "Destination: Maitri Research Station (-70.767°S, 11.731°E, Schirmacher Oasis)\n"
        "Departure Staging: Sub-Antarctic Gateway (-58.5°S, 10.5°E)\n"
        f"Selected Track: Recommended Lower-Risk Route ({rec['total_distance_km']} km, ETA {rec['estimated_duration_hours']} hours)"
    ))
    pdf.ln(4)

    # Section 2: Environmental Conditions
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "2. ENVIRONMENTAL & METEOROLOGICAL TELEMETRY", ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, (
        f"Air Temperature: {weather.get('air_temp_c', -7.0)} °C | Freezing Point Depression: Moderate\n"
        f"Wind Velocity: {weather.get('wind_speed_knots', 22.0)} kts from {weather.get('wind_direction_deg', 240)}° (Katabatic / Westerlies)\n"
        f"Significant Wave Height: {weather.get('wave_height_m', 2.2)} m (Attenuated inside ice pack to < 0.6 m)\n"
        f"Atmospheric Pressure: {weather.get('pressure_hpa', 984.0)} hPa | Visibility: {weather.get('visibility_km', 18.0)} km\n"
        "Ocean Current: Antarctic Circumpolar Current (ACC) eastward in north; Antarctic Counter-Current westward in south"
    ))
    pdf.ln(4)

    # Section 3: Sea-Ice Forecast
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "3. SEA-ICE FORECAST & CONCENTRATION OUTLOOK", ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, (
        f"Forecasting Model: {sea_ice_data['model']} (Forecast Confidence: {int(sea_ice_data['forecast_confidence']*100)}%)\n"
        "Lead Horizions: 24h, 48h, 72h physical thermodynamic & advective regression\n"
        "Ice Edge / Marginal Ice Zone: -61.5°S (15-35% concentration)\n"
        "Pack Ice Transit: -65.0°S to -68.5°S (Average concentration 52.4%)\n"
        "Polynya Access: Stable coastal lead observed near 11.5°E providing lower-resistance passage."
    ))
    pdf.ln(4)

    # Section 4: Iceberg Intelligence
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "4. ICEBERG SURVEILLANCE & TRAJECTORY PREDICTION", ln=1)
    pdf.set_font("Helvetica", "", 9)
    top_ib = icebergs[0] if icebergs else {}
    pdf.multi_cell(0, 5, (
        f"Active Tracked Icebergs: {len(icebergs)} targets in the operational sector\n"
        f"Primary Hazard: {top_ib.get('id', 'ICE-042')} ({top_ib.get('size_category', 'Large Tabular')}, {top_ib.get('length_m', 950)}m length)\n"
        f"Current Position: {top_ib.get('latitude', -64.4)}°S, {top_ib.get('longitude', 10.9)}°E\n"
        f"Drift Heading: {top_ib.get('direction_deg', 272)}° at {top_ib.get('velocity_knots', 1.15)} kts\n"
        "Trajectory Model: Physics-Informed Drift (Water Drag 0.85, Wind Drag 0.028, Coriolis -22°, Sea-Ice Resistance)\n"
        "Proximity Advisory: Minimum Closest Point of Approach (CPA) is projected at 14.8 km from Waypoint 6."
    ))
    pdf.ln(4)

    # Section 5: Multi-Objective Route Comparison
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "5. MULTI-OBJECTIVE ROUTE EVALUATION & TRADE-OFFS", ln=1)
    pdf.set_font("Helvetica", "", 8)

    # Table Header
    pdf.set_fill_color(220, 230, 242)
    pdf.cell(45, 6, "Route Name", 1, 0, "L", fill=True)
    pdf.cell(25, 6, "Distance", 1, 0, "C", fill=True)
    pdf.cell(25, 6, "ETA", 1, 0, "C", fill=True)
    pdf.cell(25, 6, "Fuel (MT)", 1, 0, "C", fill=True)
    pdf.cell(25, 6, "Risk (/100)", 1, 0, "C", fill=True)
    pdf.cell(45, 6, "Trade-Off Rationale", 1, 1, "L", fill=True)

    summary_items = routes.get("comparison_summary", [])
    for item in summary_items:
        pdf.cell(45, 6, item["route"], 1, 0, "L")
        pdf.cell(25, 6, f"{item['distance_nm']} NM", 1, 0, "C")
        pdf.cell(25, 6, f"{item['eta_hours']} hrs", 1, 0, "C")
        pdf.cell(25, 6, f"{item['fuel_mt']} MT", 1, 0, "C")
        pdf.cell(25, 6, f"{item['risk_score']}", 1, 0, "C")
        pdf.cell(45, 6, item["trade_off"][:30] + "...", 1, 1, "L")

    pdf.ln(4)

    # Section 6: Fuel & Cost Estimation
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "6. FUEL CONSUMPTION & EMISSIONS MODEL", ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, (
        f"Estimated Total Fuel: {rec['estimated_fuel_mt']} MT Marine Gas Oil (MGO)\n"
        f"Estimated Fuel Expenditure: ${rec['estimated_fuel_cost_usd']:,.0f} USD (@ $780/MT)\n"
        "Fuel Economy Optimization: Routing through open polynya leads saves an estimated 7.2 MT of fuel "
        "compared to forcing direct transit through high-concentration ridge ice."
    ))
    pdf.ln(4)

    # Section 7: Safety Notice & Disclaimer
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(180, 40, 40)
    pdf.cell(0, 6, "RESEARCH & DECISION SUPPORT DISCLAIMER", ln=1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 4, (
        "This platform is a research and decision-support prototype developed for the Ministry of Earth Sciences (MoES) "
        "and National Centre for Polar and Ocean Research (NCPOR). Environmental predictions, iceberg drift projections, "
        "and navigation routes shown in DEMO MODE are simulated and must not be used as the sole basis for real-world "
        "Antarctic navigation. Real-world navigation requires certified ice-pilotage and master authorization."
    ))

    pdf.output(output_path)
    return output_path
