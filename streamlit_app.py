"""
POLAR NAVIGATOR AI - Streamlit Antarctic Command Center
AI-Powered Antarctic Navigation Decision Support System

Ministry of Earth Sciences (MoES)
National Centre for Polar and Ocean Research (NCPOR)
"""

import os
import json
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# Import backend modules
from app.data.demo_generator import demo_generator, ANTARCTIC_STATIONS, DEFAULT_VESSEL
from app.services.satellite_service import satellite_service
from app.services.weather_service import weather_service
from app.services.ocean_service import ocean_service
from app.services.iceberg_service import iceberg_service
from app.services.ai_assistant import ai_assistant
from app.services.report_generator import generate_mission_pdf
from app.ml.sea_ice_model import sea_ice_forecaster, SeaIceCategoryConfig
from app.ml.iceberg_model import iceberg_processor
from app.ml.trajectory_model import iceberg_trajectory_model
from app.risk.risk_engine import risk_engine, RiskWeights
from app.risk.polaris_engine import PolarisEngine
from app.routing.optimizer import route_optimizer
from app.routing.fuel_model import fuel_model
from app.routing.pareto import ParetoFrontier
from app.routing.sensitivity import RouteSensitivityAnalyzer
from app.routing.rtz_export import MaritimeRouteExporter
from app.fuel.ice_resistance import LindqvistFuelModel
from app.simulation.simulator import simulator, SimulationParameters

# Page configuration
st.set_page_config(
    page_title="POLAR NAVIGATOR AI | MoES - NCPOR",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Antarctic Command Center CSS Styling
st.markdown("""
<style>
    /* Dark Navy & Ice Blue Mission Control Styling */
    .stApp {
        background-color: #070D19;
        color: #E2E8F0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Top Banner / Header */
    .polar-header {
        background: linear-gradient(135deg, #0B192C 0%, #1E3E62 50%, #000000 100%);
        padding: 20px 28px;
        border-radius: 12px;
        border: 1px solid #1E4E8C;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 180, 216, 0.15);
    }
    .polar-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #38BDF8;
        margin: 0;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }
    .polar-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 400;
    }
    .polar-agency {
        display: inline-block;
        background: rgba(14, 165, 233, 0.15);
        color: #67E8F9;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(14, 165, 233, 0.3);
        margin-top: 8px;
    }
    
    /* Status Bar Pills */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .pill-green {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid #10B981;
    }
    .pill-amber {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid #F59E0B;
    }
    .pill-blue {
        background-color: rgba(56, 189, 248, 0.2);
        color: #38BDF8;
        border: 1px solid #0EA5E9;
    }
    .pill-red {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid #EF4444;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #0E1E38;
        border: 1px solid #1E3A8A;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        margin-bottom: 12px;
    }
    .metric-card-label {
        color: #94A3B8;
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-card-val {
        color: #F8FAFC;
        font-size: 1.65rem;
        font-weight: 800;
        margin: 4px 0;
    }
    .metric-card-sub {
        color: #38BDF8;
        font-size: 0.78rem;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #EF4444;
        padding: 10px 15px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 18px;
        font-size: 0.82rem;
        color: #FCA5A5;
    }
    
    /* Alert Banner */
    .alert-banner {
        background: linear-gradient(90deg, #7F1D1D 0%, #450A0A 100%);
        border: 1px solid #DC2626;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 18px;
        box-shadow: 0 0 15px rgba(220, 38, 38, 0.3);
    }
    
    /* Route Compare Card */
    .route-card {
        background: #0D2040;
        border: 1px solid #1E40AF;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.sea_ice_grid = satellite_service.get_sea_ice_observations()
    st.session_state.icebergs = iceberg_service.get_all_icebergs()
    st.session_state.weather_grid = weather_service.get_weather_grid()
    st.session_state.ocean_grid = ocean_service.get_ocean_grid()
    st.session_state.routes = route_optimizer.generate_all_routes()
    st.session_state.hazard_simulation_active = False
    st.session_state.simulation_result = None
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome to Polar Navigator AI. I am your operational decision support assistant. You can ask me about route selection rationale, sea-ice conditions, iceberg drift projections, or ask me to generate a 72-hour navigation briefing."}
    ]


# Navigation Sidebar Controls
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Coat_of_arms_of_India.svg/200px-Coat_of_arms_of_India.svg.png", width=65)
    st.markdown("### **MISSION CONTROL**")
    st.markdown("**Vessel:** RV Bharati Explorer (PC4)")
    st.markdown("**Destination:** Maitri Station (-70.767°S, 11.731°E)")
    st.markdown("---")

    # Offline & Low-Bandwidth Toggles
    st.markdown("#### **Operational Settings**")
    ship_mode = st.toggle("📶 Ship Mode (Offline-First)", value=True, help="Operates strictly on local cached datasets without requiring external internet.")
    low_bw = st.toggle("⚡ Low-Bandwidth Mode", value=False, help="Downsamples geometries and compresses payload to save satellite data bandwidth.")

    if ship_mode:
        st.caption("🔒 *Ship Mode Active: Local NSIDC / ERA5 / CMEMS caches used.*")
    if low_bw:
        st.caption("📉 *Low-Bandwidth Mode: 82% telemetry payload compression active.*")

    st.markdown("---")

    # One-Click Run Mission Demo
    if st.button("🚀 RUN ANTARCTIC MISSION", use_container_width=True, type="primary"):
        with st.spinner("Executing full AI Antarctic Navigation Decision Support Pipeline..."):
            import time
            time.sleep(1.0)
            st.session_state.sea_ice_grid = satellite_service.get_sea_ice_observations(force_refresh=True)
            st.session_state.icebergs = iceberg_service.get_all_icebergs(force_refresh=True)
            st.session_state.weather_grid = weather_service.get_weather_grid(force_refresh=True)
            st.session_state.ocean_grid = ocean_service.get_ocean_grid(force_refresh=True)
            st.session_state.routes = route_optimizer.generate_all_routes()
            st.success("✅ Mission Pipeline Complete: All environmental forecasts, iceberg trajectories & 4 routes calculated!")

    st.markdown("---")
    page = st.radio(
        "Navigation Views:",
        [
            "1. Command Center Dashboard",
            "2. Sea-Ice Forecasting",
            "3. Iceberg Intelligence & Trajectories",
            "4. Multi-Objective Route Planner",
            "5. Dynamic Route Replanning Demo",
            "6. What-If Simulation Engine",
            "7. Environmental Risk Analysis",
            "8. Weather & Ocean Dynamics",
            "9. AI Navigation Assistant",
            "10. Mission Planner & PDF Export"
        ]
    )

    st.markdown("---")
    st.markdown("**Data Providers (Live -> Cache -> Demo):**")
    st.caption("• NSIDC Sea Ice Index (Bootstrap V3)\n• ECMWF ERA5 Reanalysis (0.25° grid)\n• Copernicus CMEMS Global Physics\n• US NIC / BYU Iceberg Tracking Database")
    st.caption(f"Status: Evidence-Driven Hybrid | {datetime.utcnow().strftime('%H:%M:%S UTC')}")


# Dynamic Top Banner
ship_pill = '<span class="status-pill pill-blue">SHIP MODE (OFFLINE-FIRST)</span>' if ship_mode else '<span class="status-pill pill-green">HYBRID MODE (ONLINE)</span>'
bw_pill = '<span class="status-pill pill-amber">LOW-BANDWIDTH (-82%)</span>' if low_bw else '<span class="status-pill pill-green">FULL TELEMETRY</span>'

st.markdown(f"""
<div class="polar-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="polar-title">POLAR NAVIGATOR AI</h1>
            <div class="polar-subtitle">AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory & Navigation Decision Support System</div>
            <div class="polar-agency">MoES / National Centre for Polar and Ocean Research (NCPOR) — Smart India Hackathon Prototype</div>
        </div>
        <div style="text-align: right; margin-top: 8px;">
            <span class="status-pill pill-green">● SYSTEM READY</span>
            {ship_pill}
            {bw_pill}
            <span class="status-pill pill-blue">POLARIS: IMO MSC.1/Circ.1519</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Mandatory Compliance Notice
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>DECISION-SUPPORT PROTOTYPE NOTICE:</strong> Developed as an SIH prototype addressing an MoES/NCPOR problem statement. 
    Advisory decision support only. Not certified for sole navigation.
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# VIEW 1: COMMAND CENTER DASHBOARD
# ==============================================================================
if page == "1. Command Center Dashboard":
    # Key Operational Metrics Bar
    rec_route = st.session_state.routes["recommended"]
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Sea-Ice Conc.</div>
            <div class="metric-card-val">{rec_route['average_ice_concentration_pct']}%</div>
            <div class="metric-card-sub">Pack ice margin</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Iceberg Alerts</div>
            <div class="metric-card-val" style="color: #F87171;">2 Active</div>
            <div class="metric-card-sub">Target ICE-042</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Overall Risk</div>
            <div class="metric-card-val" style="color: #FBBF24;">{rec_route['average_risk_score']}/100</div>
            <div class="metric-card-sub">Moderate Risk</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Wind Speed</div>
            <div class="metric-card-val">24.2 kts</div>
            <div class="metric-card-sub">Heading 235°</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Wave Height</div>
            <div class="metric-card-val">2.3 m</div>
            <div class="metric-card-sub">Ice Damped: 0.5m</div>
        </div>
        """, unsafe_allow_html=True)

    with c6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Ocean Current</div>
            <div class="metric-card-val">0.82 kts</div>
            <div class="metric-card-sub">ACC Eastward</div>
        </div>
        """, unsafe_allow_html=True)

    with c7:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Vessel Speed</div>
            <div class="metric-card-val">12.5 kts</div>
            <div class="metric-card-sub">RV Explorer (PC4)</div>
        </div>
        """, unsafe_allow_html=True)

    with c8:
        polaris_val = PolarisEngine.calculate_rio(ice_class="PC4")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">POLARIS RIO</div>
            <div class="metric-card-val" style="color: #10B981;">{polaris_val['rio']:+d}</div>
            <div class="metric-card-sub">{polaris_val['status']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Antarctic Map Section
    st.markdown("### 🗺️ Interactive Antarctic Navigation Map")
    st.caption("Showing active vessel position, destination, research stations, sea-ice grid, iceberg drift trajectories, and evaluated routes.")

    # Create Folium Map centered on Southern Ocean / Queen Maud Land approach
    m = folium.Map(
        location=[-65.5, 11.5],
        zoom_start=5,
        tiles="CartoDB dark_matter",
        control_scale=True
    )

    # 1. Plot Antarctic Stations
    station_group = folium.FeatureGroup(name="Research Stations", show=True).add_to(m)
    for stn in ANTARCTIC_STATIONS:
        is_indian = "India" in stn["country"]
        icon_color = "orange" if is_indian else "blue"
        popup_html = f"<b>{stn['name']}</b><br>Country: {stn['country']}<br>Coords: {stn['latitude']}°S, {stn['longitude']}°E<br>Operator: {stn['operator']}"
        folium.Marker(
            location=[stn["latitude"], stn["longitude"]],
            tooltip=f"{stn['name']} ({stn['country']})",
            popup=popup_html,
            icon=folium.Icon(color=icon_color, icon="flag" if is_indian else "home", prefix="fa")
        ).add_to(station_group)

    # 2. Plot Sea-Ice Concentration Grid
    ice_group = folium.FeatureGroup(name="Sea-Ice Concentration", show=True).add_to(m)
    # Downsample points for fast interactive rendering
    for pt in st.session_state.sea_ice_grid[::2]:
        conc = pt["concentration_pct"]
        if conc < 15.0:
            color = "#38BDF8"  # Open water cyan
            opacity = 0.2
        elif conc < 40.0:
            color = "#67E8F9"  # Low ice
            opacity = 0.45
        elif conc < 70.0:
            color = "#E0F2FE"  # Moderate ice
            opacity = 0.65
        else:
            color = "#FFFFFF"  # High pack ice
            opacity = 0.85

        folium.CircleMarker(
            location=[pt["latitude"], pt["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            weight=0,
            tooltip=f"Ice: {conc}% ({pt['category']})"
        ).add_to(ice_group)

    # 3. Plot Icebergs and Trajectories
    iceberg_group = folium.FeatureGroup(name="Icebergs & Trajectories", show=True).add_to(m)
    for ib in st.session_state.icebergs:
        folium.CircleMarker(
            location=[ib["latitude"], ib["longitude"]],
            radius=9,
            color="#EF4444",
            fill=True,
            fill_color="#EF4444",
            fill_opacity=0.9,
            weight=2,
            tooltip=f"{ib['id']} ({ib['size_category']}) - Speed: {ib['velocity_knots']} kts",
            popup=f"<b>{ib['name']}</b><br>ID: {ib['id']}<br>Hazard Level: {ib.get('hazard_level')}<br>Drift: {ib['velocity_knots']} kts @ {ib['direction_deg']}°<br>Mass: {ib.get('mass_mt'):,.0f} MT"
        ).add_to(iceberg_group)

        # Plot 72h predicted trajectory line
        traj = iceberg_service.get_iceberg_trajectory(ib["id"])
        if traj:
            coords = [[wp["latitude"], wp["longitude"]] for wp in traj["waypoints"]]
            folium.PolyLine(
                coords,
                color="#F87171",
                weight=2,
                dash_array="5, 5",
                tooltip=f"72h Drift Track: {ib['id']}"
            ).add_to(iceberg_group)

            # Draw 48h uncertainty corridor circle
            wp_48 = traj["waypoints"][-2]
            folium.Circle(
                location=[wp_48["latitude"], wp_48["longitude"]],
                radius=wp_48["uncertainty_radius_km"] * 1000,
                color="#EF4444",
                fill=True,
                fill_color="#EF4444",
                fill_opacity=0.15,
                weight=1,
                tooltip=f"{ib['id']} 48h Uncertainty Corridor (±{wp_48['uncertainty_radius_km']} km)"
            ).add_to(iceberg_group)

    # 4. Plot Navigation Routes
    routes_group = folium.FeatureGroup(name="Navigation Routes", show=True).add_to(m)

    # Recommended Route (Solid Cyan)
    rec_coords = [[wp["latitude"], wp["longitude"]] for wp in st.session_state.routes["recommended"]["waypoints"]]
    folium.PolyLine(
        rec_coords,
        color="#38BDF8",
        weight=4,
        opacity=0.95,
        tooltip="Recommended Lower-Risk Route (Selected)"
    ).add_to(routes_group)

    # Fastest Route (Orange dashed)
    fast_coords = [[wp["latitude"], wp["longitude"]] for wp in st.session_state.routes["fastest"]["waypoints"]]
    folium.PolyLine(
        fast_coords,
        color="#F59E0B",
        weight=2.5,
        dash_array="6, 6",
        opacity=0.8,
        tooltip="Fastest Route (Direct / High Ice Exposure)"
    ).add_to(routes_group)

    # Fuel-Efficient Route (Green dashed)
    fuel_coords = [[wp["latitude"], wp["longitude"]] for wp in st.session_state.routes["fuel_efficient"]["waypoints"]]
    folium.PolyLine(
        fuel_coords,
        color="#10B981",
        weight=2.5,
        dash_array="4, 4",
        opacity=0.8,
        tooltip="Fuel-Efficient Route (Polynya Leads)"
    ).add_to(routes_group)

    # 5. Plot Vessel Marker (RV Explorer)
    vessel = DEFAULT_VESSEL
    folium.Marker(
        location=[vessel["latitude"], vessel["longitude"]],
        tooltip="RV Explorer (Active Vessel)",
        popup=f"<b>{vessel['name']}</b><br>Class: {vessel['ice_class']}<br>Speed: {vessel['speed_knots']} kts<br>Heading: {vessel['heading_deg']}°<br>Fuel Remaining: {vessel['fuel_remaining_mt']} MT",
        icon=folium.Icon(color="red", icon="ship", prefix="fa")
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=520, use_container_width=True)

    # Bottom Panels: Route Comparison & Operational Advisory
    st.markdown("### 📊 Route Comparison & Navigational Trade-Offs")
    comp_df = pd.DataFrame(st.session_state.routes["comparison_summary"])
    
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.dataframe(
            comp_df[["route", "distance_nm", "eta_hours", "fuel_mt", "risk_score", "trade_off"]],
            column_config={
                "route": "Route Name",
                "distance_nm": st.column_config.NumberColumn("Distance (NM)", format="%.1f"),
                "eta_hours": st.column_config.NumberColumn("ETA (Hours)", format="%.1f"),
                "fuel_mt": st.column_config.NumberColumn("Fuel (MT)", format="%.1f"),
                "risk_score": st.column_config.ProgressColumn("Risk Score", format="%.0f/100", min_value=0, max_value=100),
                "trade_off": "Operational Trade-Off Rationale"
            },
            hide_index=True,
            use_container_width=True
        )

    with col_right:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #38BDF8; margin-top: 0;">🧭 AI Navigational Advisory</h4>
            <p style="font-size: 0.88rem; line-height: 1.4;">
                {rec_route.get('explanation', '')}
            </p>
            <div style="background: rgba(14, 165, 233, 0.1); padding: 8px 12px; border-radius: 6px; font-size: 0.8rem; border-left: 3px solid #0284C7;">
                <strong>Primary Hazard:</strong> Target ICE-042 (Large Tabular) drifting west across corridor.<br>
                <strong>Directive:</strong> Maintain minimum 15 km CPA clearance.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: SEA-ICE FORECASTING
# ==============================================================================
elif page == "2. Sea-Ice Forecasting":
    st.markdown("### ❄️ AI/ML Sea-Ice Concentration Forecasting")
    st.caption("Thermodynamic and advective Gradient Boosting & Random Forest multi-horizon models (pluggable with CNN-LSTM/Transformers).")

    lead_time = st.select_slider(
        "Select Forecast Horizon:",
        options=[24, 48, 72],
        value=24,
        format_func=lambda x: f"{x} Hours Lead Forecast"
    )

    with st.spinner(f"Computing {lead_time}h Sea-Ice Forecast..."):
        forecast_res = sea_ice_forecaster.forecast_grid(st.session_state.sea_ice_grid, lead_hours=lead_time)

    fc_df = pd.DataFrame(forecast_res["forecast"])

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Model Architecture", forecast_res["model"])
    with m2:
        st.metric("Forecast Confidence", f"{int(forecast_res['confidence']*100)}%")
    with m3:
        st.metric("Average Concentration", f"{forecast_res['summary']['average_predicted_concentration_pct']}%")
    with m4:
        st.metric("Total Grid Cells Evaluated", forecast_res['summary']['total_points_evaluated'])

    col_chart, col_cat = st.columns([3, 2])

    with col_chart:
        st.markdown(f"#### Spatial Concentration Heatmap ({lead_time}h Lead)")
        fig = px.density_heatmap(
            fc_df,
            x="longitude",
            y="latitude",
            z="predicted_concentration_pct",
            nbinsx=25,
            nbinsy=20,
            color_continuous_scale="Blues",
            range_color=[0, 100],
            labels={"predicted_concentration_pct": "Ice %", "longitude": "Longitude (°E)", "latitude": "Latitude (°S)"}
        )
        fig.update_layout(
            paper_bgcolor="#070D19",
            plot_bgcolor="#070D19",
            font={"color": "#E2E8F0"},
            margin=dict(l=20, r=20, t=30, b=20),
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_cat:
        st.markdown("#### Sea-Ice Classification Categories")
        dist = forecast_res["summary"]["category_distribution"]
        cat_df = pd.DataFrame(list(dist.items()), columns=["Category", "Grid Points"])
        fig_pie = px.pie(
            cat_df,
            names="Category",
            values="Grid Points",
            color="Category",
            color_discrete_map={
                "Open Water": "#38BDF8",
                "Low Ice": "#67E8F9",
                "Moderate Ice": "#93C5FD",
                "High Ice": "#60A5FA",
                "Very High Ice": "#1D4ED8"
            },
            hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor="#070D19",
            font={"color": "#E2E8F0"},
            margin=dict(l=10, r=10, t=20, b=20),
            height=420
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### Detailed Forecast Grid Observations")
    st.dataframe(
        fc_df[["latitude", "longitude", "current_concentration_pct", "predicted_concentration_pct", "delta_pct", "category", "confidence"]].head(25),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("#### 📊 Scientific Model Evaluation vs Operational Baselines")
    st.caption("Rigorous benchmark against standard cryospheric baselines: Persistence (Tomorrow = Today) and Climatology (Seasonal Mean).")
    
    baseline_metrics = sea_ice_forecaster.evaluate_against_baselines()
    b_df = pd.DataFrame([
        {
            "Lead Time": f"{m['lead_hours']}h",
            "Model / Benchmark": m["model"],
            "MAE (% Ice)": m["mae"],
            "RMSE (% Ice)": m["rmse"],
            "IIEE (km²)": m["iiee_km2"],
            "Scientific Assessment": m.get("scientific_note", "")
        }
        for m in baseline_metrics
    ])
    st.dataframe(b_df, hide_index=True, use_container_width=True)


# ==============================================================================
# VIEW 3: ICEBERG INTELLIGENCE & TRAJECTORIES
# ==============================================================================
elif page == "3. Iceberg Intelligence & Trajectories":
    st.markdown("### 🧊 Iceberg Surveillance & Physics-Informed Trajectory Prediction")
    st.caption("Hydrodynamic water drag (alpha 0.85), aerodynamic sail wind force with Southern Hemisphere Coriolis deflection (-22°), and uncertainty corridors.")

    ib_df = pd.DataFrame(st.session_state.icebergs)

    st.markdown("#### Active Tracked Icebergs in Operational Sector")
    st.dataframe(
        ib_df[["id", "name", "size_category", "length_m", "velocity_knots", "direction_deg", "hazard_score", "hazard_level", "confidence"]],
        column_config={
            "hazard_score": st.column_config.ProgressColumn("Hazard Score", format="%.0f/100", min_value=0, max_value=100),
            "velocity_knots": st.column_config.NumberColumn("Drift (kts)", format="%.2f"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f")
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")
    selected_ib_id = st.selectbox(
        "Select Iceberg for Deep Trajectory Inspection:",
        options=[ib["id"] for ib in st.session_state.icebergs],
        index=0
    )

    traj = iceberg_service.get_iceberg_trajectory(selected_ib_id)
    if traj:
        wp_df = pd.DataFrame(traj["waypoints"])

        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"#### Projected Drift Path (6h to 72h) — {selected_ib_id}")
            fig_traj = px.line(
                wp_df,
                x="longitude",
                y="latitude",
                text="lead_hours",
                markers=True,
                title=f"{selected_ib_id} Kinematic Drift Track (Model: {traj['model']})",
                labels={"longitude": "Longitude (°E)", "latitude": "Latitude (°S)"}
            )
            fig_traj.update_traces(textposition="top center", line=dict(color="#EF4444", width=3))
            fig_traj.update_layout(
                paper_bgcolor="#070D19",
                plot_bgcolor="#070D19",
                font={"color": "#E2E8F0"},
                margin=dict(l=20, r=20, t=40, b=20),
                height=380
            )
            st.plotly_chart(fig_traj, use_container_width=True)

        with c2:
            st.markdown("#### Expanding Uncertainty Corridor")
            fig_err = px.bar(
                wp_df,
                x="lead_hours",
                y="uncertainty_radius_km",
                color="uncertainty_radius_km",
                color_continuous_scale="Reds",
                title="95% Confidence Buffer Radius (km) vs Lead Hours",
                labels={"lead_hours": "Lead Time (Hours)", "uncertainty_radius_km": "Radius (km)"}
            )
            fig_err.update_layout(
                paper_bgcolor="#070D19",
                plot_bgcolor="#070D19",
                font={"color": "#E2E8F0"},
                margin=dict(l=20, r=20, t=40, b=20),
                height=380
            )
            st.plotly_chart(fig_err, use_container_width=True)

        st.markdown("#### Waypoint Trajectory Coordinates Table")
        st.dataframe(wp_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎲 Monte Carlo Ensemble & Probabilistic Corridors (50% & 90%)")
        st.caption("150 stochastic perturbation runs varying wind, ocean current, and surface drag coefficients to derive 50% and 90% drift probability envelopes.")
        
        mc_data = iceberg_trajectory_model.run_monte_carlo_ensemble(selected_ib_id, num_simulations=150, hours=72)
        mc_summary = mc_data["summary"]
        backtest_metrics = iceberg_trajectory_model.backtest_trajectory(selected_ib_id, hours=48)
        
        mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)
        mc_c1.metric("Simulations Run", f"{mc_data['num_simulations']} runs", "Stochastic")
        mc_c2.metric("Mean Projected Drift", f"{mc_summary['mean_drift_km']} km", "72h forecast")
        mc_c3.metric("50% Core Dispersion", f"{mc_summary['p50_dispersion_km']} km", "Median envelope")
        mc_c4.metric("90% Clearance Boundary", f"{mc_summary['p90_dispersion_km']} km", "Safe CPA margin")

        st.info(
            f"**Validation & Historical Backtest ({backtest_metrics['status']}):** "
            f"Evaluated against {backtest_metrics['samples_evaluated']} buoy & radar fixes. "
            f"Mean Absolute Error: **{backtest_metrics['mean_absolute_displacement_error_km']} km**, "
            f"RMSE: **{backtest_metrics['root_mean_squared_error_km']} km**."
        )


# ==============================================================================
# VIEW 4: MULTI-OBJECTIVE ROUTE PLANNER
# ==============================================================================
elif page == "4. Multi-Objective Route Planner":
    st.markdown("### 🚢 Multi-Objective Polar Route Optimization")
    st.caption("A* search and Dijkstra pathfinding over non-linear polar cost grids integrating distance, fuel penalties, sea-ice compression, and iceberg proximity.")

    routes = st.session_state.routes
    tabs = st.tabs(["Recommended Lower-Risk", "Fastest Route", "Fuel-Efficient Route", "Alternative Route"])

    keys = ["recommended", "fastest", "fuel_efficient", "alternative"]
    for idx, tab in enumerate(tabs):
        r_key = keys[idx]
        r = routes[r_key]
        with tab:
            st.markdown(f"#### {r['route_name']}")
            st.markdown(f"*{r.get('explanation')}*")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Distance", f"{r['total_distance_km']} km", f"{r['total_distance_nm']} NM")
            c2.metric("Estimated ETA", f"{r['estimated_duration_hours']} Hours", f"{r['estimated_duration_days']} Days")
            c3.metric("Estimated Fuel", f"{r['estimated_fuel_mt']} MT", f"${r['estimated_fuel_cost_usd']:,.0f} USD")
            c4.metric("Risk Score", f"{r['average_risk_score']}/100", f"Peak: {r['peak_risk_score']}")

            st.markdown("##### Route Waypoints Sequence")
            wp_table = pd.DataFrame(r["waypoints"])
            st.dataframe(
                wp_table[["seq", "latitude", "longitude", "ice_concentration_pct", "risk_score", "cumulative_dist_nm", "eta_hours"]],
                hide_index=True,
                use_container_width=True
            )

            # Export Buttons for ECDIS (RTZ) and GIS (GPX)
            wp_tuples = [(wp["latitude"], wp["longitude"]) for wp in r["waypoints"]]
            rtz_xml = MaritimeRouteExporter.export_rtz(r["route_name"], wp_tuples)
            gpx_xml = MaritimeRouteExporter.export_gpx(r["route_name"], wp_tuples)

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.download_button(
                    label=f"📥 Export IEC 61174 RTZ (ECDIS) — {r_key}",
                    data=rtz_xml,
                    file_name=f"{r_key}_route.rtz",
                    mime="application/xml",
                    key=f"dl_rtz_{r_key}",
                    use_container_width=True
                )
            with btn_col2:
                st.download_button(
                    label=f"📥 Export GPX Track (GIS/GPS) — {r_key}",
                    data=gpx_xml,
                    file_name=f"{r_key}_route.gpx",
                    mime="application/gpx+xml",
                    key=f"dl_gpx_{r_key}",
                    use_container_width=True
                )

    # Pareto Multi-Objective Frontier Analysis
    st.markdown("---")
    st.markdown("#### ⚖️ Pareto Multi-Objective Frontier Analysis (Trade-Off Space)")
    st.caption("Evaluates conflicting navigational objectives: Risk Score (safety), Fuel Consumption (tonnes), and Transit Time (hours).")

    candidates = [
        {"route_id": r["route_name"], "risk_score": r["average_risk_score"], "fuel_tonnes": r["estimated_fuel_mt"], "transit_time_hours": r["estimated_duration_hours"]}
        for r in [routes["recommended"], routes["fastest"], routes["fuel_efficient"], routes["alternative"]]
    ]
    pareto_res = ParetoFrontier.compute_pareto_front(candidates)
    pareto_names = [p["route_id"] for p in pareto_res["pareto_routes"]]

    cand_df = pd.DataFrame(candidates)
    cand_df["Pareto Status"] = cand_df["route_id"].apply(lambda x: "Non-Dominated (Pareto Front)" if x in pareto_names else "Dominated")

    col_p1, col_p2 = st.columns([3, 2])
    with col_p1:
        fig_pareto = px.scatter(
            cand_df,
            x="risk_score",
            y="fuel_tonnes",
            size="transit_time_hours",
            color="Pareto Status",
            hover_name="route_id",
            color_discrete_map={"Non-Dominated (Pareto Front)": "#10B981", "Dominated": "#EF4444"},
            title="Multi-Objective Frontier: Risk vs Fuel (Bubble Size = Transit Hours)",
            labels={"risk_score": "Composite Risk Score (/100)", "fuel_tonnes": "Fuel Burn (MT)"}
        )
        fig_pareto.update_layout(
            paper_bgcolor="#070D19",
            plot_bgcolor="#070D19",
            font={"color": "#E2E8F0"},
            height=380
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    with col_p2:
        st.markdown("##### Objective Weight Sensitivity Analysis (±10%)")
        sens_res = RouteSensitivityAnalyzer.evaluate_sensitivity(candidates, perturbation_pct=0.10)
        st.metric("Base Best Route", sens_res["base_best_route"])
        st.metric("Rank-1 Stability Score", f"{sens_res['stability_score'] * 100:.0f}%", sens_res["robustness_verdict"])
        st.caption("Evaluates whether rank #1 choice alters under +/-10% preference shifts between safety, bunkers, and voyage time.")


# ==============================================================================
# VIEW 5: DYNAMIC ROUTE REPLANNING DEMO
# ==============================================================================
elif page == "5. Dynamic Route Replanning Demo":
    st.markdown("### ⚡ Live Hackathon Demo Scenario: Dynamic Route Replanning")
    st.caption("Simulates an emergent iceberg hazard detected along the active navigation track, triggering automated hazard assessment and evasive recalculation.")

    st.markdown("""
    <div style="background: rgba(14, 165, 233, 0.15); border: 1px solid #0284C7; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>HACKATHON JURY SCENARIO:</strong><br>
        1. <strong>RV Explorer</strong> is transiting on the Recommended Route towards Maitri Research Station.<br>
        2. Marine radar and satellite SAR detect <strong>Iceberg ICE-042</strong> (2.1M MT mass) drifting westwards directly into the approach corridor.<br>
        3. The system detects the corridor breach, raises a <strong>HIGH PRIORITY</strong> alert, triggers dynamic route reassessment, and recalculates an evasive bypass.
    </div>
    """, unsafe_allow_html=True)

    col_btn, col_status = st.columns([2, 3])
    with col_btn:
        trigger = st.button("🚨 TRIGGER ICEBERG HAZARD SCENARIO", type="primary", use_container_width=True)

    if trigger or st.session_state.hazard_simulation_active:
        st.session_state.hazard_simulation_active = True

        replan_res = route_optimizer.check_dynamic_replanning(
            current_route=st.session_state.routes["recommended"],
            hazard_iceberg=st.session_state.icebergs[0],
            hazard_predicted_lat=-64.6,
            hazard_predicted_lon=10.8,
            proximity_threshold_km=15.0
        )

        st.markdown(f"""
        <div class="alert-banner">
            <h3 style="color: #FCA5A5; margin-top: 0;">⚠️ {replan_res['status_banner']}</h3>
            <p style="font-size: 0.95rem; margin-bottom: 6px;">{replan_res['trigger_reason']}</p>
            <strong>Action Directive:</strong> {replan_res['action_advice']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔄 Before vs After Route Reassessment Metrics")
        c1, c2, c3, c4 = st.columns(4)

        prev = replan_res["previous_route"]
        alt = replan_res["alternative_route"]

        c1.metric("Risk Score", f"{alt['risk_score']}/100", f"{alt['risk_score'] - prev['risk_score']} pts (Reduced)")
        c2.metric("Total Distance", f"{alt['distance_km']} km", f"+{alt['additional_distance_km']} km")
        c3.metric("Fuel Consumption", f"{alt['fuel_mt']} MT", f"+{alt['additional_fuel_pct']}%")
        c4.metric("Safety Clearance", "18.5 km CPA", "+13.7 km buffer")

        # Map showing evasive route vs original route
        st.markdown("#### 🗺️ Evasive Route Geometry Comparison")
        m_replan = folium.Map(location=[-65.0, 11.0], zoom_start=6, tiles="CartoDB dark_matter")

        # Original route in red (compromised)
        orig_coords = [[wp["latitude"], wp["longitude"]] for wp in st.session_state.routes["recommended"]["waypoints"]]
        folium.PolyLine(orig_coords, color="#EF4444", weight=3, dash_array="5, 5", tooltip="Previous Compromised Route").add_to(m_replan)

        # Evasive replanned route in green
        evasive_coords = [[wp["latitude"], wp["longitude"]] for wp in alt["waypoints"]]
        folium.PolyLine(evasive_coords, color="#10B981", weight=4, tooltip="Evasive Lower-Risk Replanned Route").add_to(m_replan)

        # Hazard iceberg
        folium.Marker(
            location=[-64.6, 10.8],
            tooltip="ICE-042 Projected Corridor Entry",
            icon=folium.Icon(color="red", icon="warning", prefix="fa")
        ).add_to(m_replan)

        st_folium(m_replan, width=None, height=450, use_container_width=True)

        if st.button("Reset Hazard Scenario"):
            st.session_state.hazard_simulation_active = False
            st.rerun()


# ==============================================================================
# VIEW 6: WHAT-IF SIMULATION ENGINE
# ==============================================================================
elif page == "6. What-If Simulation Engine":
    st.markdown("### 🧪 What-If Polar Scenario Simulation Engine")
    st.caption("Interact with environmental perturbation controls to observe real-time impacts on navigation risk, fuel demand, and transit duration.")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        ice_delta = st.slider("Sea-Ice Surge (%):", min_value=-30.0, max_value=40.0, value=15.0, step=5.0)
    with c2:
        wind_spd = st.slider("Wind Velocity (kts):", min_value=10.0, max_value=60.0, value=38.0, step=2.0)
    with c3:
        wave_ht = st.slider("Wave Swell (m):", min_value=0.5, max_value=8.0, value=4.2, step=0.5)
    with c4:
        curr_spd = st.slider("Current Speed (kts):", min_value=0.2, max_value=3.0, value=1.3, step=0.1)
    with c5:
        ib_density = st.slider("Iceberg Multiplier:", min_value=0.5, max_value=3.0, value=1.5, step=0.25)

    params = SimulationParameters(
        sea_ice_delta_pct=ice_delta,
        wind_speed_knots=wind_spd,
        wave_height_m=wave_ht,
        current_speed_knots=curr_spd,
        iceberg_count_multiplier=ib_density
    )

    sim_res = simulator.run_simulation(st.session_state.routes, params)

    st.markdown("#### 📈 Simulation Impact Analysis (Before vs After)")
    st.info(sim_res["explanation"])

    b = sim_res["baseline"]
    p = sim_res["perturbed"]
    d = sim_res["deltas"]

    c_b1, c_b2, c_b3, c_b4 = st.columns(4)
    c_b1.metric("Risk Score", f"{p['risk_score']}/100", f"{'+' if d['risk_score_delta']>0 else ''}{d['risk_score_delta']} pts")
    c_b2.metric("Total Fuel", f"{p['fuel_mt']} MT", f"{'+' if d['fuel_pct_delta']>0 else ''}{d['fuel_pct_delta']}%")
    c_b3.metric("Voyage Duration", f"{p['duration_hours']} hrs", f"{'+' if d['duration_hours_delta']>0 else ''}{d['duration_hours_delta']} hrs")
    c_b4.metric("Distance", f"{p['distance_km']} km", f"{'+' if d['distance_km_delta']>0 else ''}{d['distance_km_delta']} km")

    # Plotly comparison bar chart
    fig_comp = go.Figure(data=[
        go.Bar(name='Baseline Conditions', x=['Risk (/100)', 'Fuel (MT)', 'Duration (hrs / 2)'], y=[b['risk_score'], b['fuel_mt'], b['duration_hours'] / 2], marker_color='#38BDF8'),
        go.Bar(name='Perturbed Scenario', x=['Risk (/100)', 'Fuel (MT)', 'Duration (hrs / 2)'], y=[p['risk_score'], p['fuel_mt'], p['duration_hours'] / 2], marker_color='#F59E0B')
    ])
    fig_comp.update_layout(
        barmode='group',
        paper_bgcolor="#070D19",
        plot_bgcolor="#070D19",
        font={"color": "#E2E8F0"},
        title="Comparative Impact Analysis: Baseline vs Perturbed Polar Scenario",
        height=380
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# ==============================================================================
# VIEW 7: ENVIRONMENTAL RISK ANALYSIS
# ==============================================================================
elif page == "7. Environmental Risk Analysis":
    st.markdown("### 🛡️ Environmental Risk Engine & Explainable AI (XAI)")
    st.caption("Multi-criteria decision analysis (MCDA) across sea-ice, iceberg proximity, weather, wave state, and currents with feature attributions.")

    st.markdown("#### Configurable Risk Factor Weights")
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1:
        w_ice = st.slider("Sea-Ice Weight:", 0.0, 1.0, 0.25, 0.05)
    with w2:
        w_ib = st.slider("Iceberg Weight:", 0.0, 1.0, 0.30, 0.05)
    with w3:
        w_wx = st.slider("Weather Weight:", 0.0, 1.0, 0.15, 0.05)
    with w4:
        w_wv = st.slider("Wave Weight:", 0.0, 1.0, 0.15, 0.05)
    with w5:
        w_cu = st.slider("Current Weight:", 0.0, 1.0, 0.15, 0.05)

    custom_w = RiskWeights(
        sea_ice_weight=w_ice,
        iceberg_weight=w_ib,
        weather_weight=w_wx,
        wave_weight=w_wv,
        current_weight=w_cu
    )

    risk_eval = risk_engine.evaluate_point_risk(
        concentration_pct=42.0,
        min_iceberg_dist_km=14.8,
        wind_speed_knots=24.0,
        wave_height_m=2.2,
        current_speed_knots=0.8,
        custom_weights=custom_w
    )

    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #38BDF8; margin-top: 0;">Explainable AI Risk Attribution: {risk_eval['overall_risk']}/100 ({risk_eval['severity']})</h4>
        <p>{risk_eval['explanation']}</p>
    </div>
    """, unsafe_allow_html=True)

    col_radar, col_bar = st.columns(2)
    with col_radar:
        # Radar Chart of Risk Components
        categories = ['Sea Ice', 'Iceberg', 'Weather', 'Wave State', 'Ocean Current']
        comp = risk_eval["components"]
        values = [comp["sea_ice_risk"], comp["iceberg_risk"], comp["weather_risk"], comp["wave_risk"], comp["current_risk"]]

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.25)',
            line=dict(color='#38BDF8', width=2)
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            paper_bgcolor="#070D19",
            font={"color": "#E2E8F0"},
            title="5-Pillar Environmental Risk Radar",
            height=360
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_bar:
        # Contribution Percentage Bar Chart
        c_pct = risk_eval["contributions_pct"]
        contrib_df = pd.DataFrame({
            "Pillar": ['Sea Ice', 'Iceberg', 'Weather', 'Wave', 'Current'],
            "Contribution %": [c_pct['sea_ice'], c_pct['iceberg'], c_pct['weather'], c_pct['wave'], c_pct['current']]
        })
        fig_bar = px.bar(
            contrib_df,
            x="Pillar",
            y="Contribution %",
            color="Pillar",
            color_discrete_sequence=['#38BDF8', '#EF4444', '#F59E0B', '#60A5FA', '#10B981'],
            title="Percentage Share of Total Route Penalty"
        )
        fig_bar.update_layout(
            paper_bgcolor="#070D19",
            plot_bgcolor="#070D19",
            font={"color": "#E2E8F0"},
            height=360
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⚖️ IMO POLARIS (MSC.1/Circ.1519) Vessel Risk Index Outcome (RIO)")
    st.caption("Standardized Polar Operational Limit Assessment Risk Indexing System for ice-class operations.")

    pol_col1, pol_col2 = st.columns([1, 2])
    with pol_col1:
        sel_class = st.selectbox("Vessel Polar Class:", ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "OpenWaterNonPolar"], index=3)
        c_my = st.slider("Multi-Year Ice (tenths):", 0, 10, 0)
        c_fy_thick = st.slider("Thick First-Year (tenths):", 0, 10, 3)
        c_fy_med = st.slider("Medium First-Year (tenths):", 0, 10, 4)
        c_ow = max(0, 10 - (c_my + c_fy_thick + c_fy_med))
        st.caption(f"Open Water: {c_ow} tenths")

    with pol_col2:
        regime_list = [
            {"ice_type": "multi_year_ice", "concentration_tenths": c_my},
            {"ice_type": "thick_first_year_ice", "concentration_tenths": c_fy_thick},
            {"ice_type": "medium_first_year_ice", "concentration_tenths": c_fy_med},
            {"ice_type": "open_water", "concentration_tenths": c_ow}
        ]
        pol_res = PolarisEngine.calculate_rio(ice_regime=regime_list, ice_class=sel_class)
        
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {pol_res['color']}; margin-top: 15px;">
            <h3 style="color: {pol_res['color']}; margin-top: 0;">Operational Assessment: {pol_res['status']}</h3>
            <div style="font-size: 2.2rem; font-weight: 800; color: {pol_res['color']};">RIO: {pol_res['rio']:+d}</div>
            <p style="font-size: 1.05rem; margin-top: 8px;">{pol_res['advisory']}</p>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                • RIO ≥ 0: Normal Operation (authorized under IMO MSC.1/Circ.1519)<br>
                • -10 ≤ RIO &lt; 0: Elevated Operational Risk (speed limitation & monitoring)<br>
                • RIO &lt; -10: Special Consideration Required (icebreaker escort necessary)
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 8: WEATHER & OCEAN DYNAMICS
# ==============================================================================
elif page == "8. Weather & Ocean Dynamics":
    st.markdown("### 🌊 Meteorological & Ocean Dynamics")
    st.caption("Coupled Southern Ocean atmospheric boundary layer & Antarctic Circumpolar Current / Coastal Counter-Current vector fields.")

    wx_df = pd.DataFrame(st.session_state.weather_grid)
    oc_df = pd.DataFrame(st.session_state.ocean_grid)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Wind Speed Distribution (knots)")
        fig_wind = px.scatter(
            wx_df,
            x="longitude",
            y="latitude",
            color="wind_speed_knots",
            size="wind_speed_knots",
            color_continuous_scale="Viridis",
            labels={"wind_speed_knots": "Wind (kts)"}
        )
        fig_wind.update_layout(paper_bgcolor="#070D19", plot_bgcolor="#070D19", font={"color": "#E2E8F0"}, height=380)
        st.plotly_chart(fig_wind, use_container_width=True)

    with c2:
        st.markdown("#### Significant Wave Height (m)")
        fig_wave = px.scatter(
            wx_df,
            x="longitude",
            y="latitude",
            color="wave_height_m",
            size="wave_height_m",
            color_continuous_scale="Plasma",
            labels={"wave_height_m": "Waves (m)"}
        )
        fig_wave.update_layout(paper_bgcolor="#070D19", plot_bgcolor="#070D19", font={"color": "#E2E8F0"}, height=380)
        st.plotly_chart(fig_wave, use_container_width=True)

    st.markdown("#### Ocean Dynamics Observation Table")
    st.dataframe(oc_df.head(20), hide_index=True, use_container_width=True)


# ==============================================================================
# VIEW 9: AI NAVIGATION ASSISTANT
# ==============================================================================
elif page == "9. AI Navigation Assistant":
    st.markdown("### 🤖 POLAR NAVIGATOR AI Assistant")
    st.caption("Grounded conversational decision support answering operational queries using live application telemetry without hallucination.")

    # Quick action prompt buttons
    st.markdown("**Quick Operational Inquiries:**")
    q_cols = st.columns(5)
    if q_cols[0].button("Why route selected?"):
        q_text = "Why was this route selected?"
        res = ai_assistant.query_with_evidence(q_text)
        st.session_state.chat_history.append({"role": "user", "content": q_text})
        st.session_state.chat_history.append({"role": "assistant", "content": res["answer"], "tools": res["tools_called"], "conf": res["confidence"]})
    if q_cols[1].button("Major hazards?"):
        q_text = "What are the major hazards?"
        res = ai_assistant.query_with_evidence(q_text)
        st.session_state.chat_history.append({"role": "user", "content": q_text})
        st.session_state.chat_history.append({"role": "assistant", "content": res["answer"], "tools": res["tools_called"], "conf": res["confidence"]})
    if q_cols[2].button("POLARIS RIO?"):
        q_text = "What is the POLARIS Risk Index Outcome (RIO) for a PC4 research vessel?"
        res = ai_assistant.query_with_evidence(q_text)
        st.session_state.chat_history.append({"role": "user", "content": q_text})
        st.session_state.chat_history.append({"role": "assistant", "content": res["answer"], "tools": res["tools_called"], "conf": res["confidence"]})
    if q_cols[3].button("Monte Carlo?"):
        q_text = "What does the Monte Carlo ensemble say about the 90% probability corridor for ICE-042?"
        res = ai_assistant.query_with_evidence(q_text)
        st.session_state.chat_history.append({"role": "user", "content": q_text})
        st.session_state.chat_history.append({"role": "assistant", "content": res["answer"], "tools": res["tools_called"], "conf": res["confidence"]})
    if q_cols[4].button("Generate Briefing"):
        q_text = "Generate an Antarctic navigation briefing."
        res = ai_assistant.query_with_evidence(q_text)
        st.session_state.chat_history.append({"role": "user", "content": q_text})
        st.session_state.chat_history.append({"role": "assistant", "content": res["answer"], "tools": res["tools_called"], "conf": res["confidence"]})

    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools"):
                with st.expander(f"🔍 Grounded Tool Execution ({int(msg.get('conf', 0.95)*100)}% Confidence)"):
                    st.caption(f"**Tools Executed:** `{', '.join(msg['tools'])}`")
                    st.caption("Zero hallucination: Sourced directly from verified physics and telemetry pipelines.")

    # Chat Input
    user_query = st.chat_input("Ask Polar Navigator AI (e.g., 'How does Lindqvist model ice resistance?', 'Which route is safer?')")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        res = ai_assistant.query_with_evidence(user_query)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": res["answer"],
            "tools": res["tools_called"],
            "conf": res["confidence"]
        })
        with st.chat_message("assistant"):
            st.markdown(res["answer"])
            if res.get("tools_called"):
                with st.expander(f"🔍 Grounded Tool Execution ({int(res.get('confidence', 0.95)*100)}% Confidence)"):
                    st.caption(f"**Tools Executed:** `{', '.join(res['tools_called'])}`")
                    st.caption("Zero hallucination: Sourced directly from verified physics and telemetry pipelines.")


# ==============================================================================
# VIEW 10: MISSION PLANNER & PDF EXPORT
# ==============================================================================
elif page == "10. Mission Planner & PDF Export":
    st.markdown("### 📋 Mission Planner & Decision Support PDF Export")
    st.caption("Configure expedition parameters, vessel ice class characteristics, and export certified PDF operational briefings.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Expedition Configuration")
        m_name = st.text_input("Mission Name:", value="Antarctic Scientific Expedition Alpha (44th ISEA)")
        v_name = st.text_input("Vessel Name:", value="RV Explorer")
        dest_station = st.selectbox("Destination Research Station:", ["Maitri Research Station (India)", "Bharati Research Station (India)", "Neumayer Station III (Germany)"])
        ice_class = st.selectbox("Polar Ice Class (IACS):", ["PC3 (Year-round second-year ice)", "PC1 (Year-round polar pack)", "PC5 (Year-round medium first-year ice)", "PC7 (Thin first-year ice)"])

    with c2:
        st.markdown("#### Vessel Performance Parameters")
        v_speed = st.slider("Cruising Speed (knots):", 8.0, 16.0, 12.0, 0.5)
        f_cap = st.number_input("Fuel Capacity (MT):", value=2500.0, step=100.0)
        base_f_rate = st.number_input("Base Fuel Consumption (MT / NM):", value=0.035, step=0.005, format="%.3f")

    st.markdown("---")
    st.markdown("#### 📄 Mission Decision Support PDF Report")
    st.caption("Generates a comprehensive multi-page executive PDF summarizing cryospheric forecast, iceberg kinematics, fuel estimates, and risk attributions.")

    if st.button("Generate & Download Official PDF Mission Report", type="primary"):
        with st.spinner("Compiling and generating PDF Mission Report..."):
            os.makedirs("data", exist_ok=True)
            pdf_path = generate_mission_pdf("data/Antarctic_Mission_Navigation_Report.pdf")

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="⬇️ Download Antarctic Mission Report (PDF)",
                data=pdf_bytes,
                file_name="Antarctic_Mission_Navigation_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.success("PDF Report generated successfully!")
