# System Architecture & Engineering Design
## Polar Navigator AI — MoES / NCPOR Decision Support System

> **Compliance & Legal Notice:**  
> Developed as an SIH prototype addressing an MoES/NCPOR problem statement.  
> Advisory decision support only. Not certified for sole navigation.

---

### 1. High-Level Architecture Overview

Polar Navigator AI is an evidence-driven decision-support system designed for high-latitude Southern Ocean polar expeditions. It integrates satellite earth observation, meteorological reanalysis, hydrodynamic drift mechanics, vessel-specific ice resistance physics, and heuristic multi-objective pathfinding into an offline-first deployment architecture.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                            │
│  Streamlit Command Center (10 Views)  │  FastAPI REST Endpoints (20+)  │
│  - Interactive Map (Folium/GeoJSON)   │  - Open-API / Docs & Health    │
│  - Ship Mode / Low-Bandwidth Toggles  │  - RTZ & GPX Route Exporters   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                       DECISION & ENGINE LAYER                          │
│ ┌──────────────────────────┐  ┌──────────────────────────────────────┐ │
│ │   4D Time-Dependent A*   │  │   POLARIS Risk Engine (Circ.1519)    │ │
│ │  - WGS84 Geodesic Dist   │  │  - RIO = Sum(C_i * RIV_i)            │ │
│ │  - Antarctic Land Mask   │  │  - PC1 to PC7 Regulatory Profiles    │ │
│ └────────────┬─────────────┘  └──────────────────┬───────────────────┘ │
│              │                                   │                     │
│ ┌────────────▼─────────────┐  ┌──────────────────▼───────────────────┐ │
│ │  Pareto Frontier Solver  │  │    Lindqvist (1989) Fuel Model       │ │
│ │  - Risk vs Fuel vs Time  │  │  - Crushing + Breaking + Submersion  │ │
│ │  - +/- 10% Sensitivity   │  │  - SFOC 185 g/kWh Engine Model       │ │
│ └──────────────────────────┘  └──────────────────────────────────────┘ │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                         ANALYTICS & ML LAYER                           │
│ ┌──────────────────────────┐  ┌──────────────────────────────────────┐ │
│ │  Sea-Ice Forecasting ML  │  │     Monte Carlo Iceberg Ensemble     │ │
│ │  - Gradient Boosting/RF  │  │  - 100-300 Stochastic Perturbations  │ │
│ │  - Baseline Benchmark    │  │  - 50% & 90% Probability Corridors   │ │
│ │    (Persistence/Climato) │  │  - Historical Backtesting Error      │ │
│ └──────────────────────────┘  └──────────────────────────────────────┘ │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                    DATA ABSTRACTION & ACCESS LAYER                     │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │       Standardized DataResponse Envelope (Live -> Cached -> Demo)  │ │
│ │  - NSIDC Sea Ice Index (Bootstrap V3 NetCDF)                       │ │
│ │  - Copernicus CDS ERA5 Marine Winds & Temperatures                 │ │
│ │  - CMEMS Southern Ocean Hydrodynamics (Currents & SST)             │ │
│ │  - US NIC / BYU Antarctic Iceberg Tracking Database                │ │
│ └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Core Subsystems

#### 2.1 Provider Abstraction Hierarchy (`Live -> Cached -> Demo`)
To operate reliably in remote polar deployments with intermittent satellite connectivity:
1. **Live:** Fetches latest NetCDF / GRIB / JSON feeds if remote APIs are reachable.
2. **Cached:** Loads previously synchronized files from local disk (`data/cache/`), enabling full **Ship Mode (Offline-First)** without Internet access.
3. **Demo / Simulated:** Synthesizes physically grounded cryospheric patterns when no local cache is available.
4. **Metadata Registry:** Every dataset is stamped with source, spatial resolution, CRS (EPSG:4326), update frequency, license, and citation.

#### 2.2 Vessel-Specific Ice Risk: IMO POLARIS (MSC.1/Circ.1519)
Implements the Polar Operational Limit Assessment Risk Indexing System:
$$\text{RIO} = \sum_{i} \left( C_i \times \text{RIV}_i \right)$$
Where:
- $C_i$: Ice concentration in tenths (0 to 10) for ice development stage $i$.
- $\text{RIV}_i$: Risk Index Value loaded from versioned YAML tables (`config/polaris_riv.yaml`) corresponding to the vessel's IACS Polar Class (PC1 through PC7).

**Operational Categories:**
- $\text{RIO} \ge 0$: **Normal Operation** (Standard polar navigation).
- $-10 \le \text{RIO} < 0$: **Elevated Operational Risk** (Speed reduction, continuous ice watch).
- $\text{RIO} < -10$: **Special Consideration Required** (Operation subject to icebreaker escort or non-entry).

#### 2.3 Monte Carlo Iceberg Trajectory Ensemble
Rather than predicting a single deterministic line, the system models stochastic variability:
- 100 to 300 Monte Carlo runs per target iceberg.
- Gaussian perturbations applied to wind speed ($\pm 15\%$), wind heading ($\pm 12^\circ$), ocean current speed ($\pm 18\%$), current heading ($\pm 12^\circ$), and initial radar fix uncertainty ($\sim 300\text{ m}$).
- Generates **50% median probability core** and **90% clearance boundary polygon** across 24h, 48h, and 72h lead times.

#### 2.4 Time-Dependent Geodesic A* Pathfinder
- Computes true geodesic distances over the WGS84 reference ellipsoid using `pyproj.Geod`.
- Intersects all paths against the continental boundary and ice-shelf mask (`app/geospatial/land_mask.py`).
- 4th dimension (time): calculates dynamic arrival hour $t = t_0 + \Delta t$ at each intermediate waypoint and samples the forecasted ice concentration at that specific arrival time.
- Geodesic line-of-sight path smoothing eliminates zigzag artifacts while preserving land clearance.

#### 2.5 Multi-Objective Pareto Frontier & Sensitivity
Evaluates candidate routes over conflicting metrics:
- **Minimizing Risk:** Composite environmental risk and POLARIS RIO.
- **Minimizing Fuel:** Lindqvist (1989) level ice resistance ($R_{ice} = R_c + R_b + R_s$) + open-water hydrodynamic drag.
- **Minimizing Time:** Distance divided by ice-adjusted vessel transit speed.
- **Sensitivity Analysis:** Perturbs objective weights by $\pm 10\%$ and calculates a Rank-1 Stability Score ($0.0 - 1.0$) to guarantee route selection robustness.

#### 2.6 Maritime Route Interchange (IEC 61174 & GPX)
- **IEC 61174 (Edition 4) RTZ XML:** Industry-standard Route Plan Exchange schema used by Electronic Chart Display and Information Systems (ECDIS).
- **GPX 1.1:** Universal GPS exchange format for bridge hand-helds, chart plotters, and GIS packages.
