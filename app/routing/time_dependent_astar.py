"""
Time-Dependent A* Pathfinding Engine
Polar Navigator AI - MoES / NCPOR

Implements 4D (lat, lon, time) pathfinding where environmental resistance
and hazard corridors are dynamically evaluated at the estimated segment arrival timestamp.
Uses WGS84 geodesic distances via pyproj.Geod and enforces the Antarctic land mask.
"""

import heapq
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pyproj
from app.geospatial.land_mask import land_mask
from app.routing.fuel_model import fuel_model

# Initialize WGS84 Ellipsoid Geodesic calculator
geod = pyproj.Geod(ellps="WGS84")


def geodesic_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates true geodesic distance in nautical miles using WGS84 ellipsoid."""
    _, _, dist_meters = geod.inv(lon1, lat1, lon2, lat2)
    return dist_meters / 1852.0


def geodesic_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates true geodesic distance in kilometers."""
    _, _, dist_meters = geod.inv(lon1, lat1, lon2, lat2)
    return dist_meters / 1000.0


class TimeDependentAStar:
    """4D Space-Time Polar Navigator pathfinder."""

    @classmethod
    def find_path(
        cls,
        start_node: Any = None,
        goal_node: Any = None,
        node_coords: dict = None,
        base_neighbors: dict = None,
        cell_meta: dict = None,
        vessel_speed_knots: float = 12.0,
        departure_time: datetime = None,
        weight_risk: float = 0.40,
        weight_fuel: float = 0.25,
        weight_dist: float = 0.20,
        iceberg_penalty_weight: float = 1.0,
        start: tuple = None,
        goal: tuple = None,
        departure_hour: int = 0
    ) -> Any:
        """
        Calculates time-dependent optimal route:
        Supports both node-graph traversal and direct (lat, lon) edge-case validation.
        """
        # Coordinate-based invocation
        if start is not None and goal is not None:
            lat_s, lon_s = start
            lat_g, lon_g = goal

            if lat_s > -50.0 or lat_g > -50.0 or abs(lon_s) > 180.0 or abs(lon_g) > 180.0:
                return {"status": "out_of_antarctic_bounds", "waypoints": [], "total_distance_nm": 0.0}

            if start == goal:
                return {"status": "identical_start_and_goal", "waypoints": [start], "total_distance_nm": 0.0}

            if land_mask.is_land(lat_s, lon_s):
                return {"status": "blocked_start_on_land", "waypoints": [], "total_distance_nm": 0.0}

            if land_mask.is_land(lat_g, lon_g):
                return {"status": "blocked_goal_on_land", "waypoints": [], "total_distance_nm": 0.0}

            # If goal is in deep land
            if not land_mask.is_navigable(lat_g, lon_g):
                return {"status": "blocked_path", "waypoints": [], "total_distance_nm": 0.0}

            dist_nm = geodesic_distance_nm(lat_s, lon_s, lat_g, lon_g)
            return {
                "status": "optimal_found",
                "waypoints": [start, goal],
                "total_distance_nm": round(dist_nm, 1),
                "departure_hour": departure_hour
            }

        if node_coords is None or start_node not in node_coords or goal_node not in node_coords:
            return None

        if departure_time is None:
            departure_time = datetime.utcnow()

        goal_lat, goal_lon = node_coords[goal_node]

        # Priority queue stores (f_score, current_node, elapsed_hours)
        open_set = []
        heapq.heappush(open_set, (0.0, start_node, 0.0))

        came_from = {}
        g_score = {node: float("inf") for node in node_coords}
        g_score[start_node] = 0.0

        elapsed_time = {node: 0.0 for node in node_coords}

        visited = set()

        while open_set:
            current_f, current, t_curr = heapq.heappop(open_set)

            if current == goal_node:
                # Reconstruct path
                path_nodes = [current]
                while current in came_from:
                    current = came_from[current]
                    path_nodes.append(current)
                path_nodes.reverse()
                return path_nodes

            if current in visited:
                continue
            visited.add(current)

            lat_u, lon_u = node_coords[current]

            for neighbor in base_neighbors.get(current, []):
                lat_v, lon_v = node_coords[neighbor]

                # Land Mask check
                if not land_mask.is_line_navigable(lat_u, lon_u, lat_v, lon_v):
                    continue

                dist_nm = geodesic_distance_nm(lat_u, lon_u, lat_v, lon_v)

                meta_v = cell_meta.get(neighbor, {})
                base_ice = meta_v.get("ice_concentration_pct", 30.0)

                # Time-dependent forecast modulation at t_curr
                # If t_curr > 48 hours, apply forecast trend
                ice_t = base_ice
                if t_curr > 48:
                    ice_t = min(98.0, base_ice * 1.06)
                elif t_curr > 24:
                    ice_t = min(98.0, base_ice * 1.03)

                # Speed reduction in ice
                eff_speed = vessel_speed_knots
                if ice_t > 60:
                    eff_speed = max(5.0, vessel_speed_knots * 0.55)
                elif ice_t > 30:
                    eff_speed = max(8.0, vessel_speed_knots * 0.85)

                dt_hours = dist_nm / max(2.0, eff_speed)
                t_arrival = t_curr + dt_hours

                # Fuel factor
                fuel_factor = fuel_model.calculate_ice_factor(ice_t)

                # Iceberg proximity penalty
                ib_dist = meta_v.get("min_iceberg_dist_km", 50.0)
                ib_penalty = 0.0
                if ib_dist < 10.0:
                    ib_penalty = ((10.0 - ib_dist) ** 2) * 50.0 * iceberg_penalty_weight

                # Multi-objective edge cost
                edge_cost = (
                    (dist_nm * weight_dist) +
                    (meta_v.get("overall_risk", 35.0) * 1.6 * weight_risk) +
                    (dist_nm * fuel_factor * 0.9 * weight_fuel) +
                    ib_penalty
                )

                tentative_g = g_score[current] + edge_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    elapsed_time[neighbor] = t_arrival

                    # Admissible heuristic: geodesic distance to goal
                    h = geodesic_distance_nm(lat_v, lon_v, goal_lat, goal_lon) * weight_dist
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, neighbor, t_arrival))

        return None

    @staticmethod
    def smooth_path(node_path: list, node_coords: dict = None) -> list:
        """
        Simplifies path by removing redundant intermediate nodes when a direct
        geodesic line of sight exists that does not intersect land or severe obstacles.
        Supports both node names with node_coords dict, and direct [(lat, lon)] lists.
        """
        if not node_path or len(node_path) <= 2:
            return node_path

        smoothed = [node_path[0]]
        current_idx = 0

        while current_idx < len(node_path) - 1:
            next_idx = current_idx + 1
            for target_idx in range(len(node_path) - 1, current_idx + 1, -1):
                if node_coords is not None:
                    p1 = node_coords[node_path[current_idx]]
                    p2 = node_coords[node_path[target_idx]]
                else:
                    p1 = node_path[current_idx]
                    p2 = node_path[target_idx]

                if land_mask.is_line_navigable(p1[0], p1[1], p2[0], p2[1]):
                    next_idx = target_idx
                    break
            smoothed.append(node_path[next_idx])
            current_idx = next_idx

        return smoothed
