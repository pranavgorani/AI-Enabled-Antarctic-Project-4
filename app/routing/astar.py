"""
A* Pathfinding Algorithm for Polar Navigation
Polar Navigator AI - MoES / NCPOR

Computes the optimal path through a weighted marine grid graph using
the Haversine great-circle distance as the admissible heuristic.
"""

import heapq
import math


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates nautical miles between two coordinates."""
    r_nm = 3440.065  # Earth radius in nautical miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r_nm * c


class AStarRouter:
    """Finds path minimizing combined distance, environmental risk, and fuel cost."""

    @staticmethod
    def find_path(
        start_node: str,
        goal_node: str,
        graph: dict,
        node_coords: dict,
        heuristic_weight: float = 1.0
    ) -> list[str] | None:
        """
        A* search on a weighted graph.
        - graph[u] = [(v, weight), ...]
        - node_coords[u] = (lat, lon)
        """
        if start_node not in graph or goal_node not in graph:
            return None

        goal_lat, goal_lon = node_coords[goal_node]

        # Priority queue stores (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0.0, start_node))

        came_from = {}
        g_score = {node: float("inf") for node in graph}
        g_score[start_node] = 0.0

        f_score = {node: float("inf") for node in graph}
        h_start = haversine_nm(node_coords[start_node][0], node_coords[start_node][1], goal_lat, goal_lon)
        f_score[start_node] = h_start * heuristic_weight

        visited = set()

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == goal_node:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if current in visited:
                continue
            visited.add(current)

            curr_lat, curr_lon = node_coords[current]

            for neighbor, edge_weight in graph.get(current, []):
                tentative_g = g_score[current] + edge_weight

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    neigh_lat, neigh_lon = node_coords[neighbor]
                    h = haversine_nm(neigh_lat, neigh_lon, goal_lat, goal_lon) * heuristic_weight
                    f_score[neighbor] = tentative_g + h

                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None
