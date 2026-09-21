"""
Dijkstra Pathfinding Algorithm for Polar Navigation
Polar Navigator AI - MoES / NCPOR
"""

import heapq


class DijkstraRouter:
    """Computes exact shortest path on weighted navigation graph."""

    @staticmethod
    def find_path(start_node: str, goal_node: str, graph: dict) -> list[str] | None:
        """
        Dijkstra search on weighted graph.
        - graph[u] = [(v, weight), ...]
        """
        if start_node not in graph or goal_node not in graph:
            return None

        distances = {node: float("inf") for node in graph}
        distances[start_node] = 0.0
        came_from = {}

        pq = [(0.0, start_node)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current == goal_node:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if current in visited:
                continue
            visited.add(current)

            for neighbor, weight in graph.get(current, []):
                new_dist = current_dist + weight
                if new_dist < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_dist
                    came_from[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

        return None
