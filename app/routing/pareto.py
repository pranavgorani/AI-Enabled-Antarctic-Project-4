"""
Pareto Multi-Objective Analysis for Antarctic Route Selection.

Evaluates alternative routes across conflicting objectives:
1. Operational Risk (POLARIS RIO and iceberg proximity)
2. Fuel Consumption (Lindqvist ice resistance + open water propulsion)
3. Transit Time (nautical miles at vessel operating speed in ice/open water)

Computes non-dominated Pareto frontier and identifies optimal compromise solutions.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class ParetoFrontier:
    """
    Computes Pareto-optimal routes given multiple objective functions:
    - minimize Risk Score (0 = lowest risk, 100 = critical risk)
    - minimize Fuel (tons)
    - minimize Transit Time (hours)
    """

    @staticmethod
    def is_dominated(candidate: Dict[str, float], others: List[Dict[str, float]], keys: List[str]) -> bool:
        """
        Check if candidate is strictly dominated by any other point.
        A dominates B if A is <= B in all objectives and < B in at least one.
        """
        for other in others:
            # Check if other <= candidate for all keys
            all_leq = all(other[k] <= candidate[k] for k in keys)
            strict_less = any(other[k] < candidate[k] for k in keys)
            if all_leq and strict_less:
                return True
        return False

    @classmethod
    def compute_pareto_front(
        cls,
        routes: List[Dict[str, Any]],
        objectives: List[str] = None
    ) -> Dict[str, Any]:
        """
        Extract non-dominated routes from candidate routes.

        Args:
            routes: List of route dictionaries containing objective metrics.
            objectives: List of metric keys to minimize. Default: ['risk_score', 'fuel_tonnes', 'transit_time_hours']

        Returns:
            Dict containing:
                - 'pareto_routes': List of non-dominated routes
                - 'dominated_routes': List of dominated routes
                - 'recommended_balanced_route': Route with minimum normalized Euclidean distance to ideal point (0,0,0)
        """
        if objectives is None:
            objectives = ["risk_score", "fuel_tonnes", "transit_time_hours"]

        if not routes:
            return {
                "pareto_routes": [],
                "dominated_routes": [],
                "recommended_balanced_route": None,
                "count_candidates": 0,
                "count_pareto": 0
            }

        pareto_routes = []
        dominated_routes = []

        for r in routes:
            metrics = {k: float(r.get(k, 0.0)) for k in objectives}
            is_dom = False
            for other in routes:
                if other is r:
                    continue
                other_metrics = {k: float(other.get(k, 0.0)) for k in objectives}
                all_leq = all(other_metrics[k] <= metrics[k] for k in objectives)
                strict_less = any(other_metrics[k] < metrics[k] for k in objectives)
                if all_leq and strict_less:
                    is_dom = True
                    break

            if is_dom:
                dominated_routes.append(r)
            else:
                pareto_routes.append(r)

        # Identify balanced compromise route (Utopia point distance minimization)
        balanced_route = None
        if pareto_routes:
            # Normalize objectives across pareto set
            min_vals = {k: min(r[k] for r in pareto_routes) for k in objectives}
            max_vals = {k: max(r[k] for r in pareto_routes) for k in objectives}

            best_dist = float("inf")
            for r in pareto_routes:
                dist_sq = 0.0
                for k in objectives:
                    span = max_vals[k] - min_vals[k]
                    norm_val = (r[k] - min_vals[k]) / (span if span > 1e-6 else 1.0)
                    dist_sq += norm_val ** 2
                dist = np.sqrt(dist_sq)
                if dist < best_dist:
                    best_dist = dist
                    balanced_route = r

        return {
            "pareto_routes": pareto_routes,
            "dominated_routes": dominated_routes,
            "recommended_balanced_route": balanced_route,
            "count_candidates": len(routes),
            "count_pareto": len(pareto_routes)
        }
