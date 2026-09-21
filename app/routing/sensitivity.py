"""
Sensitivity Analysis for Route Optimization Weights.

Performs weight perturbation testing (+/- 10%, +/- 20%) across route cost components:
- Safety Weight (w_risk)
- Fuel Weight (w_fuel)
- Time Weight (w_time)

Determines whether the selected optimal route is robust to preference shifts
or sensitive to small parameter changes.
"""

from typing import List, Dict, Any
import numpy as np


class RouteSensitivityAnalyzer:
    """
    Analyzes route ranking stability under objective weight variations.
    """

    @classmethod
    def evaluate_sensitivity(
        cls,
        routes: List[Dict[str, Any]],
        base_weights: Dict[str, float] = None,
        perturbation_pct: float = 0.10
    ) -> Dict[str, Any]:
        """
        Evaluate route stability under +/- perturbation_pct shifts.

        Args:
            routes: Candidate routes with 'risk_score', 'fuel_tonnes', 'transit_time_hours'.
            base_weights: Base weights for risk, fuel, time (default: equal weights).
            perturbation_pct: Shift magnitude (default 0.10 for +/- 10%).

        Returns:
            Dict containing:
                - 'base_ranking': Route IDs ranked by base utility
                - 'stability_score': Proportion of perturbations where base #1 route remains #1 (0.0 to 1.0)
                - 'robustness_verdict': 'High Robustness', 'Moderate Robustness', or 'Sensitive to Weighting'
                - 'perturbation_results': Details of each perturbed scenario
        """
        if not routes:
            return {
                "base_ranking": [],
                "stability_score": 1.0,
                "robustness_verdict": "No candidates",
                "perturbation_results": []
            }

        if base_weights is None:
            base_weights = {"w_risk": 0.4, "w_fuel": 0.3, "w_time": 0.3}

        # Normalize metrics to 0-1 for fair utility scoring
        objectives = ["risk_score", "fuel_tonnes", "transit_time_hours"]
        weight_keys = ["w_risk", "w_fuel", "w_time"]
        obj_to_w = dict(zip(objectives, weight_keys))

        min_vals = {k: min(float(r.get(k, 0.0)) for r in routes) for k in objectives}
        max_vals = {k: max(float(r.get(k, 0.0)) for r in routes) for k in objectives}

        def score_route(route: Dict[str, Any], weights: Dict[str, float]) -> float:
            score = 0.0
            for obj in objectives:
                span = max_vals[obj] - min_vals[obj]
                norm = (float(route.get(obj, 0.0)) - min_vals[obj]) / (span if span > 1e-6 else 1.0)
                score += weights[obj_to_w[obj]] * norm
            return score

        # Compute base scores
        base_scores = [(r.get("route_id", f"route_{i}"), score_route(r, base_weights), r) for i, r in enumerate(routes)]
        base_scores.sort(key=lambda x: x[1])  # Lowest cost is best
        base_best_id = base_scores[0][0]
        base_ranking = [x[0] for x in base_scores]

        # Generate perturbations: increase one weight by perturbation_pct and renormalize
        perturbations = []
        best_matches = 0
        total_runs = 0

        shifts = [-perturbation_pct, perturbation_pct]
        for w_target in weight_keys:
            for shift in shifts:
                perturbed = dict(base_weights)
                perturbed[w_target] = max(0.01, perturbed[w_target] * (1.0 + shift))
                # Renormalize sum to 1.0
                total_w = sum(perturbed.values())
                perturbed = {k: v / total_w for k, v in perturbed.items()}

                p_scores = [(r.get("route_id", f"route_{i}"), score_route(r, perturbed)) for i, r in enumerate(routes)]
                p_scores.sort(key=lambda x: x[1])
                p_best_id = p_scores[0][0]

                is_match = (p_best_id == base_best_id)
                if is_match:
                    best_matches += 1
                total_runs += 1

                perturbations.append({
                    "altered_weight": w_target,
                    "shift_pct": shift * 100,
                    "weights": {k: round(v, 4) for k, v in perturbed.items()},
                    "winner_route_id": p_best_id,
                    "rank_1_stable": is_match
                })

        stability_score = round(best_matches / total_runs, 3) if total_runs > 0 else 1.0

        if stability_score >= 0.8:
            verdict = "High Robustness"
        elif stability_score >= 0.5:
            verdict = "Moderate Robustness"
        else:
            verdict = "Sensitive to Weight Shifts"

        return {
            "base_best_route": base_best_id,
            "base_ranking": base_ranking,
            "base_weights": base_weights,
            "stability_score": stability_score,
            "robustness_verdict": verdict,
            "perturbation_results": perturbations
        }
