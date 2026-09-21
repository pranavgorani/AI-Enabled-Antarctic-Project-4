"""
Automated 20-Question AI Assistant Grounding & Evaluation Suite.

Evaluates the grounded decision-support capabilities of PolarNavigatorAI across:
- Route rationale and trade-offs
- Environmental hazard identification
- POLARIS RIO outcomes
- Monte Carlo iceberg trajectory dispersion
- Lindqvist (1989) ice resistance formulation
- ML model vs baseline performance
- What-if cryospheric simulations
- Vessel-specific constraints
"""

import pytest
from app.services.ai_assistant import ai_assistant


EVALUATION_QUESTIONS = [
    {
        "id": 1,
        "category": "Route Selection",
        "question": "Why was the recommended route selected over the direct track?",
        "expected_keywords": ["recommended", "risk score", "fastest", "iceberg", "polynya"],
        "expected_tool": "calculate_route"
    },
    {
        "id": 2,
        "category": "Hazards",
        "question": "What is the primary navigation hazard currently detected in the approach sector?",
        "expected_keywords": ["iceberg", "ice-042", "pack ice", "wind"],
        "expected_tool": "get_icebergs"
    },
    {
        "id": 3,
        "category": "Risk Comparison",
        "question": "Which route provides the lower predicted risk score?",
        "expected_keywords": ["recommended", "lowest", "risk score"],
        "expected_tool": "calculate_route"
    },
    {
        "id": 4,
        "category": "Fuel Efficiency",
        "question": "Which route is projected to burn less estimated fuel?",
        "expected_keywords": ["fuel-efficient", "fuel", "mt", "saving"],
        "expected_tool": "calculate_route"
    },
    {
        "id": 5,
        "category": "POLARIS Compliance",
        "question": "What is the POLARIS Risk Index Outcome (RIO) for a PC4 research vessel?",
        "expected_keywords": ["polaris", "pc4", "rio", "normal operation"],
        "expected_tool": "calculate_polaris_rio"
    },
    {
        "id": 6,
        "category": "POLARIS Limits",
        "question": "How does the POLARIS operational status change for a lower ice class like PC7?",
        "expected_keywords": ["pc7", "rio", "escort"],
        "expected_tool": "calculate_polaris_rio"
    },
    {
        "id": 7,
        "category": "Iceberg Uncertainty",
        "question": "What does the Monte Carlo ensemble say about the 90% probability corridor for ICE-042?",
        "expected_keywords": ["monte carlo", "corridor", "90%", "dispersion", "km"],
        "expected_tool": "run_monte_carlo_iceberg"
    },
    {
        "id": 8,
        "category": "Ice Resistance",
        "question": "Explain how the Lindqvist (1989) model breaks down ice resistance.",
        "expected_keywords": ["lindqvist", "crushing", "breaking", "submersion", "kn"],
        "expected_tool": "evaluate_fuel_lindqvist"
    },
    {
        "id": 9,
        "category": "ML Baselines",
        "question": "How does the sea-ice forecast model compare against persistence and climatology baselines?",
        "expected_keywords": ["baseline", "persistence", "climatology", "mae", "rmse"],
        "expected_tool": "get_sea_ice"
    },
    {
        "id": 10,
        "category": "What-If Simulation",
        "question": "What happens if sea-ice concentration increases by 15% along the track?",
        "expected_keywords": ["simulation", "risk", "fuel", "eta", "hours"],
        "expected_tool": "simulator.run_simulation"
    },
    {
        "id": 11,
        "category": "Target Proximity",
        "question": "Which iceberg is closest to our navigation track?",
        "expected_keywords": ["ice-042", "waypoint", "drift"],
        "expected_tool": "get_icebergs"
    },
    {
        "id": 12,
        "category": "Briefing",
        "question": "Generate an operational Antarctic navigation briefing.",
        "expected_keywords": ["antarctic navigation briefing", "executive summary", "polar class"],
        "expected_tool": "generate_briefing"
    },
    {
        "id": 13,
        "category": "Route Trade-Off",
        "question": "What trade-off is accepted by taking the fastest route?",
        "expected_keywords": ["fastest", "risk", "iceberg", "trade-off"],
        "expected_tool": "calculate_route"
    },
    {
        "id": 14,
        "category": "Weather Outlook",
        "question": "What are the major meteorological danger factors in the operational area?",
        "expected_keywords": ["katabatic", "icing", "wind"],
        "expected_tool": "get_weather"
    },
    {
        "id": 15,
        "category": "IIEE Metric",
        "question": "What is the Integrated Ice Edge Error (IIEE) for the ML model?",
        "expected_keywords": ["iiee", "km²", "baseline"],
        "expected_tool": "get_sea_ice"
    },
    {
        "id": 16,
        "category": "Vessel Specifics",
        "question": "What is the active vessel name and polar class?",
        "expected_keywords": ["rv bharati explorer", "pc4"],
        "expected_tool": "calculate_route"
    },
    {
        "id": 17,
        "category": "Monte Carlo Sizing",
        "question": "How many perturbation runs are conducted in the iceberg Monte Carlo simulation?",
        "expected_keywords": ["100", "monte carlo", "perturbations"],
        "expected_tool": "run_monte_carlo_iceberg"
    },
    {
        "id": 18,
        "category": "Fuel Physics",
        "question": "What specific fuel oil consumption (SFOC) coefficient is assumed for the diesel engines?",
        "expected_keywords": ["185", "g/kwh", "sfoc"],
        "expected_tool": "evaluate_fuel_lindqvist"
    },
    {
        "id": 19,
        "category": "Polaris RIV",
        "question": "Is operation authorized under IMO MSC.1/Circ.1519 for RV Bharati Explorer?",
        "expected_keywords": ["normal operation", "authorized", "rio"],
        "expected_tool": "calculate_polaris_rio"
    },
    {
        "id": 20,
        "category": "Safety Margin",
        "question": "Why is the recommended route considered safer despite being longer?",
        "expected_keywords": ["lowest", "risk", "polynya", "corridor"],
        "expected_tool": "calculate_route"
    }
]


class TestAIGroundingSuite:
    """Automated evaluation test suite verifying grounded AI responses."""

    def test_run_20_question_suite(self):
        passed_count = 0
        results = []

        for item in EVALUATION_QUESTIONS:
            res = ai_assistant.query_with_evidence(item["question"])
            answer_lower = res["answer"].lower()

            # Check keyword presence (at least 2 matching expected keywords)
            matches = [k for k in item["expected_keywords"] if k.lower() in answer_lower]
            keyword_pass = len(matches) >= min(2, len(item["expected_keywords"]))

            # Check tool invocation
            tool_pass = item["expected_tool"] in res["tools_called"]

            is_pass = keyword_pass and tool_pass
            if is_pass:
                passed_count += 1

            results.append({
                "id": item["id"],
                "category": item["category"],
                "pass": is_pass,
                "matches": matches,
                "tools_called": res["tools_called"],
                "confidence": res["confidence"]
            })

        pass_rate = (passed_count / len(EVALUATION_QUESTIONS)) * 100.0
        print(f"\n[AI Evaluation Suite] Completed 20 questions. Pass rate: {pass_rate:.1f}% ({passed_count}/20)")

        # Target: >= 90% pass rate
        assert pass_rate >= 90.0, f"AI Assistant grounding pass rate ({pass_rate}%) is below 90% threshold"

    def test_no_hallucinated_regulatory_approval(self):
        """Verifies that the assistant explicitly disclaims official IMO/MoES certification."""
        res = ai_assistant.query_with_evidence("Is this system officially certified by IMO or MoES?")
        assert "SIH prototype" in res["disclaimer"]
        assert "Not certified" in res["disclaimer"]
