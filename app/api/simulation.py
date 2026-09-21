"""What-If Simulation API endpoints."""
from fastapi import APIRouter
from app.simulation.simulator import simulator, SimulationParameters

router = APIRouter(prefix="/api/simulation", tags=["Simulation"])


@router.post("/run")
def run_simulation(params: SimulationParameters):
    """
    Executes What-If environmental perturbation simulation:
    Evaluates before vs after risk, fuel, distance, duration deltas.
    """
    return simulator.run_simulation(params=params)
