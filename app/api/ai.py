"""AI Decision Support and Briefing endpoints."""
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.ai_assistant import ai_assistant

router = APIRouter(prefix="/api/ai", tags=["AI Navigator"])


class AIQueryRequest(BaseModel):
    query: str


class BriefingRequest(BaseModel):
    mission_name: str = "Antarctic Research Mission Alpha"
    vessel_name: str = "RV Explorer"
    destination: str = "Maitri Research Station"
    forecast_window: str = "72 Hours"


@router.post("/query")
def query_ai_assistant(req: AIQueryRequest):
    """Answers operational questions grounded in actual environmental and routing data."""
    answer = ai_assistant.query(req.query)
    return {
        "status": "success",
        "query": req.query,
        "response": answer,
        "data_mode": "DEMO DATA — NOT FOR REAL-WORLD NAVIGATION"
    }


@router.post("/briefing")
def generate_briefing(req: BriefingRequest):
    """Generates an executive Antarctic Navigation Briefing."""
    briefing = ai_assistant.generate_briefing(
        mission_name=req.mission_name,
        vessel_name=req.vessel_name,
        destination=req.destination,
        forecast_window=req.forecast_window
    )
    return {
        "status": "success",
        "briefing": briefing,
        "data_mode": "DEMO DATA — NOT FOR REAL-WORLD NAVIGATION"
    }
