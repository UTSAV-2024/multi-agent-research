from fastapi import APIRouter
from app.orchestrator.workflow import research
from app.schemas.research_schema import ResearchRequest
router = APIRouter()


@router.get("/")
def home():

    return {
        "message": "AI Research Platform Running"
    }


@router.post("/research")
def run_research(payload: ResearchRequest):

    topic = payload.get("topic")

    result = research(topic)

    return {
        "topic": topic,
        "report": result
    }