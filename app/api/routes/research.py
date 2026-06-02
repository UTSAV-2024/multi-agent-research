from fastapi import APIRouter, Request

from app.orchestrator.workflow import research

from app.schemas.research_schema import ResearchRequest

from app.utils.logger import logger


router = APIRouter()


# ==========================================
# RESEARCH ENDPOINT
# ==========================================

@router.post("/research")
async def run_research(
    payload: ResearchRequest,
    request: Request
):

    request_id = getattr(
        request.state,
        "request_id",
        "unknown"
    )

    logger.info(
        f"[API] Research request received | "
        f"request_id={request_id}"
    )

    result = await research(
        payload.topic,
        request_id=request_id
    )

    logger.info(
        f"[API] Research request completed | "
        f"request_id={request_id}"
    )

    return result
