from fastapi import APIRouter

from app.repositories.report_repository import (
    get_history
)

from app.utils.response_builder import (
    paginated_response
)

from app.utils.logger import logger


router = APIRouter()


@router.get("/history")

async def get_history_endpoint(
    limit: int = 20,
    skip: int = 0
):

    logger.info(
        f"[API] Fetching history | "
        f"limit={limit} skip={skip}"
    )

    reports = await get_history(
        limit=limit,
        skip=skip
    )

    return paginated_response(
        items=reports,
        total=len(reports),
        limit=limit,
        skip=skip
    )