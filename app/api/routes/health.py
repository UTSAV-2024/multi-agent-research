from fastapi import APIRouter

from app.config.settings import settings

from app.utils.response_builder import success_response


router = APIRouter()


# ==========================================
# HEALTH ENDPOINT
# ==========================================

@router.get("/health")
async def health_check():

    return success_response(
        data={
            "environment": settings.ENVIRONMENT,
            "version": settings.API_VERSION
        },
        message="Service is healthy"
    )
