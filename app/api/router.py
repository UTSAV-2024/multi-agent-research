# ==========================================
# MAIN API ROUTER
# ==========================================
#
# Aggregates all route modules.
# No business logic — only route registration.
#
# ==========================================

from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.research import router as research_router
from app.api.routes.history import router as history_router
from app.api.routes.reports import router as reports_router


router = APIRouter()

router.include_router(health_router)
router.include_router(research_router)
router.include_router(history_router)
router.include_router(reports_router)
