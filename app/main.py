from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api.router import router
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.request_logging import RequestLoggingMiddleware
from app.middleware.request_timing import RequestTimingMiddleware

from app.utils.exceptions import AppException

from app.api.exception_handlers import (
    app_exception_handler,
    validation_exception_handler,
    global_exception_handler
)

from app.services.embedding_service import (
    get_embedding_service
)

from app.utils.logger import logger

from app.config.settings import settings


# =========================================================
# FastAPI App Initialization
# =========================================================

app = FastAPI(
    title="Multi-Agent Research Platform",
    version="1.0.0"
)


# =========================================================
# Startup Initialization
# =========================================================

@app.on_event("startup")
async def startup_load_embedding_model():
    """
    Pre-load the embedding model once at application startup
    so subsequent research requests reuse the cached model
    instead of loading it on every request.
    """
    import time
    start = time.time()
    logger.info(
        "[EMBEDDINGS] Loading model at startup..."
    )
    svc = get_embedding_service()
    elapsed = round(time.time() - start, 2)
    logger.info(
        "[EMBEDDINGS] Model loaded successfully"
    )
    logger.info(
        f"[EMBEDDINGS] Dimension={svc.dimension}"
    )
    logger.info(
        f"[EMBEDDINGS] Startup load time: {elapsed}s"
    )

    # =============================================
    # Ensure MongoDB Indexes
    # =============================================

    from app.db.collections.chunks_collection import (
        ensure_chunks_indexes
    )
    from app.db.collections.reports_collection import (
        ensure_reports_indexes
    )
    from app.db.collections.evaluation_runs_collection import (
        ensure_evaluation_runs_indexes
    )

    try:

        await ensure_chunks_indexes()
        await ensure_reports_indexes()
        await ensure_evaluation_runs_indexes()

        logger.info("[INDEXES] All MongoDB indexes ensured at startup")

    except Exception as e:

        logger.warning(
            f"[INDEXES] Failed to ensure indexes at startup: {e}"
        )

    # =============================================
    # Groq Configuration Validation
    # =============================================
    #
    # Verify that the Groq API key is configured
    # before any research requests arrive.
    # Fail fast with a clear error message if
    # configuration is invalid.

    try:
        from groq import Groq

        logger.info(f"[LLM] Provider: {settings.LLM_PROVIDER}")
        logger.info(f"[LLM] Model: {settings.MODEL_NAME}")
        logger.info(f"[LLM] Timeout: 60s")
        logger.info(f"[LLM] Retries: {settings.MAX_RETRIES}")

        # Validate API key is set
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add GROQ_API_KEY=your-key-here to your .env file."
            )

        # Verify client initialization succeeds
        _test_client = Groq(
            api_key=settings.GROQ_API_KEY
        )

        logger.info(
            "[LLM] Groq client initialised successfully"
        )

    except Exception as e:
        logger.error(
            f"[LLM] Startup validation FAILED: {e}"
        )
        raise RuntimeError(f"Groq configuration invalid: {e}")


# =========================================================
# Shutdown Handler
# =========================================================

@app.on_event("shutdown")
async def shutdown_cleanup():

    # Close shared httpx client from content fetch agent
    try:
        from app.agents.content_fetch_agent import close_http_client
        await close_http_client()
    except Exception as e:
        logger.warning(
            f"[SHUTDOWN] httpx client close failed: {e}"
        )

    logger.info("[SHUTDOWN] Cleanup complete")


# =========================================================
# Middleware Registration
# =========================================================

app.add_middleware(RequestTimingMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)


# =========================================================
# Exception Handlers
# =========================================================

app.add_exception_handler(
    AppException,
    app_exception_handler
)

app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler
)

app.add_exception_handler(
    Exception,
    global_exception_handler
)


# =========================================================
# Routes
# =========================================================

app.include_router(router)


# =========================================================
# Root Endpoint
# =========================================================

@app.get("/")
async def root():
    return {
        "message": "Multi-Agent Research Platform API",
        "status": "running"
    }
