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
