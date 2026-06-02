from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.utils.logger import logger
from app.utils.exceptions import AppException


async def app_exception_handler(
    request: Request,
    exc: AppException
):
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"[APP EXCEPTION] "
        f"id={request_id} "
        f"error_code={exc.error_code} "
        f"message={exc.message}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "failed",
            "error": {
                "code": exc.error_code,
                "message": exc.message
            },
            "request_id": request_id
        }
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"[VALIDATION ERROR] "
        f"id={request_id} "
        f"errors={exc.errors()}"
    )

    return JSONResponse(
        status_code=422,
        content={
            "status": "failed",
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request payload",
                "details": exc.errors()
            },
            "request_id": request_id
        }
    )


async def global_exception_handler(
    request: Request,
    exc: Exception
):
    request_id = getattr(request.state, "request_id", "unknown")

    logger.exception(
        f"[UNHANDLED ERROR] "
        f"id={request_id} "
        f"error={str(exc)}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "status": "failed",
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Something went wrong"
            },
            "request_id": request_id
        }
    )