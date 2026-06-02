# ==========================================
# RESPONSE BUILDER
# ==========================================
#
# Standardized API response formatting.
# All API routes should use these helpers
# for consistent response structure.
#
# ==========================================

from typing import Any, Optional


def success_response(
    data: Any = None,
    message: str = "Success",
    status_code: int = 200,
    **extra
) -> dict:

    response = {
        "status": "success",
        "message": message,
    }

    if data is not None:
        response["data"] = data

    if extra:
        response.update(extra)

    return response


def error_response(
    message: str = "Something went wrong",
    status_code: int = 500,
    error_code: str = "INTERNAL_SERVER_ERROR",
    details: Optional[Any] = None
) -> dict:

    response = {
        "status": "failed",
        "error": {
            "code": error_code,
            "message": message
        }
    }

    if details is not None:
        response["error"]["details"] = details

    return response


def paginated_response(
    items: list,
    total: int,
    limit: int,
    skip: int,
    message: str = "Success"
) -> dict:

    return {
        "status": "success",
        "message": message,
        "count": total,
        "limit": limit,
        "skip": skip,
        "data": items
    }
