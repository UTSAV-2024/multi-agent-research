import time

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

from app.utils.logger import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        request_id = getattr(request.state, "request_id", "unknown")

        logger.info(
            f"[REQUEST START] "
            f"id={request_id} "
            f"method={request.method} "
            f"path={request.url.path}"
        )

        start_time = time.perf_counter()

        try:
            response = await call_next(request)

            process_time = time.perf_counter() - start_time

            logger.info(
                f"[REQUEST END] "
                f"id={request_id} "
                f"status={response.status_code} "
                f"duration={process_time:.2f}s"
            )

            return response

        except Exception as e:

            process_time = time.perf_counter() - start_time

            logger.error(
                f"[REQUEST ERROR] "
                f"id={request_id} "
                f"duration={process_time:.2f}s "
                f"error={str(e)}"
            )

            raise e