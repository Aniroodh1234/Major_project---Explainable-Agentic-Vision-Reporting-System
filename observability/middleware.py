"""
FastAPI middleware that automatically records HTTP request metrics
for every incoming request.
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from observability.metrics_registry import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_IN_PROGRESS,
    HTTP_REQUESTS_TOTAL,
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Transparent middleware that records per-request Prometheus metrics.

    Tracked metrics
    ---------------
    * ``medvision_http_requests_total`` – counter with method / endpoint / status
    * ``medvision_http_request_duration_seconds`` – histogram per endpoint
    * ``medvision_http_requests_in_progress`` – gauge of concurrent requests
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        method = request.method
        endpoint = request.url.path

        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
        start_time = time.time()

        status_code = "500"
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response
        except Exception:
            raise
        finally:
            elapsed = time.time() - start_time
            HTTP_REQUESTS_TOTAL.labels(
                method=method, endpoint=endpoint, status_code=status_code,
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method, endpoint=endpoint,
            ).observe(elapsed)
            HTTP_REQUESTS_IN_PROGRESS.labels(
                method=method, endpoint=endpoint,
            ).dec()
