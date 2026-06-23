import logging, time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter(); response = await call_next(request)
        logging.getLogger("http").info("%s %s %s %.1fms", request.method, request.url.path, response.status_code, (time.perf_counter()-start)*1000)
        return response
