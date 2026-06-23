import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger(__name__).exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})
