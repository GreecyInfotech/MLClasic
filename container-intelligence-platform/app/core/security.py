from fastapi import Header, HTTPException, status
from app.config import get_settings
async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    s = get_settings()
    if s.api_key and x_api_key != s.api_key: raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key.")
