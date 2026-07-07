"""
API key authentication.

Simple header-based API key check — appropriate for a single-tenant
local/internal deployment (this isn't a multi-user SaaS auth system,
just a gate against the API being hit by anything other than your own
frontend/clients). Swap for OAuth2/JWT if this ever needs multi-user
support.
"""

from fastapi import Header, HTTPException, status

from app.core.config.settings import get_settings


async def require_api_key(x_api_key: str = Header(default=None, alias="X-API-Key")) -> None:
    settings = get_settings()
    if not x_api_key or x_api_key != settings.backend_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Provide it via the X-API-Key header.",
        )
