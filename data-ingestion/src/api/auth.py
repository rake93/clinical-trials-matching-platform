"""JWT token verification for the ingestion API server.

Tokens are issued by the reasoning server (POST /api/auth/login on :8000).
This module only verifies them — it does NOT issue new tokens.

Both servers share AUTH_JWT_SECRET from .env.local so that a single login
session authorises requests to both :8000 (reasoning) and :8001 (ingestion).
"""
from __future__ import annotations

import logging
import os

import bcrypt
from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

_ALGORITHM = "HS256"
_bearer = HTTPBearer(auto_error=False)
# Module-level dependency instance avoids B008 (function call in default arg)
_bearer_dep: HTTPAuthorizationCredentials | None = Security(_bearer)


def _get_secret() -> str:
    secret = os.environ.get("AUTH_JWT_SECRET", "").strip()
    if not secret:
        raise RuntimeError(
            "AUTH_JWT_SECRET must be set. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return secret


def verify_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = _bearer_dep,
) -> str:
    """FastAPI dependency — accept JWT from Authorization or X-Auth-Token header.

    RunPod nginx strips Authorization; X-Auth-Token is the fallback.
    Returns the sub claim on success, raises HTTP 401 otherwise.
    """
    token: str | None = None
    if credentials is not None:
        token = credentials.credentials
    if not token:
        token = request.headers.get("X-Auth-Token")
    if not token:
        raise HTTPException(
            status_code=401,
            detail={"code": "missing_token", "message": "Authentication required.", "retryable": False},
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[_ALGORITHM])
        sub: str | None = payload.get("sub")
        if not sub:
            raise JWTError("missing sub claim")
        return sub
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail={"code": "invalid_token", "message": "Invalid or expired token. Please log in again.", "retryable": False},
            headers={"WWW-Authenticate": "Bearer"},
        )
