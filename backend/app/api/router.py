"""Root API router that wires the module routers together."""

from __future__ import annotations

from fastapi import APIRouter

from . import apply, chat, validate

api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(validate.router)
api_router.include_router(apply.router)
