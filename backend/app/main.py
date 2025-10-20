"""FastAPI application exposing the agent backend services."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.router import api_router
from .core.config import get_settings
from .dependencies import get_schema_loader


@asynccontextmanager
def lifespan(_: FastAPI):
    # Trigger early validation of configuration and schema availability.
    get_settings()
    get_schema_loader()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic Water Permit API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    return app


app = create_app()
