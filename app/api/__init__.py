from fastapi import APIRouter
from app.api.endpoints import chat, health

# Create main API router
router = APIRouter()

# Include endpoint routers
router.include_router(chat.router, prefix="/chat", tags=["chat"])
router.include_router(health.router, prefix="/health", tags=["health"])
