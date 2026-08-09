"""Health check routes: /api/health, /api/health/ready, /apis."""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from app.db import database
from app.services import ollama, vision
from app.schemas.perceive import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get("/apis/", include_in_schema=True)
@router.get("/apis", include_in_schema=False)
async def list_available_apis():
    """Dynamically return all registered API endpoints."""
    from app.main import app  # local import to avoid circular dependency
    endpoints = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            endpoints.append({
                "path": route.path,
                "name": route.name,
                "methods": sorted(list(route.methods)),
                "summary": route.summary or (
                    route.description.strip().split("\n")[0] if route.description else ""
                ),
            })
    return {"apis": endpoints}


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Basic liveness health check."""
    return {"status": "online"}


@router.get("/api/health/ready", response_model=ReadinessResponse)
async def readiness_check():
    """Deep readiness check: verifies SQLite DB and Ollama availability."""
    db_ok, db_msg = database.check_db_health()
    ollama_ok, ollama_msg = await ollama.check_ollama_health()

    _, face_msg = vision.check_face_recognition_health()

    checks = {
        "database": db_msg,
        "ollama": ollama_msg,
        "face_recognition": face_msg,
    }

    if db_ok and ollama_ok and checks["face_recognition"] == "ok":
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ready", "checks": checks}
        )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "unhealthy", "checks": checks}
    )
