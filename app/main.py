"""FastAPI application factory for Gemi Brain."""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.db import database
from app.routes import health, perceive


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite database schema on startup."""
    database.init_db()
    yield


app = FastAPI(
    title="Gemi Brain API",
    description=(
        "Python AI backend for Gemi — the family pet robot. "
        "Handles perception (vision + audio), face recognition, memory, and Ollama AI responses."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(health.router)
app.include_router(perceive.router)
