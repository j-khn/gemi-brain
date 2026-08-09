"""main.py — Gemi Brain FastAPI application."""

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

import database
import face_utils
import ollama_client
from database import add_face_embedding, get_all_face_embeddings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite database schema on startup."""
    database.init_db()
    yield


app = FastAPI(
    title="Gemi Brain API",
    description="Python AI backend for the family robot Gemi",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PerceiveRequest(BaseModel):
    prompt: Optional[str] = Field(default=None, description="Audio transcript or text prompt from user")
    image_frame: Optional[str] = Field(default=None, description="Base64 encoded image frame")
    sensor_status: Optional[Dict[str, Any]] = Field(default=None, description="Robot sensor telemetry status")
    person_name: Optional[str] = Field(default=None, description="Claimed name of the person (optional)")


class PerceiveResponse(BaseModel):
    speech: str = Field(description="Spoken response (strictly under 20 words)")
    action: str = Field(description="Physical motion action command")


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    checks: Dict[str, str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/apis/", include_in_schema=True)
@app.get("/apis", include_in_schema=False)
async def list_available_apis():
    """Dynamically return list of available API endpoints in JSON format."""
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


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Basic liveness health check endpoint."""
    return {"status": "online"}


@app.get("/api/health/ready", response_model=ReadinessResponse)
async def readiness_check():
    """Deep readiness check verifying SQLite DB and local Ollama instance."""
    db_ok, db_msg = database.check_db_health()
    ollama_ok, ollama_msg = await ollama_client.check_ollama_health()

    checks = {
        "database": db_msg,
        "ollama": ollama_msg,
        "face_recognition": "available" if face_utils.is_face_recognition_available() else "unavailable (dlib not compiled)",
    }

    if db_ok and ollama_ok:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready", "checks": checks})
    else:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "unhealthy", "checks": checks})


@app.post("/api/perceive", response_model=PerceiveResponse)
async def perceive(request: PerceiveRequest):
    """
    Perception endpoint.

    Accepts text prompt, optional image frame, and sensor status.
    Fetches memory context → calls local Ollama model → logs interaction → returns speech + action.
    """
    try:
        # ------------------------------------------------------------------
        # 1. Vision: decode image and identify person (if face_recognition available)
        # ------------------------------------------------------------------
        vision_available = bool(request.image_frame)
        identified_person: Optional[str] = None
        face_detected = False  # True when a face was found in the image

        if vision_available and face_utils.is_face_recognition_available():
            try:
                image_bytes = face_utils.decode_image_base64(request.image_frame)
                query_emb = face_utils.extract_face_embedding(image_bytes)
                face_detected = True  # extract_face_embedding raises if no face found
                stored_embeddings = get_all_face_embeddings()
                match = face_utils.match_face(query_emb, stored_embeddings)
                if match:
                    person_id, name, cos_sim, euc_dist = match
                    if cos_sim > 0.85 or euc_dist < 0.42:
                        identified_person = name
                        # Auto-learn new angle: store if meaningfully different
                        if euc_dist > 0.2:
                            add_face_embedding(person_id, query_emb)
                # If match is None or below threshold, identified_person stays None
            except ValueError:
                # ValueError from extract_face_embedding means no face in frame
                face_detected = False
            except Exception:
                # Other errors (decode, etc.) — treat as vision unavailable
                vision_available = False

        # ------------------------------------------------------------------
        # 2. Detect identity mismatch and resolve effective person name
        # ------------------------------------------------------------------
        identity_mismatch = False
        claimed_name: Optional[str] = request.person_name

        if identified_person:
            effective_name = identified_person
            # Check if user claimed a different identity
            if claimed_name and claimed_name.lower() != identified_person.lower():
                identity_mismatch = True
        elif claimed_name:
            effective_name = claimed_name
        else:
            effective_name = None

        memory_context = database.get_person_context(effective_name)

        # ------------------------------------------------------------------
        # 3. Build enriched prompt and query Ollama
        # ------------------------------------------------------------------
        # Build enriched prompt text including identity mismatch warning
        enriched_prompt = request.prompt or ""
        if identity_mismatch:
            enriched_prompt = (
                f"{enriched_prompt} "
                f"[SYSTEM NOTE: User claims to be '{claimed_name}' but face recognition "
                f"identified them as '{identified_person}'. Politely challenge/verify their identity.]"
            ).strip()

        result = await ollama_client.perceive_and_generate(
            prompt=enriched_prompt or request.prompt,
            image_frame=request.image_frame,
            memory_context=memory_context,
            sensor_status=request.sensor_status,
            vision_available=vision_available,
            identified_person=identified_person,
            face_detected=face_detected,
        )

        speech = result.get("speech", "Hello!")
        action = result.get("action", "stop_and_look")

        # ------------------------------------------------------------------
        # 4. Log interaction
        # ------------------------------------------------------------------
        prompt_summary = request.prompt or "Visual/Sensor input"
        event_summary = f"Prompt: {prompt_summary} -> Speech: '{speech}' | Action: {action}"
        database.save_interaction_log(event_summary)

        return PerceiveResponse(speech=speech, action=action)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
