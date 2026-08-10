"""Pydantic schemas for the /api/perceive endpoint."""

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class PerceiveRequest(BaseModel):
    prompt: Optional[str] = Field(
        default=None,
        description="Audio transcript or text prompt from the user"
    )
    image_frame: Optional[str] = Field(
        default=None,
        description="Base64-encoded image frame from the robot camera"
    )
    image_frames: Optional[List[str]] = Field(
        default=None,
        description="Optional list of base64-encoded image frames (keyframes / sequential snapshots)"
    )
    sensor_status: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Robot sensor telemetry (battery, obstacles, etc.)"
    )
    person_name: Optional[str] = Field(
        default=None,
        description="Claimed name of the person (optional, used for identity verification)"
    )


class PerceiveResponse(BaseModel):
    speech: str = Field(description="Spoken response (strictly under 20 words)")
    action: str = Field(description="Physical motion action command for the robot")


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    checks: Dict[str, str]
