"""Settings API: GET/POST runtime-configurable parameters."""

import json
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db import database

router = APIRouter()


class SettingsUpdateRequest(BaseModel):
    settings: Dict[str, Any]


@router.get("/api/settings")
async def get_settings():
    """Return all runtime-configurable settings with types and descriptions."""
    try:
        return database.get_all_settings()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/settings")
async def update_settings(payload: SettingsUpdateRequest):
    """Update multiple settings. Accepts a dict of key->value pairs.

    Values are stored as strings; type is inferred from existing setting or left as 'str'.
    """
    try:
        existing = database.get_all_settings()
        for k, v in payload.settings.items():
            # infer type for incoming value when possible
            if isinstance(v, (list, dict)):
                typ = "json"
                stored_val = json.dumps(v)
            elif isinstance(v, int):
                typ = "int"
                stored_val = str(v)
            elif isinstance(v, float):
                typ = "float"
                stored_val = str(v)
            else:
                typ = existing.get(k, {}).get("type", "str")
                stored_val = str(v)
            database.set_setting(k, stored_val, typ)
        return {"status": "ok", "updated": list(payload.settings.keys())}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
