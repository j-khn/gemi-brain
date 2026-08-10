import json
import os

from fastapi.testclient import TestClient

os.environ["DB_PATH"] = "test_gemi_memory.db"

from app.main import app
from app.db import database
from app.services import ollama


def setup_module():
    # ensure fresh DB
    if os.path.exists("test_gemi_memory.db"):
        os.remove("test_gemi_memory.db")
    database.init_db()


def test_settings_api_and_allowed_actions():
    client = TestClient(app)

    # GET settings
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    settings = resp.json()
    assert "allowed_actions" in settings
    assert "nod" in settings["allowed_actions"]["value"]

    # Update allowed_actions to include a custom action
    new_actions = ["walk_around", "sit", "custom_action"]
    resp = client.post("/api/settings", json={"settings": {"allowed_actions": new_actions}})
    assert resp.status_code == 200

    # Now parsing a response with custom_action should preserve it
    parsed = ollama.parse_ollama_response('{"speech": "Hi", "action": "custom_action"}')
    assert parsed["action"] == "custom_action"
