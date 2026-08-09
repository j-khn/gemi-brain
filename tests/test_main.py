"""Integration tests for Gemi Brain API."""

import asyncio
import os
import sqlite3

from fastapi.testclient import TestClient

# Isolate test DB before any app imports
os.environ["DB_PATH"] = "test_gemi_memory.db"

from app.db import database
from app.routes import perceive as perceive_route
from app.services import ollama
from app.main import app


def test_database_init_and_helpers():
    """Test DB initialization, person context, and log helpers."""
    print("Testing database module...")
    if os.path.exists("test_gemi_memory.db"):
        os.remove("test_gemi_memory.db")

    database.init_db()

    # Verify all tables exist
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "people" in tables, "people table missing"
        assert "preferences" in tables, "preferences table missing"
        assert "logs" in tables, "logs table missing"

    # Verify face_embedding column exists (schema migration check)
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(people)")
        cols = {row["name"] for row in cursor.fetchall()}
        assert "face_embedding" in cols, "face_embedding column missing"

    # Insert person + preference
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO people (name, notes) VALUES (?, ?)", ("Alice", "Loves green tea"))
        person_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO preferences (person_id, category, item) VALUES (?, ?, ?)",
            (person_id, "beverage", "green tea"),
        )
        conn.commit()

    context = database.get_person_context("Alice")
    assert "Alice" in context, f"Alice missing from context: {context}"
    assert "Loves green tea" in context, f"Notes missing from context: {context}"
    assert "green tea" in context, f"Preferences missing from context: {context}"

    log_id = database.save_interaction_log("Test event summary")
    assert log_id > 0, "Failed to save interaction log"
    print("Database module tests passed!")


def test_ollama_client_helpers():
    """Test word-limit enforcement and JSON parsing helpers."""
    print("Testing ollama client module helpers...")
    long_speech = " ".join(str(i) for i in range(30))  # 30 words
    truncated = ollama.enforce_word_limit(long_speech, 20)
    assert len(truncated.split()) == 20, f"Expected 20 words, got {len(truncated.split())}"

    parsed = ollama.parse_ollama_response('{"speech": "Hello!", "action": "nod"}')
    assert parsed["speech"] == "Hello!"
    assert parsed["action"] == "nod"

    markdown_json = ollama.parse_ollama_response('```json\n{"speech": "Hi there", "action": "wave"}\n```')
    assert markdown_json["speech"] == "Hi there"
    assert markdown_json["action"] == "stop_and_look"

    # Non-existent model should return False
    ok, msg = asyncio.run(ollama.check_ollama_health(required_model="non_existent_model_999"))
    assert ok is False
    assert "missing model" in msg or "unreachable" in msg

    print("Ollama client helper tests passed!")


def test_api_endpoints():
    """Test all FastAPI endpoints via TestClient."""
    print("Testing FastAPI endpoints...")
    client = TestClient(app)

    # GET /api/health
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "online"}

    # GET /api/health/ready
    response = client.get("/api/health/ready")
    ready_data = response.json()
    assert "status" in ready_data
    assert "checks" in ready_data
    assert ready_data["checks"]["database"] == "ok"
    assert ready_data["checks"]["face_recognition"] in ("ok", "missing face_recognition library")
    if response.status_code == 200:
        assert ready_data["status"] == "ready"
    else:
        assert ready_data["status"] == "unhealthy"

    # GET /apis/
    response = client.get("/apis/")
    assert response.status_code == 200
    apis_data = response.json()
    assert "apis" in apis_data
    paths = [item["path"] for item in apis_data["apis"]]
    assert "/api/health" in paths
    assert "/api/perceive" in paths
    assert "/apis/" in paths

    # POST /api/perceive with audio-only
    payload = {
        "prompt": "Hello Gemi!",
        "sensor_status": {"battery": 95, "obstacle": False},
        "person_name": "Alice",
    }
    response = client.post("/api/perceive", json=payload)
    assert response.status_code == 200, f"Perceive failed: {response.status_code} - {response.text}"
    data = response.json()
    assert "speech" in data
    assert "action" in data
    assert len(data["speech"].split()) <= 20, f"Speech exceeded 20 words: {data['speech']}"
    print("FastAPI endpoint tests passed!")


def test_health_requires_face_recognition(monkeypatch):
    """Verify readiness fails when face_recognition is unavailable."""
    client = TestClient(app)
    monkeypatch.setattr(perceive_route.vision, "check_face_recognition_health", lambda: (False, "missing face_recognition library"))

    response = client.get("/api/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["face_recognition"] == "missing face_recognition library"


def test_perceive_image_requires_face_recognition(monkeypatch):
    """Verify image-based perception fails when face_recognition is unavailable."""
    client = TestClient(app)
    monkeypatch.setattr(perceive_route.vision, "is_face_recognition_available", lambda: False)

    response = client.post(
        "/api/perceive",
        json={"image_frame": "aGVsbG8=", "person_name": "Alice"},
    )
    assert response.status_code == 503
    assert "Face recognition library unavailable" in response.json()["detail"]


if __name__ == "__main__":
    try:
        test_database_init_and_helpers()
        test_ollama_client_helpers()
        test_api_endpoints()
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
