import os
import sqlite3
import asyncio
from fastapi.testclient import TestClient

# Ensure test DB is isolated
os.environ["DB_PATH"] = "test_gemi_memory.db"

import database
import ollama_client
from main import app


def test_database_init_and_helpers():
    print("Testing database module...")
    if os.path.exists("test_gemi_memory.db"):
        os.remove("test_gemi_memory.db")

    database.init_db()
    
    # Verify tables exist
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "people" in tables, "people table missing"
        assert "preferences" in tables, "preferences table missing"
        assert "logs" in tables, "logs table missing"

    # Test adding a person and preference
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO people (name, notes) VALUES (?, ?)", ("Alice", "Loves green tea"))
        person_id = cursor.lastrowid
        cursor.execute("INSERT INTO preferences (person_id, category, item) VALUES (?, ?, ?)", (person_id, "beverage", "green tea"))
        conn.commit()

    context = database.get_person_context("Alice")
    assert "Alice" in context, f"Alice missing from context: {context}"
    assert "Loves green tea" in context, f"Notes missing from context: {context}"
    assert "green tea" in context, f"Preferences missing from context: {context}"

    # Test saving log
    log_id = database.save_interaction_log("Test event summary")
    assert log_id > 0, "Failed to save interaction log"
    print("Database module tests passed!")


def test_ollama_client_helpers():
    print("Testing ollama client module helpers...")
    long_speech = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two"
    truncated = ollama_client.enforce_word_limit(long_speech, 20)
    assert len(truncated.split()) == 20, f"Expected 20 words, got {len(truncated.split())}"

    parsed = ollama_client.parse_ollama_response('{"speech": "Hello!", "action": "nod"}')
    assert parsed["speech"] == "Hello!"
    assert parsed["action"] == "nod"

    markdown_json = ollama_client.parse_ollama_response('```json\n{"speech": "Hi there", "action": "wave"}\n```')
    assert markdown_json["speech"] == "Hi there"
    assert markdown_json["action"] == "wave"

    # Test checking for a non-existent model
    ok, msg = asyncio.run(ollama_client.check_ollama_health(required_model="non_existent_model_999"))
    assert ok is False
    assert "missing model" in msg or "unreachable" in msg

    print("Ollama client helper tests passed!")



def test_api_endpoints():
    print("Testing FastAPI endpoints...")
    client = TestClient(app)
    
    # Test GET /api/health
    response = client.get("/api/health")
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    assert response.json() == {"status": "online"}, f"Unexpected health response: {response.json()}"

    # Test GET /api/health/ready
    response = client.get("/api/health/ready")
    assert response.status_code in (200, 503), f"Unexpected readiness status code: {response.status_code}"
    ready_data = response.json()
    assert "status" in ready_data
    assert "checks" in ready_data
    assert "database" in ready_data["checks"]
    assert "ollama" in ready_data["checks"]
    assert ready_data["checks"]["database"] == "ok"


    # Test GET /apis/
    response = client.get("/apis/")
    assert response.status_code == 200, f"Apis listing failed: {response.status_code}"
    apis_data = response.json()
    assert "apis" in apis_data, "apis key missing from response"
    paths = [item["path"] for item in apis_data["apis"]]
    assert "/api/health" in paths
    assert "/api/perceive" in paths
    assert "/apis/" in paths

    # Test POST /api/perceive

    payload = {
        "prompt": "Hello Gemi!",
        "sensor_status": {"battery": 95, "obstacle": False},
        "person_name": "Alice"
    }
    response = client.post("/api/perceive", json=payload)
    assert response.status_code == 200, f"Perceive endpoint failed: {response.status_code} - {response.text}"
    data = response.json()
    assert "speech" in data, "speech field missing"
    assert "action" in data, "action field missing"
    assert len(data["speech"].split()) <= 20, f"Speech exceeded 20 words: {data['speech']}"
    print("FastAPI endpoint tests passed!")


if __name__ == "__main__":
    try:
        test_database_init_and_helpers()
        test_ollama_client_helpers()
        test_api_endpoints()
        print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    finally:
        if os.path.exists("test_gemi_memory.db"):
            try:
                os.remove("test_gemi_memory.db")
            except Exception:
                pass
