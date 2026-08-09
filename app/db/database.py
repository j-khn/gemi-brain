import os
import sqlite3
from typing import Optional

DB_PATH = os.getenv("DB_PATH", "gemi_memory.db")


def get_db_connection() -> sqlite3.Connection:
    """Connect to SQLite database and return connection instance."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables if they do not exist, and run any schema migrations."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                face_embedding TEXT DEFAULT '[]'
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                category TEXT,
                item TEXT,
                FOREIGN KEY (person_id) REFERENCES people(id)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_summary TEXT
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                user_prompt TEXT,
                gemi_response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES people(id)
            );
        """)
        conn.commit()

        # --- Schema migrations (safe to run on every startup) ---
        # Add face_embedding column if it was missing from an older database
        cursor.execute("PRAGMA table_info(people)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        if "face_embedding" not in existing_cols:
            cursor.execute("ALTER TABLE people ADD COLUMN face_embedding TEXT DEFAULT '[]'")
            conn.commit()


def get_person_context(person_name: Optional[str] = None) -> str:
    """Fetch memory context for a given person."""
    if not person_name or not person_name.strip():
        return ""

    context_parts = []
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, notes FROM people WHERE LOWER(name) = LOWER(?)",
            (person_name.strip(),),
        )
        person = cursor.fetchone()
        if person:
            context_parts.append(f"Person: {person['name']}. Notes: {person['notes'] or 'None'}")
            cursor.execute(
                "SELECT category, item FROM preferences WHERE person_id = ?",
                (person["id"],),
            )
            prefs = cursor.fetchall()
            if prefs:
                pref_str = ", ".join([f"{p['category']}: {p['item']}" for p in prefs])
                context_parts.append(f"Preferences: {pref_str}")

    return " | ".join(context_parts)



def save_interaction_log(event_summary: str) -> int:
    """Save an interaction log entry to the SQLite database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (event_summary) VALUES (?)",
            (event_summary,),
        )
        conn.commit()
        return cursor.lastrowid


def save_conversation_turn(person_id: Optional[int], user_prompt: str, gemi_response: str) -> int:
    """Save one conversation turn into the conversations table."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (person_id, user_prompt, gemi_response) VALUES (?, ?, ?)",
            (person_id, user_prompt, gemi_response),
        )
        conn.commit()
        return cursor.lastrowid


def get_conversation_history(person_id: int, limit: int = 10):
    """Return the last conversation turns for a person, newest last."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_prompt, gemi_response, timestamp FROM conversations WHERE person_id = ? ORDER BY timestamp DESC LIMIT ?",
            (person_id, limit),
        )
        rows = cursor.fetchall()
        history = []
        for row in reversed(rows):
            history.append({
                "user_prompt": row["user_prompt"],
                "gemi_response": row["gemi_response"],
                "timestamp": row["timestamp"],
            })
        return history


def get_recent_conversation_history(limit: int = 10, minutes: int = 10):
    """Return recent conversation turns across all persons within the last timeframe."""
    if minutes <= 0:
        minutes = 10
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_prompt, gemi_response, timestamp FROM conversations "
            "WHERE timestamp >= datetime('now', ? ) ORDER BY timestamp DESC LIMIT ?",
            (f'-{minutes} minutes', limit),
        )
        rows = cursor.fetchall()
        history = []
        for row in reversed(rows):
            history.append({
                "user_prompt": row["user_prompt"],
                "gemi_response": row["gemi_response"],
                "timestamp": row["timestamp"],
            })
        return history


def get_conversation_history_by_query(query: str, limit: int = 10):
    """Return conversation turns that mention query terms in user or gemi text."""
    if not query or not query.strip():
        return []
    search = f"%{query.strip()}%"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_prompt, gemi_response, timestamp FROM conversations "
            "WHERE user_prompt LIKE ? OR gemi_response LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (search, search, limit),
        )
        rows = cursor.fetchall()
        history = []
        for row in reversed(rows):
            history.append({
                "user_prompt": row["user_prompt"],
                "gemi_response": row["gemi_response"],
                "timestamp": row["timestamp"],
            })
        return history


def get_person_id_by_name(person_name: str) -> Optional[int]:
    """Return the person ID for a name if it exists."""
    if not person_name or not person_name.strip():
        return None
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM people WHERE LOWER(name) = LOWER(?)",
            (person_name.strip(),),
        )
        row = cursor.fetchone()
        return row["id"] if row else None


def check_db_health() -> tuple[bool, str]:
    """Check SQLite database health by executing SELECT 1."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.fetchone()
            return True, "ok"
    except Exception as exc:
        return False, f"database_error: {str(exc)}"

# New helpers for face embeddings (JSON stored as text)
import json
from typing import List, Tuple, Dict

def get_all_face_embeddings() -> List[Tuple[int, str, List[List[float]]]]:
    """Return list of (person_id, name, embeddings) for all people."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, face_embedding FROM people")
        rows = cursor.fetchall()
        result = []
        for row in rows:
            try:
                emb = json.loads(row["face_embedding"])
                if not isinstance(emb, list):
                    emb = []
            except Exception:
                emb = []
            result.append((row["id"], row["name"], emb))
        return result

def add_face_embedding(person_id: int, embedding: List[float]) -> None:
    """Append a new embedding vector to a person's stored embeddings, capped at 10 vectors."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT face_embedding FROM people WHERE id = ?", (person_id,))
        row = cursor.fetchone()
        if not row:
            return
        try:
            embeddings = json.loads(row["face_embedding"])
            if not isinstance(embeddings, list):
                embeddings = []
        except Exception:
            embeddings = []
        # Append if under cap
        if len(embeddings) < 10:
            embeddings.append(embedding)
            cursor.execute(
                "UPDATE people SET face_embedding = ? WHERE id = ?",
                (json.dumps(embeddings), person_id),
            )
            conn.commit()

