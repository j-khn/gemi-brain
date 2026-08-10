import os
import sqlite3
from typing import Optional, List

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

        # Create settings table for runtime-configurable parameters
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                type TEXT,
                description TEXT
            );
            """
        )
        conn.commit()

        # Seed default settings if missing
        defaults = {
            "ollama_url": (os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"), "str", "URL for Ollama generate endpoint"),
            "ollama_model": (os.getenv("OLLAMA_MODEL", "openbmb/minicpm-o2.6"), "str", "Default Ollama model"),
            "ollama_tags_url": (os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags"), "str", "Ollama tags endpoint"),
            "ollama_max_tokens": ("512", "int", "Max tokens to request from Ollama"),
            "ollama_num_predict": ("512", "int", "Alternate token/predict budget for Ollama"),
            "word_limit": ("20", "int", "Max words in Gemi speech responses"),
            "face_cosine_threshold": ("0.85", "float", "Cosine similarity threshold to accept a face match"),
            "face_euclid_threshold": ("0.42", "float", "Euclidean distance threshold to accept a face match"),
            "face_euclid_add_threshold": ("0.2", "float", "Euclid distance above which to add new embedding"),
            "face_embedding_cap": ("10", "int", "Max embeddings stored per person"),
            "image_sequence_limit": ("6", "int", "Max number of image frames accepted per request"),
            "allowed_actions": (json.dumps([
                "walk_around",
                "sit",
                "play",
                "spin",
                "tilt_head",
                "sleep",
                "investigate",
                "stop_and_look",
                "idle",
                "nod",
            ]), "json", "Allowed action set for robot responses"),
        }

        for k, (v, t, d) in defaults.items():
            cursor.execute("SELECT 1 FROM settings WHERE key = ?", (k,))
            if cursor.fetchone() is None:
                cursor.execute(
                    "INSERT INTO settings (key, value, type, description) VALUES (?, ?, ?, ?)",
                    (k, str(v), t, d),
                )
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


def get_person_contexts(names: List[str]) -> str:
    """Return an aggregated context string for a list of person names.

    Example return: "Visible people: Jam (father), Alice (daughter) | Person: Jam. Notes: ..."
    """
    if not names:
        return ""

    visible_parts = []
    details_parts = []
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for name in names:
            if not name or not name.strip():
                continue
            cursor.execute(
                "SELECT id, name, notes FROM people WHERE LOWER(name) = LOWER(?)",
                (name.strip(),),
            )
            person = cursor.fetchone()
            if person:
                note = person["notes"] or ""
                # short descriptor for visible list
                short = person["name"]
                if note:
                    # try to pick a short parenthetical from notes (first 40 chars)
                    short += f" ({note[:40].strip()})"
                visible_parts.append(short)
                details_parts.append(f"Person: {person['name']}. Notes: {note or 'None'}")
            else:
                visible_parts.append(name)

    visible_desc = ", ".join(visible_parts)
    details = " | ".join(details_parts)
    if details:
        return f"Visible people: {visible_desc} | {details}"
    return f"Visible people: {visible_desc}"



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


def get_setting(key: str, default: Optional[str] = None):
    """Retrieve a setting value (as string) or default if not present."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value, type FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        if not row:
            return default
        val, typ = row[0], row[1]
        # Parse by type
        try:
            if typ == "int":
                return int(val)
            if typ == "float":
                return float(val)
            if typ == "str":
                return str(val)
            if typ == "json":
                try:
                    return json.loads(val)
                except Exception:
                    return val
        except Exception:
            return val
        return val


def set_setting(key: str, value: str, typ: str = "str", description: Optional[str] = None):
    """Upsert a setting key with value and type.

    Returns True if a change was made.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM settings WHERE key = ?", (key,))
        exists = cursor.fetchone() is not None
        if exists:
            cursor.execute("UPDATE settings SET value = ?, type = ?, description = COALESCE(?, description) WHERE key = ?", (str(value), typ, description, key))
        else:
            cursor.execute("INSERT INTO settings (key, value, type, description) VALUES (?, ?, ?, ?)", (key, str(value), typ, description or ""))
        conn.commit()
        return True


def get_all_settings() -> dict:
    """Return all settings as parsed types in a dict."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value, type, description FROM settings ORDER BY key")
        rows = cursor.fetchall()
        result = {}
        for row in rows:
            key = row[0]
            val = row[1]
            typ = row[2]
            try:
                if typ == "int":
                    parsed = int(val)
                elif typ == "float":
                    parsed = float(val)
                elif typ == "json":
                    try:
                        parsed = json.loads(val)
                    except Exception:
                        parsed = val
                else:
                    parsed = val
            except Exception:
                parsed = val
            result[key] = {"value": parsed, "type": typ, "description": row[3]}
        return result

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

