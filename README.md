# 🤖 Gemi Brain

**Gemi Brain** is the Python AI backend for **Gemi** — a cute, curious, and energetic pet robot designed to play with and entertain kids. It runs on macOS, uses a local [Ollama](https://ollama.com/) instance for AI responses, and SQLite for persistent memory.

---

## ✨ Features

- 🎙️ **Audio + Vision perception** — accepts text prompts and optional base64 camera frames
- 👤 **Multi-vector face recognition** — identifies people using stored face embeddings (dlib-based, optional)
- 🧠 **Persistent memory** — stores people, preferences, and interaction logs in SQLite
- 🐾 **Pet robot persona** — Gemi responds like a playful pet, not a corporate assistant
- 🌍 **Multilingual** — English, Urdu, and German
- ⚡ **Lightweight** — FastAPI + SQLite, no heavy infrastructure required

---

## 📁 Project Structure

```
gemi-brain/
├── app/
│   ├── main.py              # FastAPI app factory + lifespan
│   ├── routes/
│   │   ├── health.py        # GET /api/health, /api/health/ready, /apis
│   │   └── perceive.py      # POST /api/perceive  (core perception logic)
│   ├── schemas/
│   │   └── perceive.py      # Pydantic request/response models
│   ├── db/
│   │   └── database.py      # SQLite connection, init, migrations, helpers
│   └── services/
│       ├── ollama.py         # Ollama AI client (system prompt, parsing)
│       └── vision.py         # Face recognition utilities (optional)
├── tests/
│   └── test_main.py         # Integration tests
├── run.py                   # Entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with the `openbmb/minicpm-o2.6` model
- *(Optional)* CMake + libpng for face recognition: `brew install cmake libpng`

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the server

```bash
python run.py
# or directly:
uvicorn app.main:app --reload --port 8000
```

The API is now available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness check — returns `{"status": "online"}` |
| `GET` | `/api/health/ready` | Readiness check — verifies DB + Ollama + face recognition |
| `GET` | `/apis/` | Lists all registered API endpoints |
| `POST` | `/api/perceive` | Core perception endpoint (see below) |

### POST `/api/perceive`

**Request body:**

```json
{
  "prompt": "Hello Gemi!",
  "image_frame": "<base64-encoded-jpeg>",
  "sensor_status": { "battery": 95, "obstacle": false },
  "person_name": "Alice"
}
```

All fields are optional. If `image_frame` is omitted, Gemi operates in **audio-only mode**.

**Response:**

```json
{
  "speech": "Hi! I hear you but I can't see you right now!",
  "action": "tilt_head"
}
```

**Allowed actions:** `walk_around`, `sit`, `play`, `spin`, `tilt_head`, `sleep`, `investigate`, `stop_and_look`, `idle`, `nod`, `wave`

---

## 🧠 Face Recognition

Face recognition is **optional** — the server runs normally without it.

When enabled:
- Embeddings are stored as JSON arrays (`[[float,...], ...]`) in SQLite — one array per person, up to **10 angles/lighting profiles**
- **High-confidence match** (cosine > 0.85 or euclidean < 0.42): identity confirmed, person greeted by name
- **Auto-learning**: if the same person is seen from a new angle (distance > 0.2), the new embedding is stored automatically
- **Mismatched claim**: if `person_name` doesn't match the recognized face, Gemi politely challenges the identity
- **Unknown person**: if a face is detected but not recognized, Gemi asks for their name

To enable face recognition:
```bash
brew install cmake libpng
pip install face_recognition
```

---

## 🗄️ Database

SQLite file: `gemi_memory.db`

| Table | Purpose |
|-------|---------|
| `people` | Stores name, notes, and face embeddings per person |
| `preferences` | Stores per-person preferences (category + item) |
| `logs` | Interaction log with timestamps |

Schema migrations run automatically on startup — safe to use with older databases.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `gemi_memory.db` | Path to the SQLite database file |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generation endpoint |
| `OLLAMA_MODEL` | `openbmb/minicpm-o2.6` | Model to use for inference |
| `REQUIRED_MODEL` | `openbmb/minicpm-o2.6` | Model checked during readiness probe |
| `OLLAMA_TAGS_URL` | `http://localhost:11434/api/tags` | Ollama tags endpoint for health checks |