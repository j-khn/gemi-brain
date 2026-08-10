# 🤖 Gemi Brain

**Gemi Brain** is the Python AI backend for **Gemi** — a cute, curious, and energetic pet robot designed to play with and entertain kids. It runs on macOS, uses a local [Ollama](https://ollama.com/) instance for AI responses, and SQLite for persistent memory.

---

## ✨ Features

- 🎙️ **Audio + Vision perception** — accepts text prompts and base64 camera frames
- 👤 **Multi-vector face recognition** — identifies people using stored face embeddings (dlib-based, required for visual input)
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
│       └── vision.py         # Face recognition utilities (required for image mode)
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
- CMake, pkg-config, and libpng for face recognition image mode
  - macOS example:
    ```bash
    brew install cmake pkg-config libpng
    ```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon macOS, install the required face-recognition packages with
the prebuilt `dlib-bin` wheel to avoid compiling dlib's bundled ARM libpng:

```bash
python -m pip install -r <(python -c 'from pathlib import Path; print("\n".join(line for line in Path("requirements.txt").read_text().splitlines() if line.strip().lower() != "face_recognition"))')
python -m pip install dlib-bin==19.24.6
python -m pip install --no-deps face-recognition==1.3.0
python -m pip install face-recognition-models==0.3.0
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
| `GET` | `/api/health/ready` | Readiness check — verifies DB + Ollama (face recognition required for image mode) |
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

Face recognition is **required for image mode**. If the library is not installed, `/api/health/ready` reports `unhealthy`, and `POST /api/perceive` with `image_frame` returns `503`.

When enabled:
- Embeddings are stored as JSON arrays (`[[float,...], ...]`) in SQLite — one array per person, up to **10 angles/lighting profiles**
- **High-confidence match** (cosine > 0.85 or euclidean < 0.42): identity confirmed, person greeted by name
- **Auto-learning**: if the same person is seen from a new angle (distance > 0.2), the new embedding is stored automatically
- **Mismatched claim**: if `person_name` doesn't match the recognized face, Gemi politely challenges the identity
- **Unknown person**: if a face is detected but not recognized, Gemi asks for their name

To enable face recognition for image mode:
```bash
brew install cmake libpng
pip install -r requirements.txt
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

## 🚀 Self-hosted Deployment

This repository supports self-hosted deployment on a local macOS runner.

1. Configure the runner machine and labels:
   - `self-hosted`
   - `macos`
   - `X64`
   - runner name: `MacBook-Pro`

2. Set repository/environment variables for deployment:
   - `APP_HOST`
   - `APP_PORT`
   - `OLLAMA_URL`

3. On the runner machine, start the Actions runner from the repository workspace:

```bash
cd /Users/test/Projects/gemi-brain
./actions-runner/run.sh
```

4. The CI workflow runs deployment as a reusable workflow after tests pass on `develop` or `main`, using the exact tested commit. It can also be started manually via `workflow_dispatch`.

5. Deployment copies the tested application to `~/.gemi-brain`, starts a
   macOS `launchd` service when the runner session permits it, and otherwise
   uses a detached process. In both cases it binds to `0.0.0.0:${APP_PORT}`
   independently of the Actions workspace.

6. For local testing, copy the example env file and restart the server:

```bash
cp .env.example .env
./scripts/restart_server.sh
```

7. The app will bind to `0.0.0.0` when the runner or script uses that host.

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `gemi_memory.db` | Path to the SQLite database file |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generation endpoint |
| `OLLAMA_MODEL` | `openbmb/minicpm-o2.6` | Model to use for inference |
| `REQUIRED_MODEL` | `openbmb/minicpm-o2.6` | Model checked during readiness probe |
| `OLLAMA_TAGS_URL` | `http://localhost:11434/api/tags` | Ollama tags endpoint for health checks |