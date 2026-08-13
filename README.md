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
├── scripts/
│   ├── restart_server.sh              # Deployment service restart
│   ├── dlib_bin_version.sh            # Selects a native dlib-bin wheel
│   ├── select_python.sh               # Prefers a Python with a matching wheel
│   └── repair_cursor_python_tools.sh  # Restores Cursor's Python discovery binary
├── run.py                   # Entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10–3.14 on Apple Silicon, or Python 3.10–3.13 on Intel macOS
  (CI pins Python 3.11; `scripts/select_python.sh` prefers a supported interpreter)
- [Ollama](https://ollama.com/) running locally with the `openbmb/minicpm-o2.6` model
- CMake, pkg-config, and libpng for face recognition image mode
  - macOS example:
    ```bash
    brew install cmake pkg-config libpng
    ```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools<81" wheel
```

On Apple Silicon macOS, install the non-face dependencies first, then install
the required face-recognition packages with the prebuilt `dlib-bin` wheel.
This avoids compiling dlib's bundled ARM libpng:

```bash
python - <<'PY'
from pathlib import Path

requirements = Path("requirements.txt").read_text().splitlines()
Path("/tmp/gemi-requirements.txt").write_text(
    "\n".join(line for line in requirements
              if line.strip().lower() != "face_recognition") + "\n"
)
PY
python -m pip install --no-cache-dir -r /tmp/gemi-requirements.txt
DLIB_BIN_VERSION="$(scripts/dlib_bin_version.sh \
  "$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')" \
  "$(uname -m)")"
python -m pip install "dlib-bin==${DLIB_BIN_VERSION}"
python -m pip install --no-deps face-recognition==1.3.0
python -m pip install face-recognition-models==0.3.0
```

The helper selects `dlib-bin==20.0.1` for Apple Silicon Python 3.14 and
`dlib-bin==19.24.6` for older supported versions. It fails early for an
unsupported Python/architecture combination instead of attempting a source
build.

On other platforms, install all requirements normally:

```bash
python -m pip install -r requirements.txt
```

`setuptools<81` is required because `face-recognition-models` 0.3.0 still
imports the legacy `pkg_resources` API.

### 3. Run the server

From the editor, open **Run and Debug**, select **FastAPI: gemi-brain**, and
press the play button. The task **Run Gemi Brain (localhost:8000)** starts the
same server in a terminal without the debugger.

From the command line:

```bash
python run.py
# or directly:
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The API is now available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

Local runs bind `127.0.0.1:8000` and run as a single process, so stopping the
debug session or terminal releases the port immediately. Auto-reload is opt-in
because its watcher keeps a child process bound to the port:

```bash
APP_RELOAD=1 python run.py
```

The self-hosted deployment is a separate service on `0.0.0.0:${APP_PORT}`
(`8765` by default) and keeps serving while local runs are stopped. If
`http://localhost:8000` still answers after stopping, a previous run was left
behind — check it with `lsof -nP -iTCP:8000 -sTCP:LISTEN`.

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

## 🩺 Editor Tooling Repair (macOS + Cursor)

Cursor sometimes installs the platform-neutral build of the `ms-python.python`
extension, which ships without `python-env-tools/bin/pet`. That binary performs
Python environment discovery, so without it the interpreter picker reports
"Python is not installed", a selected interpreter is rejected as invalid, and
the debugger stops immediately because it receives an empty Python command.

Restore the binary and keep it restored after future extension updates:

```bash
./scripts/repair_cursor_python_tools.sh --install-agent
```

The script copies a working `pet` from the VS Code extension directory, caches
it in `~/.local/share/gemi-brain/python-env-tools` so later repairs do not need
VS Code, and installs a launchd agent that watches
`~/.cursor/extensions/extensions.json`. Whenever Cursor installs or updates an
extension, the agent restores `pet` automatically and logs to
`~/Library/Logs/com.gemi.cursor-python-tools.log`.

Run it once without arguments to repair now, and reload the Cursor window
afterwards so the extension host picks up the binary:

```bash
./scripts/repair_cursor_python_tools.sh
./scripts/repair_cursor_python_tools.sh --uninstall-agent  # stop watching
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
| `HOST` | `127.0.0.1` | Host interface used by `run.py` and `main.py` |
| `PORT` | `8000` | Port used by `run.py` and `main.py` |
| `APP_RELOAD` | `0` | Set to `1` to enable auto-reload for local development |
| `DB_PATH` | `gemi_memory.db` | Path to the SQLite database file |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generation endpoint |
| `OLLAMA_MODEL` | `openbmb/minicpm-o2.6` | Model to use for inference |
| `REQUIRED_MODEL` | `openbmb/minicpm-o2.6` | Model checked during readiness probe |
| `OLLAMA_TAGS_URL` | `http://localhost:11434/api/tags` | Ollama tags endpoint for health checks |