---
trigger: always_on
---

# Gemi Brain Agent Rules

- **Role:** Python AI backend for the family robot Gemi (running on macOS).
- **Core Tech:** Python 3.11+, FastAPI, `sqlite3`, Ollama (`minicpm-o2.6`).
- **Database:** Single file SQLite database (`gemi_memory.db`). Keep tables simple (`people`, `preferences`, `logs`). Zero auth or complex security overhead.
- **AI Processing:** Query local Ollama instance at `http://localhost:11434`. Restrict spoken responses to under 20 words.
- **API Endpoints:**
  - `POST /api/perceive`: Accepts image frames, audio/text prompts, and sensor status. Fetches memory context -> Queries MiniCPM-o -> Returns JSON `{"speech": "...", "action": "..."}`.
  - `GET /api/health`: Health status check.
- Write reusable, portable, lightweigh and scalable code.
- Always review, test, and run/ simulate your code after writing.