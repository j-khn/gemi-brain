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
- Write reusable, portable, lightweight and scalable code.
- **After every change:**
  1. **Run tests** — always run `pytest tests/ -v` and fix any failures before finishing.
  2. **Update tests** — verify existing tests still cover the change; add or update tests in `tests/` when behavior, endpoints, or APIs change.
  3. **Update README** — verify `README.md` still matches the codebase; update setup, usage, project structure, or API docs when the change affects them.
- **After completing a change:** commit the verified changes and push them to the configured upstream branch.