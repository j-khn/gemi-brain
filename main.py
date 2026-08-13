"""Root compatibility wrapper for app.main."""

from app.main import app

import os
import uvicorn

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
# Reload runs a watcher parent plus a server child. The child keeps the port
# bound when only the parent is stopped, so it stays opt-in.
RELOAD = os.getenv("APP_RELOAD", "0").strip().lower() in {"1", "true", "yes", "on"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=RELOAD)
