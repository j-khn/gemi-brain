#!/bin/bash
set -e

# Load environment from .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

# Kill existing server on the port if running
if lsof -iTCP:${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "Stopping existing process listening on port ${PORT}..."
  kill -TERM $(lsof -iTCP:${PORT} -sTCP:LISTEN -t)
  sleep 2
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ] && [ -x .venv/bin/python ]; then
  PYTHON_BIN=".venv/bin/python"
elif [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

echo "Starting Gemi Brain on ${HOST}:${PORT}..."
"$PYTHON_BIN" -m uvicorn app.main:app --host "${HOST}" --port "${PORT}" --reload &

echo "Gemi Brain started with PID $!"
