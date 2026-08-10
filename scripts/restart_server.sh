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

mkdir -p logs
LOG_FILE="logs/gemi_brain.log"

cat <<EOF > /tmp/gemi_brain_env.txt
HOST=${HOST}
PORT=${PORT}
APP_ENV=${APP_ENV}
OLLAMA_URL=${OLLAMA_URL}
EOF

echo "Starting Gemi Brain on ${HOST}:${PORT}..."
echo "Logs will be written to ${LOG_FILE}"
nohup "$PYTHON_BIN" -m uvicorn app.main:app --host "${HOST}" --port "${PORT}" > "$LOG_FILE" 2>&1 &
PID=$!
disown "$PID"

echo "Gemi Brain started with PID ${PID}"

sleep 3
if lsof -iTCP:${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "Process is listening on port ${PORT}."
else
  echo "ERROR: Process is not listening on port ${PORT}."
  echo "Last 80 lines of log:"
  tail -n 80 "$LOG_FILE" || true
  exit 1
fi
