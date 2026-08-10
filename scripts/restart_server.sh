#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Load environment from .env if present
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
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
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
elif [ -z "$PYTHON_BIN" ] && [ -x .venv/bin/python3 ]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python3"
elif [ -z "$PYTHON_BIN" ] && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif [ -z "$PYTHON_BIN" ] && command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: No Python interpreter found."
  echo "Checked .venv/bin/python, .venv/bin/python3, python3, and python."
  exit 1
fi

echo "Using Python interpreter: $PYTHON_BIN"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip --version

if [ -d .venv/bin ]; then
  echo "Virtualenv .venv/bin contents:"
  ls -1 .venv/bin | sort
fi

if ! "$PYTHON_BIN" -c 'import uvicorn, app.main' >/dev/null 2>&1; then
  echo "ERROR: uvicorn is not installed in $PYTHON_BIN environment."
  echo "The application or its dependencies cannot be imported."
  echo "Try installing dependencies with: $PYTHON_BIN -m pip install -r requirements.txt"
  exit 1
fi

mkdir -p logs
LOG_FILE="logs/gemi_brain.log"

echo "Starting Gemi Brain on ${HOST}:${PORT}..."
echo "Logs will be written to ${LOG_FILE}"

if [ "$(uname -s)" = "Darwin" ] && command -v launchctl >/dev/null 2>&1; then
  LABEL="com.gemi.brain"
  LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
  PLIST_FILE="${LAUNCH_AGENTS_DIR}/${LABEL}.plist"
  LAUNCHD_DOMAIN="gui/$(id -u)"

  mkdir -p "${LAUNCH_AGENTS_DIR}"
  : > "${LOG_FILE}"

  "$PYTHON_BIN" - "$PLIST_FILE" "$PROJECT_DIR" "$PYTHON_BIN" "$HOST" "$PORT" "${OLLAMA_URL:-}" <<'PY'
import plistlib
import sys
from pathlib import Path

plist_path, project_dir, python_bin, host, port, ollama_url = sys.argv[1:]
environment = {
    "HOST": host,
    "PORT": port,
    "PATH": f"{project_dir}/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin",
}
if ollama_url:
    environment["OLLAMA_URL"] = ollama_url

plist = {
    "Label": "com.gemi.brain",
    "ProgramArguments": [
        python_bin,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        port,
    ],
    "WorkingDirectory": project_dir,
    "EnvironmentVariables": environment,
    "RunAtLoad": True,
    "KeepAlive": True,
    "StandardOutPath": str(Path(project_dir) / "logs" / "gemi_brain.log"),
    "StandardErrorPath": str(Path(project_dir) / "logs" / "gemi_brain.log"),
}

with open(plist_path, "wb") as file:
    plistlib.dump(plist, file)
PY

  launchctl bootout "${LAUNCHD_DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
  launchctl bootstrap "${LAUNCHD_DOMAIN}" "${PLIST_FILE}"
  launchctl kickstart -k "${LAUNCHD_DOMAIN}/${LABEL}"
  echo "Gemi Brain launched as ${LABEL}."
else
  # Fallback for non-macOS development environments.
  export RUNNER_TRACKING_ID=""
  nohup "$PYTHON_BIN" -m uvicorn app.main:app --host "${HOST}" --port "${PORT}" > "$LOG_FILE" 2>&1 &
  PID=$!
  disown "$PID" 2>/dev/null || true
  echo "Gemi Brain started with PID ${PID}"
fi

for attempt in {1..60}; do
  if curl --fail --silent --show-error --connect-timeout 1 \
      "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
    echo "Health endpoint is ready on port ${PORT}."
    exit 0
  fi
  if [ -n "${PID:-}" ] && ! kill -0 "${PID}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "ERROR: Health endpoint is not ready on port ${PORT}."
echo "Listener details:"
lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN || true
echo "Last 80 lines of log:"
tail -n 80 "$LOG_FILE" || true
exit 1
