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

PORT=${PORT:-8000}
LABEL="com.gemi.brain"
LAUNCHD_DOMAIN="gui/$(id -u)"

stop_launchd_service() {
  if [ "$(uname -s)" != "Darwin" ] || ! command -v launchctl >/dev/null 2>&1; then
    return 0
  fi

  if launchctl print "${LAUNCHD_DOMAIN}/${LABEL}" >/dev/null 2>&1; then
    echo "Stopping launchd service ${LABEL}..."
    launchctl bootout "${LAUNCHD_DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
  fi
}

stop_port_listeners() {
  local pids
  pids="$(lsof -iTCP:"${PORT}" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    return 0
  fi

  echo "Stopping process(es) listening on port ${PORT}..."
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
  sleep 2

  pids="$(lsof -iTCP:"${PORT}" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "Force-stopping remaining listener(s) on port ${PORT}..."
    # shellcheck disable=SC2086
    kill -KILL $pids 2>/dev/null || true
    sleep 1
  fi
}

echo "Stopping Gemi Brain on port ${PORT}..."

stop_launchd_service
stop_port_listeners

if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "ERROR: A process is still listening on port ${PORT}."
  lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN || true
  exit 1
fi

if curl --fail --silent --show-error --connect-timeout 1 \
    "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
  echo "ERROR: Health endpoint is still responding on port ${PORT}."
  exit 1
fi

echo "Gemi Brain stopped on port ${PORT}."
