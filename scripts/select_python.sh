#!/bin/bash
# Prefer a Python interpreter that has a matching dlib-bin wheel.
#
# Usage:
#   scripts/select_python.sh
#   scripts/select_python.sh arm64
set -Eeuo pipefail

ARCH="${1:-}"
if [ -z "$ARCH" ]; then
  if [ "$(uname -s)" = "Darwin" ] &&
    [ "$(sysctl -in hw.optional.arm64 2>/dev/null || true)" = "1" ]; then
    ARCH="arm64"
  else
    ARCH="$(uname -m)"
  fi
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERSION_SCRIPT="${SCRIPT_DIR}/dlib_bin_version.sh"

CANDIDATES=(
  python3.14
  python3.13
  python3.12
  python3.11
  python3.10
  python3
  python
)

for candidate in "${CANDIDATES[@]}"; do
  if ! command -v "$candidate" >/dev/null 2>&1; then
    continue
  fi

  PYTHON_BIN="$(command -v "$candidate")"
  PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
  if "$VERSION_SCRIPT" "$PYTHON_VERSION" "$ARCH" >/dev/null 2>&1; then
    printf '%s\n' "$PYTHON_BIN"
    exit 0
  fi
done

printf 'No supported Python interpreter found for dlib-bin on %s.\n' "$ARCH" >&2
printf 'Install Python 3.10–3.14 (Apple Silicon) or 3.10–3.13 (Intel) and retry.\n' >&2
exit 1
