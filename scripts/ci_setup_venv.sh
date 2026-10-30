#!/bin/bash
# Create a local .venv using the runner's system Python.
# Avoids actions/setup-python, which hardcodes /Users/runner/hostedtoolcache
# on macOS and fails on self-hosted runners that are not the `runner` user.
set -Eeuo pipefail

test "$(uname -m)" = "arm64" || {
  echo "CI requires an Apple Silicon arm64 runner for dlib-bin." >&2
  exit 1
}

ARCH="$(uname -m)"
PYTHON="$(scripts/select_python.sh "$ARCH")"
echo "Selected Python: ${PYTHON}"
"$PYTHON" --version
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools<81" wheel
PYTHON_VERSION="$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
DLIB_BIN_VERSION="$(scripts/dlib_bin_version.sh "$PYTHON_VERSION" "$ARCH")"
echo "Using dlib-bin ${DLIB_BIN_VERSION} for Python ${PYTHON_VERSION} on ${ARCH}"
python - <<'PY'
from pathlib import Path
requirements = Path("requirements.txt").read_text().splitlines()
Path("/tmp/gemi-requirements.txt").write_text(
    "\n".join(line for line in requirements
              if line.strip().lower() != "face_recognition") + "\n"
)
PY
python -m pip install --no-cache-dir -r /tmp/gemi-requirements.txt
python -m pip install --no-cache-dir "dlib-bin==${DLIB_BIN_VERSION}"
python -m pip install --no-cache-dir --no-deps face-recognition==1.3.0
python -m pip install --no-cache-dir face-recognition-models==0.3.0
python -c "import dlib, face_recognition; print('face recognition dependencies ready')"
