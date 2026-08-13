#!/bin/bash
# Print the dlib-bin wheel version compatible with a Python/macOS pair.
#
# Usage:
#   scripts/dlib_bin_version.sh 3.14 arm64
set -Eeuo pipefail

PYTHON_VERSION="${1:-}"
ARCH="${2:-}"

case "${ARCH}:${PYTHON_VERSION}" in
  arm64:3.14)
    # 19.24.6 has no cp314 wheel; 20.0.1 does.
    printf '%s\n' "20.0.1"
    ;;
  arm64:3.10|arm64:3.11|arm64:3.12|arm64:3.13|\
  x86_64:3.10|x86_64:3.11|x86_64:3.12|x86_64:3.13)
    printf '%s\n' "19.24.6"
    ;;
  *)
    printf 'Unsupported dlib-bin combination: Python %s on %s\n' \
      "${PYTHON_VERSION:-unknown}" "${ARCH:-unknown}" >&2
    exit 1
    ;;
esac
