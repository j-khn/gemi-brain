"""Tests for selecting a Python interpreter with a matching dlib wheel."""

import os
import platform
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "select_python.sh"

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="select_python prefers macOS Python installs used by this project",
)


def test_selects_supported_python_on_this_machine():
    result = subprocess.run(
        [str(SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
    )

    python_bin = result.stdout.strip()
    assert python_bin
    assert Path(python_bin).exists()
    assert os.access(python_bin, os.X_OK)

    version = subprocess.run(
        [python_bin, "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    major, minor = map(int, version.split("."))
    assert (major, minor) >= (3, 10)
    assert (major, minor) <= (3, 14)


def test_reports_failure_when_no_supported_python_is_available(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))

    result = subprocess.run(
        [str(SCRIPT), "arm64"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": str(tmp_path)},
    )

    assert result.returncode != 0
    assert "No supported Python interpreter found" in result.stderr
