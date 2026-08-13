"""Tests for selecting a native dlib-bin wheel."""

import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "dlib_bin_version.sh"


@pytest.mark.parametrize(
    ("python_version", "architecture", "expected"),
    [
        ("3.10", "arm64", "19.24.6"),
        ("3.11", "arm64", "19.24.6"),
        ("3.12", "arm64", "19.24.6"),
        ("3.13", "arm64", "19.24.6"),
        ("3.14", "arm64", "20.0.1"),
        ("3.10", "x86_64", "19.24.6"),
        ("3.11", "x86_64", "19.24.6"),
        ("3.12", "x86_64", "19.24.6"),
        ("3.13", "x86_64", "19.24.6"),
    ],
)
def test_selects_available_wheel(python_version, architecture, expected):
    result = subprocess.run(
        [str(SCRIPT), python_version, architecture],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    ("python_version", "architecture"),
    [
        ("3.9", "arm64"),
        ("3.14", "x86_64"),
        ("3.15", "arm64"),
    ],
)
def test_rejects_unsupported_wheel(python_version, architecture):
    result = subprocess.run(
        [str(SCRIPT), python_version, architecture],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Unsupported dlib-bin combination" in result.stderr
    assert "Supported:" in result.stderr
