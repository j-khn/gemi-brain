"""Tests for the Cursor Python tooling repair script."""

import os
import platform
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "repair_cursor_python_tools.sh"

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="the repair script targets macOS editor installs",
)

CURSOR_EXTENSIONS = (
    "ms-python.python-2025.4.0",
    "ms-python.vscode-python-envs-1.36.0-universal",
)


def build_sandbox(root: Path, with_editor_source: bool = True) -> dict:
    """Create fake Cursor/VS Code extension trees for the script to repair."""
    cursor_dir = root / "cursor"
    vscode_dir = root / "vscode"
    cache_dir = root / "cache"

    for name in CURSOR_EXTENSIONS:
        (cursor_dir / name).mkdir(parents=True)
    (cursor_dir / "extensions.json").write_text("[]")

    if with_editor_source:
        source = vscode_dir / "ms-python.vscode-python-envs-1.36.0-darwin-arm64"
        pet = source / "python-env-tools" / "bin" / "pet"
        pet.parent.mkdir(parents=True)
        pet.write_text('#!/bin/bash\necho "pet 0.1.0"\n')
        pet.chmod(0o755)

    return {
        "CURSOR_EXTENSIONS_DIR": str(cursor_dir),
        "VSCODE_EXTENSIONS_DIR": str(vscode_dir),
        "PET_CACHE_DIR": str(cache_dir),
    }


def run_script(sandbox: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, **sandbox},
        capture_output=True,
        text=True,
    )


def installed_pets(sandbox: dict) -> list:
    cursor_dir = Path(sandbox["CURSOR_EXTENSIONS_DIR"])
    return [cursor_dir / name / "python-env-tools" / "bin" / "pet" for name in CURSOR_EXTENSIONS]


def test_restores_pet_for_every_python_extension(tmp_path):
    sandbox = build_sandbox(tmp_path)

    result = run_script(sandbox)

    assert result.returncode == 0, result.stdout + result.stderr
    for pet in installed_pets(sandbox):
        assert pet.is_file()
        assert os.access(pet, os.X_OK)
        assert "pet" in subprocess.run(
            [str(pet), "--version"], capture_output=True, text=True
        ).stdout


def test_repeated_runs_leave_a_healthy_install_untouched(tmp_path):
    sandbox = build_sandbox(tmp_path)
    assert run_script(sandbox).returncode == 0
    before = [pet.stat().st_mtime_ns for pet in installed_pets(sandbox)]

    result = run_script(sandbox)

    assert result.returncode == 0
    assert "no action needed" in result.stdout
    assert [pet.stat().st_mtime_ns for pet in installed_pets(sandbox)] == before


def test_restores_from_cache_after_editor_source_disappears(tmp_path):
    """An extension update must be repairable without VS Code still installed."""
    sandbox = build_sandbox(tmp_path)
    assert run_script(sandbox).returncode == 0

    for pet in installed_pets(sandbox):
        pet.unlink()
    for stale in Path(sandbox["VSCODE_EXTENSIONS_DIR"]).glob("*"):
        subprocess.run(["rm", "-rf", str(stale)], check=True)

    result = run_script(sandbox)

    assert result.returncode == 0, result.stdout + result.stderr
    for pet in installed_pets(sandbox):
        assert pet.is_file()
        assert os.access(pet, os.X_OK)


def test_reports_failure_when_no_pet_is_available(tmp_path):
    sandbox = build_sandbox(tmp_path, with_editor_source=False)

    result = run_script(sandbox)

    assert result.returncode != 0
    assert "no usable pet available" in result.stdout
    for pet in installed_pets(sandbox):
        assert not pet.exists()
