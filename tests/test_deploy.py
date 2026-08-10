import importlib
import os

import pytest


def test_main_reads_host_and_port_from_env(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "12345")

    import main
    importlib.reload(main)

    assert main.HOST == "0.0.0.0"
    assert main.PORT == 12345


def test_main_defaults_to_localhost_and_8000(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)

    import main
    importlib.reload(main)

    assert main.HOST == "127.0.0.1"
    assert main.PORT == 8000
