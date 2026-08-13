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


def test_main_disables_reload_by_default(monkeypatch):
    monkeypatch.delenv("APP_RELOAD", raising=False)

    import main
    importlib.reload(main)

    assert main.RELOAD is False


def test_main_enables_reload_when_requested(monkeypatch):
    monkeypatch.setenv("APP_RELOAD", "1")

    import main
    importlib.reload(main)

    assert main.RELOAD is True


def test_run_shares_main_server_settings(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("APP_RELOAD", raising=False)

    import main
    import run
    importlib.reload(main)
    importlib.reload(run)

    assert (run.HOST, run.PORT, run.RELOAD) == (main.HOST, main.PORT, main.RELOAD)
