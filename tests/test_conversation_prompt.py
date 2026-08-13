"""Tests for casual conversation prompting and memory continuity."""

import asyncio
import json
import os

import httpx

os.environ.setdefault("DB_PATH", "test_gemi_memory.db")

from app.db import database
from app.services import ollama


def _capture_prompt(**kwargs):
    captured = {}

    async def handler(request: httpx.Request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"response": '{"speech":"Okay.","action":"nod"}'},
        )

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await ollama.perceive_and_generate(client=client, **kwargs)
        finally:
            await client.aclose()

    asyncio.run(run())
    return captured["system"], captured["prompt"]


def test_system_prompt_encourages_human_memory_and_feelings():
    system, _ = _capture_prompt(prompt="Look around")

    assert "remember them the way a person would" in system
    assert "does not erase what you already noticed" in system
    assert "Never say \"as an AI\"" in system or "as an AI" in system
    assert "my sensors" in system
    assert "Greet at most once" in system


def test_ongoing_chat_does_not_force_a_greeting():
    history = [
        {"user_prompt": "Look around", "gemi_response": "Hi Alice!"},
    ]
    _, prompt = _capture_prompt(
        prompt="What do you see?",
        conversation_history=history,
        visible_people=["Alice"],
        vision_available=True,
    )

    assert "do not greet again" in prompt
    assert "Greet them by name" not in prompt
    assert "This chat is already ongoing" in prompt


def test_first_meeting_allows_a_short_greeting():
    _, prompt = _capture_prompt(
        prompt="Look around",
        conversation_history=[],
        visible_people=["Alice"],
        vision_available=True,
    )

    assert "first sighting this chat" in prompt
    assert "short greeting" in prompt


def test_no_current_image_still_keeps_past_memory():
    history = [
        {
            "user_prompt": "what is this? [shared a picture]",
            "gemi_response": "Cute dog!",
        }
    ]
    _, prompt = _capture_prompt(
        prompt="What did you see earlier?",
        conversation_history=history,
        recent_notes=["Prompt: what is this? [shared a picture] -> Speech: 'Cute dog!'"],
        vision_available=False,
    )

    assert "Cute dog!" in prompt
    assert "Recent notes from earlier moments" in prompt
    assert "remember what you noticed earlier" in prompt
    assert "Your eyes are closed" not in prompt
    assert "cannot see right now" not in prompt


def test_recent_interaction_logs_are_readable():
    if os.path.exists("test_gemi_memory.db"):
        os.remove("test_gemi_memory.db")
    database.init_db()
    database.save_interaction_log("Saw something cute earlier")
    notes = database.get_recent_interaction_logs(limit=5, minutes=60)
    assert notes
    assert "Saw something cute earlier" in notes[-1]
