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

    # Core identity and first-person voice
    assert "first person" in system.lower()
    # Memory / sensing
    assert "primary truth" in system
    # Anti-assistant markers present
    assert "as an ai" in system.lower()
    assert "my sensors" in system
    # Greeting control
    assert "Greet at most once" in system or "greet at most once" in system.lower()
    # Anti-apology on hostile input
    assert "Do NOT apologize" in system or "do not apologize" in system.lower()
    # Colour grounding / vision uncertainty mentioned
    assert "unclear" in system.lower() or "blurry" in system.lower()
    # Multi-input synthesis
    assert "multi-input synthesis" in system.lower() or "vector a" in system.lower()
    assert "anti-vision-collapse" in system.lower() or "vision-collapse" in system.lower()


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
    standout = [
        {
            "user_prompt": "what is this? [shared a picture]",
            "gemi_response": "Cute dog!",
            "timestamp": "2026-08-13 21:00:00",
        }
    ]
    _, prompt = _capture_prompt(
        prompt="What did you see earlier?",
        conversation_history=history,
        recent_notes=["Prompt: what is this? [shared a picture] -> Speech: 'Cute dog!'"],
        standout_moments=standout,
        vision_available=False,
    )

    assert "Cute dog!" in prompt
    assert "Most recent visual tip" in prompt
    assert "Recent notes from earlier moments" in prompt
    assert "Stay grounded in what already happened" in prompt
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


def test_history_prep_drops_same_question_repeats_and_keeps_standouts():
    if os.path.exists("test_gemi_memory.db"):
        os.remove("test_gemi_memory.db")
    database.init_db()
    database.save_conversation_turn(
        None, "what is this? [shared a picture]", "Cute dog on the phone!"
    )
    database.save_conversation_turn(
        None, "I mean did you see anything earlier?", "Nope, my sensors are focused on audio right now."
    )
    database.save_conversation_turn(
        None, "Look around and decide how to interact.", "Got it, I'll take a look around."
    )
    database.save_conversation_turn(
        None, "Look around and decide how to interact.", "Got it, I'll take a look around."
    )

    standouts = database.get_standout_conversation_moments(limit=5, minutes=60)
    assert standouts
    assert any("Cute dog" in (t["gemi_response"] or "") for t in standouts)

    related = database.get_conversation_history_by_query(
        "I mean did you see anything earlier?", limit=6
    )
    assert all(
        database._normalize_prompt_text(t["user_prompt"])
        != database._normalize_prompt_text("I mean did you see anything earlier?")
        for t in related
    )

    prepared = database.prepare_conversation_history_for_prompt(
        related
        + database.get_recent_conversation_history(limit=10, minutes=60)
        + [
            {
                "user_prompt": "I mean did you see anything earlier?",
                "gemi_response": "Nope, my sensors are focused on audio right now.",
                "timestamp": "2026-08-13 21:07:28",
            }
        ],
        current_prompt="I mean did you see anything earlier?",
        limit=12,
    )
    assert prepared
    assert all(
        database._normalize_prompt_text(t["user_prompt"])
        != database._normalize_prompt_text("I mean did you see anything earlier?")
        for t in prepared
    )
