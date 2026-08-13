"""Tests for casual conversation prompting."""

import asyncio
import json

import httpx

from app.services import ollama


def _capture_prompt(**kwargs) -> str:
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


def test_system_prompt_discourages_repeated_greetings():
    system, _ = _capture_prompt(prompt="Look around")

    assert "Greet at most once" in system
    assert "do NOT greet again" in system
    assert "Avoid stock robot openers" in system
    assert "Hi there! What are you up to today?" in system


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
