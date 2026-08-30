"""Tests for visual variation and obstacle movement guards."""

import asyncio
import json
import os

import httpx

os.environ.setdefault("DB_PATH", "test_gemi_memory.db")

from app.services import ollama


def _run(**kwargs):
    model_response = kwargs.pop(
        "model_response",
        {"speech": "Okay.", "action": "nod"},
    )

    async def handler(request: httpx.Request):
        return httpx.Response(
            200,
            json={"response": json.dumps(model_response)},
        )

    async def run_call():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await ollama.perceive_and_generate(client=client, **kwargs)
        finally:
            await client.aclose()

    return asyncio.run(run_call())


def test_obstacle_near_blocks_walk_through():
    result = _run(
        prompt="Look around",
        sensor_status={"obstacle": True, "battery": 90},
        model_response={
            "speech": "I'll walk through and keep going.",
            "action": "walk_around",
        },
    )

    assert result["action"] == "tilt_head"
    assert "turn" in result["speech"].lower() or "there" in result["speech"].lower()


def test_obstacle_distance_threshold_triggers_guard():
    result = _run(
        prompt="Go ahead",
        sensor_status={"obstacle_distance": 12},
        model_response={"speech": "Walking forward now.", "action": "play"},
    )

    assert result["action"] == "tilt_head"


def test_similar_scene_prompt_mentions_fresh_reaction():
    captured = {}

    async def handler(request: httpx.Request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"response": '{"speech":"New angle!","action":"investigate"}'},
        )

    async def run_call():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await ollama.perceive_and_generate(
                client=client,
                prompt="What do you see?",
                image_frames=["aGVsbG8="],
                vision_similar_scene=True,
                last_visual_speech="Oh wow, I see a room with pink flooring.",
            )
        finally:
            await client.aclose()

    asyncio.run(run_call())
    prompt = captured["prompt"]
    assert "NOT the same frame" in prompt
    assert "pink flooring" in prompt


def test_repeated_visual_speech_is_replaced():
    result = _run(
        prompt="Look around",
        image_frames=["aGVsbG8="],
        last_visual_speech="Oh wow, I see a room with pink flooring.",
        model_response={
            "speech": "Oh wow, I see a room with pink flooring.",
            "action": "investigate",
        },
    )

    assert "pink flooring" not in result["speech"].lower()
    assert result["action"] == "investigate"


def test_obstacle_is_near_accepts_common_sensor_shapes():
    assert ollama._obstacle_is_near({"obstacle": True})
    assert ollama._obstacle_is_near({"obstacle_distance": 20})
    assert ollama._obstacle_is_near({"distance": 30})
    assert not ollama._obstacle_is_near({"obstacle": False, "obstacle_distance": 80})
