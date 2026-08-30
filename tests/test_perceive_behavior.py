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

    assert result["action"] == "turn_around"
    assert "there" in result["speech"].lower() or "turn" in result["speech"].lower()


def test_obstacle_distance_threshold_triggers_guard():
    result = _run(
        prompt="Go ahead",
        sensor_status={"obstacle_distance": 12},
        model_response={"speech": "Walking forward now.", "action": "walk_around"},
    )

    assert result["action"] == "turn_around"


def test_multimodal_synthesis_prompt_includes_vectors():
    captured = {}

    async def handler(request: httpx.Request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"response": '{"speech":"Hey! I see a shelf!","action":"investigate"}'},
        )

    async def run_call():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await ollama.perceive_and_generate(
                client=client,
                prompt="hey what do you see?",
                image_frames=["aGVsbG8="],
                sensor_status={"obstacle": True},
            )
        finally:
            await client.aclose()

    asyncio.run(run_call())
    prompt = captured["prompt"]
    assert "Vector A (USER TEXT)" in prompt
    assert "Vector B (VISUAL FRAME)" in prompt
    assert "Vector C (OBSTACLE)" in prompt
    assert "MULTI-INPUT SYNTHESIS REQUIRED" in prompt


def test_vision_collapse_guard_replaces_image_only_speech():
    result = _run(
        prompt="hey what do you see?",
        image_frames=["aGVsbG8="],
        sensor_status={"obstacle": True},
        model_response={
            "speech": "I see a room with pink flooring.",
            "action": "idle",
        },
    )

    assert "hey" in result["speech"].lower() or "look" in result["speech"].lower()
    assert any(
        marker in result["speech"].lower()
        for marker in ("front", "block", "way", "something", "though")
    )
    assert result["action"] in ollama._OBSTACLE_AWARE_ACTIONS


def test_obstacle_near_upgrades_generic_action():
    result = _run(
        prompt="look at this!",
        image_frames=["aGVsbG8="],
        sensor_status={"obstacle": True},
        model_response={
            "speech": "Whoa, cool picture!",
            "action": "idle",
        },
    )

    assert result["action"] == "stop_and_inspect"


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


def test_movement_intent_adds_action_guidance_to_prompt():
    captured = {}

    async def handler(request: httpx.Request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"response": '{"speech":"Backing up!","action":"move_back"}'},
        )

    async def run_call():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await ollama.perceive_and_generate(
                client=client,
                prompt="move back",
            )
        finally:
            await client.aclose()

    result = asyncio.run(run_call())
    assert "Movement command detected" in captured["prompt"]
    assert "move_back" in captured["prompt"]
    assert result["action"] == "move_back"


def test_move_back_survives_obstacle_guard():
    result = _run(
        prompt="move back",
        sensor_status={"obstacle": True},
        model_response={"speech": "Backing up now.", "action": "move_back"},
    )
    assert result["action"] == "move_back"


def test_nearest_distance_cm_triggers_obstacle_guard():
    result = _run(
        prompt="go forward",
        sensor_status={"obstacle": False, "nearest_distance_cm": 10},
        model_response={"speech": "Going forward.", "action": "move_forward"},
    )
    assert result["action"] == "turn_around"


def test_approach_action_is_allowed():
    assert "approach" in ollama._get_allowed_actions()
    parsed = ollama.parse_ollama_response('{"speech":"Heading there!","action":"approach"}')
    assert parsed["action"] == "approach"
