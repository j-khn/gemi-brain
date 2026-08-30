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


def test_obstacle_distance_above_10cm_does_not_trigger_guard():
    result = _run(
        prompt="Go ahead",
        sensor_status={"obstacle_distance": 12},
        model_response={"speech": "Walking forward now.", "action": "walk_around"},
    )

    assert result["action"] == "walk_around"


def test_obstacle_distance_at_10cm_triggers_guard():
    result = _run(
        prompt="Go ahead",
        sensor_status={"obstacle_distance": 10},
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


def test_short_non_visual_answer_survives_with_image():
    result = _run(
        prompt="what is your name",
        image_frames=["aGVsbG8="],
        vision_color_hint="red",
        model_response={"speech": "Gemi!", "action": "nod"},
    )
    assert result["speech"] == "Gemi!"
    assert result["action"] == "nod"
    assert "mostly red" not in result["speech"].lower()


def test_vision_only_reply_to_non_visual_question_is_rewritten():
    result = _run(
        prompt="what is your name",
        image_frames=["aGVsbG8="],
        vision_color_hint="red",
        model_response={"speech": "I see a red wall.", "action": "investigate"},
    )
    assert "mostly red" not in result["speech"].lower()
    assert "red wall" not in result["speech"].lower()
    assert "good question" in result["speech"].lower()


def test_generic_non_visual_question_keeps_short_answer():
    result = _run(
        prompt="how are you feeling",
        image_frames=["aGVsbG8="],
        vision_color_hint="red",
        model_response={"speech": "Pretty good!", "action": "nod"},
    )
    assert result["speech"] == "Pretty good!"
    assert "mostly red" not in result["speech"].lower()


def test_visual_question_keeps_vision_heavy_answer():
    result = _run(
        prompt="what do you see?",
        image_frames=["aGVsbG8="],
        model_response={"speech": "I see a red wall!", "action": "investigate"},
    )
    assert "red wall" in result["speech"].lower()
    assert result["action"] == "investigate"


def test_im_gemi_is_allowed_for_name_question():
    result = _run(
        prompt="what is your name",
        model_response={"speech": "I'm Gemi!", "action": "nod"},
    )
    assert "gemi" in result["speech"].lower()
    assert result["action"] == "nod"


def test_grief_toxic_positivity_rewritten_to_sit_with():
    result = _run(
        prompt="my pet died and I'm sad",
        negative_valence=True,
        model_response={"speech": "Cheer up, look on the bright side!", "action": "nod"},
    )
    assert result["speech"] == "Aw, that's hard."
    assert result["action"] == "nod"


def test_sad_prompt_downgrades_spin_action():
    result = _run(
        prompt="I'm really sad today",
        negative_valence=True,
        model_response={"speech": "Aw…", "action": "spin"},
    )
    assert result["action"] == "nod"


def test_explicit_emotion_label_simplified():
    result = _run(
        prompt="how are you?",
        model_response={"speech": "I feel sadness right now.", "action": "sit"},
    )
    assert result["speech"] == "Aw…"
    assert result["action"] == "nod"


def test_play_invite_upgrades_idle_to_play():
    result = _run(
        prompt="let's play!",
        model_response={"speech": "Yay!", "action": "idle"},
    )
    assert result["action"] == "play"


def test_repair_warmth_hint_after_hostile_turn():
    captured = {}

    async def handler(request: httpx.Request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"response": '{"speech":"Hey!","action":"nod"}'},
        )

    async def run_call():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await ollama.perceive_and_generate(
                client=client,
                prompt="hey",
                conversation_history=[
                    {
                        "user_prompt": "shut up",
                        "gemi_response": "Whoa, not cool.",
                    }
                ],
            )
        finally:
            await client.aclose()

    asyncio.run(run_call())
    assert "normal again" in captured["prompt"].lower() or "no grudge" in captured["prompt"].lower()


def test_negative_valence_hint_in_prompt():
    captured = {}

    async def handler(request: httpx.Request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"response": '{"speech":"Aw, that stinks.","action":"nod"}'},
        )

    async def run_call():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await ollama.perceive_and_generate(
                client=client,
                prompt="I'm sad",
                negative_valence=True,
            )
        finally:
            await client.aclose()

    asyncio.run(run_call())
    assert "upset" in captured["prompt"].lower() or "sit with" in captured["prompt"].lower()


def test_play_invite_forces_play_action_even_from_investigate():
    result = _run(
        prompt="let's play!",
        model_response={"speech": "Sure thing!", "action": "investigate"},
    )
    assert result["action"] == "play"
    assert "oh wow" in result["speech"].lower()


def test_play_invite_adds_wonder_tone_to_flat_reply():
    result = _run(
        prompt="let's play!",
        model_response={"speech": "Yay! Let's have some fun together!", "action": "play"},
    )
    assert result["action"] == "play"
    assert "oh wow" in result["speech"].lower()


def test_greeting_prompt_stays_brief():
    result = _run(
        prompt="hey",
        model_response={"speech": "Hey there! Good to see you again. How's your day going?", "action": "nod"},
    )
    assert result["speech"] == "Hey there!"


def test_grief_apology_rewritten_to_sit_with():
    result = _run(
        prompt="my hamster died",
        negative_valence=True,
        model_response={"speech": "I'm so sorry to hear that.", "action": "nod"},
    )
    assert result["speech"] == "Aw, that's hard."


def test_approach_action_is_allowed():
    assert "approach" in ollama._get_allowed_actions()
    parsed = ollama.parse_ollama_response('{"speech":"Heading there!","action":"approach"}')
    assert parsed["action"] == "approach"


def test_spin_survives_obstacle_guard():
    result = _run(
        prompt="spin!",
        sensor_status={"obstacle": True},
        model_response={"speech": "Whee, spinning!", "action": "spin"},
    )
    assert result["action"] == "spin"


def test_play_rewrites_to_play_with_obstacle_when_blocked():
    result = _run(
        prompt="let's play",
        sensor_status={"obstacle": True},
        model_response={"speech": "Let's play!", "action": "play"},
    )
    assert result["action"] == "play_with_obstacle"


def test_spin_and_play_survive_clear_12cm_path():
    spin = _run(
        prompt="spin!",
        sensor_status={"obstacle": False, "nearest_distance_cm": 12},
        model_response={"speech": "Whee, spinning!", "action": "spin"},
    )
    play = _run(
        prompt="let's play",
        sensor_status={"obstacle": False, "obstacle_distance": 12},
        model_response={"speech": "Let's play!", "action": "play"},
    )
    assert spin["action"] == "spin"
    assert play["action"] == "play"


def test_spin_and_play_intent_adds_action_guidance_to_prompt():
    captured = {}

    async def handler(request: httpx.Request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"response": '{"speech":"Whee, spinning!","action":"spin"}'},
        )

    async def run_call(prompt: str):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await ollama.perceive_and_generate(client=client, prompt=prompt)
        finally:
            await client.aclose()

    result = asyncio.run(run_call("spin!"))
    assert "Movement command detected" in captured["prompt"]
    assert "spin" in captured["prompt"]
    assert result["action"] == "spin"

    captured.clear()

    async def play_handler(request: httpx.Request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"response": '{"speech":"Let\'s play!","action":"play"}'},
        )

    async def run_play():
        client = httpx.AsyncClient(transport=httpx.MockTransport(play_handler))
        try:
            return await ollama.perceive_and_generate(client=client, prompt="let's play")
        finally:
            await client.aclose()

    play_result = asyncio.run(run_play())
    assert "Movement command detected" in captured["prompt"]
    assert "play" in captured["prompt"]
    assert play_result["action"] == "play"
