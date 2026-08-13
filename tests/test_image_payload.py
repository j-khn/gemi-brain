"""Regression tests for JSON base64 images reaching Ollama."""

import asyncio
import base64
import io
import json

import httpx
import pytest
from PIL import Image

from app.services import ollama, vision


def jpeg_base64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (220, 30, 30)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_normalize_image_base64_accepts_data_url_and_whitespace():
    image = jpeg_base64()
    wrapped = f"data:image/jpeg;base64,\n{image[:20]} \n{image[20:]}"

    normalized = vision.normalize_image_base64(wrapped)

    assert normalized == image
    assert base64.b64decode(normalized).startswith(b"\xff\xd8\xff")


def test_normalize_image_base64_rejects_invalid_input():
    with pytest.raises(ValueError, match="not valid base64"):
        vision.normalize_image_base64("not an image base64 value")


def test_ollama_payload_contains_the_received_image():
    image = jpeg_base64()
    captured = {}

    async def handler(request: httpx.Request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"response": '{"speech":"I see red.","action":"nod"}'},
        )

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await ollama.perceive_and_generate(
                prompt="What do you see?",
                image_frames=[f"data:image/jpeg;base64,{image}"],
                client=client,
                vision_available=True,
            )
        finally:
            await client.aclose()

    result = asyncio.run(run())

    assert result["speech"] == "I see red."
    assert captured["images"] == [image]
    assert "multimodal images input" in captured["prompt"]
