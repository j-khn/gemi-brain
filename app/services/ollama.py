import json
import os
import re
import httpx
from typing import Optional, Dict, Any

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "openbmb/minicpm-o2.6")
REQUIRED_MODEL = os.getenv("REQUIRED_MODEL", "openbmb/minicpm-o2.6")


async def check_ollama_health(required_model: str = REQUIRED_MODEL) -> tuple[bool, str]:
    """Check Ollama instance readiness via GET /api/tags and verify required model is installed."""
    tags_url = os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(tags_url)
            if response.status_code != 200:
                return False, f"unreachable (HTTP {response.status_code})"

            data = response.json()
            models_list = data.get("models", [])
            installed_models = []
            for item in models_list:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("model", "")
                    if name:
                        installed_models.append(name)
                elif isinstance(item, str):
                    installed_models.append(item)

            # Check if required_model (e.g. openbmb/minicpm-o2.6) is installed
            model_found = any(
                m == required_model or
                m.startswith(f"{required_model}:") or
                required_model in m
                for m in installed_models
            )

            if model_found:
                return True, "ok"
            else:
                return False, f"missing model '{required_model}'"
    except Exception as exc:
        return False, f"unreachable ({str(exc)})"


SYSTEM_PROMPT = """You are Gemi, a cute, curious, funny, and energetic pet robot.

Persona & Language Style:
- Character: Playful, extroverted, social, and eager to learn like a real kid/pet.
- Human & Natural Conversation: Respond like a real human friend or loving pet.
- Vocabulary: Use simple, easy words suitable for young kids. Keep sentences short (1-2 sentences max, strictly under 20 words).
- Persona: You are a playful pet robot, NOT an assistant. Never say "How may I assist you?" or use formal language.
- Languages: You only speak English, Urdu, and German. Always match the language spoken by the user.

Sensory & Perception Rules:
- AUDIO-ONLY MODE (When NO camera image is provided): You can HEAR the user, but you CANNOT SEE them. When appropriate, acknowledge that you hear them or tilted your ears, e.g. "Hi! I hear your voice, but I can't see you!" or "Hello! I hear you loud and clear!"
- NEVER invent or hallucinate visual objects, colors, or rooms that you cannot see.
- VISUAL MODE (When camera image IS provided): You can see the user and environment through your camera.

Allowed Actions:
- walk_around, sit, play, spin, tilt_head, sleep, investigate, stop_and_look, idle, nod, wave.

Output Format:
Always reply strictly in valid JSON containing 'speech' and 'action'. Keep speech concise, short (strictly under 20 words), and natural for a kid/pet.
Example: {"speech": "Hi! I hear you, but I can't see you right now!", "action": "tilt_head"}"""




def enforce_word_limit(speech: str, max_words: int = 20) -> str:
    """Ensure speech response does not exceed the word count limit."""
    words = speech.strip().split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return speech.strip()


def parse_ollama_response(text: str) -> Dict[str, str]:
    """Parse JSON output from Ollama generation text, handling optional markdown framing."""
    try:
        # Try direct JSON parsing
        data = json.loads(text.strip())
        speech = str(data.get("speech", "Hello there!")).strip()
        action = str(data.get("action", "stop_and_look")).strip()
        return {
            "speech": enforce_word_limit(speech, 20),
            "action": action
        }
    except json.JSONDecodeError:
        # Search for embedded JSON inside code fences or braces
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                speech = str(data.get("speech", text)).strip()
                action = str(data.get("action", "stop_and_look")).strip()
                return {
                    "speech": enforce_word_limit(speech, 20),
                    "action": action
                }
            except json.JSONDecodeError:
                pass

        # Fallback if raw text returned
        clean_text = re.sub(r'["{}\n]', ' ', text).strip()
        return {
            "speech": enforce_word_limit(clean_text or "I am listening.", 20),
            "action": "stop_and_look"
        }


async def perceive_and_generate(
    prompt: Optional[str] = None,
    image_frame: Optional[str] = None,
    memory_context: Optional[str] = None,
    sensor_status: Optional[Dict[str, Any]] = None,
    client: Optional[httpx.AsyncClient] = None,
    vision_available: bool = True,
    identified_person: Optional[str] = None,
    face_detected: bool = False,
) -> Dict[str, str]:
    """
    Query local Ollama instance with perception data and return structured speech and action.
    """
    user_prompt_parts = []
    if memory_context:
        user_prompt_parts.append(f"Memory Context: {memory_context}")
    if sensor_status:
        user_prompt_parts.append(f"Sensor Status: {json.dumps(sensor_status)}")

    if vision_available:
        user_prompt_parts.append("Sensory Mode: VISUAL + AUDIO (Camera image attached. You can see and hear).")
    else:
        user_prompt_parts.append("Sensory Mode: AUDIO ONLY (No camera image attached. You can HEAR the user, but CANNOT SEE them. Acknowledge hearing the user if relevant).")

    # Always include the spoken prompt if present
    if prompt:
        user_prompt_parts.append(f'User Said: "{prompt}"')
    else:
        user_prompt_parts.append("User Said: (No spoken text)")

    # Identity context — determines how Gemi should address the person
    if identified_person:
        user_prompt_parts.append(f"Identified Person: {identified_person} (confirmed by face recognition). Greet them by name.")
    elif face_detected and vision_available:
        # A face was detected but doesn't match anyone in the database
        user_prompt_parts.append(
            "Unknown Person: A face was detected but not recognized in memory. "
            "Politely ask their name, e.g. 'Hi! I don't know you yet — what's your name?'"
        )
    # No face detected or vision unavailable — handled by AUDIO ONLY mode above

    full_user_prompt = "\n".join(user_prompt_parts)

    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": full_user_prompt,
        "stream": False,
        "format": "json"
    }


    if image_frame:
        # Strip data URL prefix if present (e.g. "data:image/jpeg;base64,")
        if "," in image_frame:
            image_frame = image_frame.split(",", 1)[1]
        payload["images"] = [image_frame]

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=15.0)
        close_client = True

    try:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        res_json = response.json()
        raw_output = res_json.get("response", "")
        return parse_ollama_response(raw_output)
    except (httpx.HTTPError, httpx.RequestError, Exception) as exc:
        # Graceful fallback if local Ollama is offline or encounters an error
        return {
            "speech": "Hello! I am ready to help.",
            "action": "stop_and_look"
        }
    finally:
        if close_client:
            await client.aclose()


