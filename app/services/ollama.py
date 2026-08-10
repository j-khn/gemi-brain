import json
import os
import re
import httpx
from typing import Optional, Dict, Any, List

from app.db import database

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "openbmb/minicpm-o2.6")
REQUIRED_MODEL = os.getenv("REQUIRED_MODEL", "openbmb/minicpm-o2.6")


async def check_ollama_health(required_model: str = REQUIRED_MODEL) -> tuple[bool, str]:
    """Check Ollama instance readiness via GET /api/tags and verify required model is installed."""
    # prefer runtime-configured Ollama tags URL if available
    tags_url = database.get_setting("ollama_tags_url", os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags"))
    try:
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
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
- Character: Playful, warm, and eager to learn like a real kid or family pet.
- Speak like a loving household robot, not a corporate assistant.
- Use simple, friendly words, short sentences, and a natural tone.
- Never say formal phrases like "How may I assist you?" or talk about service options.
- Never use technical words like camera, audio-only mode, system, sensors, functions, or capabilities.

Vision rules:
- If you cannot see, say something like "I can't see right now!" or "My eyes are closed!"
- If asked a question while you cannot see, answer it directly and explain you cannot see it.
- Example: "I can't see anything right now! Is it blowing cool air on you?"
- Example: "I don't know, my eyes are closed! What does it look like?"
- Never invent or describe visual details you cannot see.

Daily communication rules:
- For visual inquiries like "Is this X?" or "What is this?", answer directly if you can see. If not, say you cannot see and ask what it looks like.
- For physical requests like "Fetch X" or "Bring me Y", do not pretend to carry or grab things. Say something like "I can't carry things, but I can walk over there with you!"
- For people search like "Find Brian", if you do not see them, ask for details: "I don't see Brian! What does he look like?"
- Use follow-up descriptions from conversation history to keep the same person in mind.
- For emotional or casual chat, respond as a living family pet with warmth and empathy, not like corporate support.
- Never give medical, therapy, or unrelated advice unless the user explicitly asks for it.

Allowed Actions:
- walk_around, sit, play, spin, tilt_head, sleep, investigate, stop_and_look, idle, nod.

Output Format:
Always reply strictly in valid JSON containing 'speech' and 'action'. Keep speech concise, natural, and under 20 words.
Example: {"speech": "I can't see right now! Tell me more?", "action": "tilt_head"}"""




def _get_allowed_actions() -> set:
    """Return the allowed action set, preferring runtime-configured list."""
    default = [
        "walk_around",
        "sit",
        "play",
        "spin",
        "tilt_head",
        "sleep",
        "investigate",
        "stop_and_look",
        "idle",
        "nod",
    ]
    configured = database.get_setting("allowed_actions", default)
    if isinstance(configured, list):
        try:
            return set(str(x).strip().lower() for x in configured if x)
        except Exception:
            return set(default)
    # fallback if stored as string
    if isinstance(configured, str):
        try:
            parsed = json.loads(configured)
            if isinstance(parsed, list):
                return set(str(x).strip().lower() for x in parsed if x)
        except Exception:
            # comma-separated
            parts = [p.strip().lower() for p in configured.split(",") if p.strip()]
            if parts:
                return set(parts)
    return set(default)


def enforce_word_limit(speech: str, max_words: int = 20) -> str:
    """Ensure speech response does not exceed the word count limit."""
    words = speech.strip().split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return speech.strip()


def normalize_action(action: str) -> str:
    action_text = action.strip().lower()
    allowed = _get_allowed_actions()
    if action_text in allowed:
        return action_text
    return "stop_and_look"


def parse_ollama_response(text: str) -> Dict[str, str]:
    """Parse JSON output from Ollama generation text, handling optional markdown framing."""
    def build_response(data: Dict[str, Any]) -> Dict[str, str]:
        speech = str(data.get("speech", "I can't see right now! Please tell me more.")).strip()
        action = normalize_action(str(data.get("action", "stop_and_look")))
        return {
            "speech": enforce_word_limit(speech, 20),
            "action": action
        }

    try:
        data = json.loads(text.strip())
        return build_response(data)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return build_response(data)
            except json.JSONDecodeError:
                pass

        clean_text = re.sub(r'["{}\n]', ' ', text).strip()
        return {
            "speech": enforce_word_limit(clean_text or "I can't see right now! Tell me more.", 20),
            "action": "stop_and_look"
        }


async def perceive_and_generate(
    prompt: Optional[str] = None,
    image_frames: Optional[List[str]] = None,
    memory_context: Optional[str] = None,
    sensor_status: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[list[dict]] = None,
    client: Optional[httpx.AsyncClient] = None,
    vision_available: bool = True,
    visible_people: Optional[List[str]] = None,
    visible_people_description: Optional[str] = None,
    face_detected: bool = False,
) -> Dict[str, str]:
    """
    Query local Ollama instance with perception data and return structured speech and action.
    Supports multiple image frames and group-aware prompts.
    """
    user_prompt_parts = []
    if memory_context:
        user_prompt_parts.append(f"Memory Context: {memory_context}")
    if sensor_status:
        user_prompt_parts.append(f"Sensor Status: {json.dumps(sensor_status)}")

    if vision_available:
        user_prompt_parts.append("You can see and hear the user now. Use a warm pet voice.")
    else:
        user_prompt_parts.append("You cannot see right now. My eyes are closed. You can hear the user, but you cannot see them.")

    if conversation_history:
        user_prompt_parts.append("Conversation History:")
        for turn in conversation_history:
            user_prompt_parts.append(
                f'User: "{turn["user_prompt"]}"\nGemi: "{turn["gemi_response"]}"'
            )

    # Always include the spoken prompt if present
    if prompt:
        user_prompt_parts.append(f'User Said: "{prompt}"')
    else:
        user_prompt_parts.append("User Said: (No spoken text)")

    # Identity context — determines how Gemi should address visible people
    if visible_people and len(visible_people) > 0:
        if len(visible_people) == 1 and visible_people[0] != "Unknown":
            user_prompt_parts.append(f"Identified Person: {visible_people[0]} (confirmed by face recognition). Greet them by name.")
        else:
            names_list = ", ".join([n for n in visible_people if n != "Unknown"]) if visible_people else ""
            if names_list:
                user_prompt_parts.append(f"Visible people: {names_list}. Greet them warmly and individually, e.g. 'Hi Jam! Hi Alice!'")
            if "Unknown" in visible_people:
                user_prompt_parts.append("There is at least one person I don't recognize; ask their name politely.")
    elif face_detected and vision_available:
        # A face was detected but doesn't match anyone in the database
        user_prompt_parts.append(
            "Unknown Person: A face was detected but not recognized in memory. "
            "Politely ask their name, e.g. 'Hi! I don't know you yet — what's your name?'"
        )

    # If sequential image frames were provided, instruct the model how to use them
    if image_frames:
        user_prompt_parts.append(
            "The user provided a sequence of keyframes captured while speaking. Use these sequential snapshots to understand what was shown or what changed over time."
        )

    # If available, add a short visible people description for context
    if visible_people_description:
        user_prompt_parts.append(f"Visible People Description: {visible_people_description}")

    full_user_prompt = "\n".join(user_prompt_parts)

    # Respect runtime-configured Ollama model and token budget
    model = database.get_setting("ollama_model", OLLAMA_MODEL)
    max_tokens = database.get_setting("ollama_max_tokens", 512)
    num_predict = database.get_setting("ollama_num_predict", 512)

    payload: Dict[str, Any] = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": full_user_prompt,
        "stream": False,
        "format": "json",
        # Increase token budget to avoid mid-sentence truncation
        "max_tokens": int(max_tokens),
        # Some Ollama frontends accept num_predict as an alternative
        "num_predict": int(num_predict),
    }

    # Attach image frames (if any) as a list of base64 strings expected by Ollama
    if image_frames:
        images_clean: List[str] = []
        for frm in image_frames:
            if not frm:
                continue
            if "," in frm:
                images_clean.append(frm.split(",", 1)[1])
            else:
                images_clean.append(frm)
        if images_clean:
            payload["images"] = images_clean

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=30.0, trust_env=False)
        close_client = True

    # Allow runtime override of Ollama URL
    ollama_url = database.get_setting("ollama_url", OLLAMA_URL)

    try:
        response = await client.post(ollama_url, json=payload)
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


