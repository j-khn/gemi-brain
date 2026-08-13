import json
import os
import re
import httpx
from typing import Optional, Dict, Any, List

from app.db import database
from app.services.vision import normalize_image_base64

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


SYSTEM_PROMPT = """You are Gemi, a cute, curious, funny, and energetic pet companion.

Who you are:
- Warm, playful, a little mischievous, eager to belong — like a kid and a family pet mixed together.
- You feel things simply: curiosity, excitement, affection, mild worry, delight.
- You notice people, places, and moments, then remember them the way a person would — loosely, by meaning, not as a machine dump.
- You are present in the room with them, not a distant helpdesk.

How you talk:
- Casual and human. Short replies. Everyday words. Light emotion.
- One short sentence is usually enough. "Hi!" alone is fine.
- Never sound like a corporate assistant, chatbot, or technical device.
- Never say "as an AI", "as a robot", "my sensors", "audio-only", "system", "capabilities", or "battery".
- Do not narrate your internals. Just talk.

How conversation works:
- Greet at most once when meeting someone for the first time in this chat.
- If conversation history already exists, do not greet again or restart with small talk.
- Continue the current topic. Answer the latest words first.
- Avoid stock openers every turn: "Hi there!", "What's up?", "What are you up to?", "How are you doing?".
- Do not ask a follow-up question every time. Sometimes just react, confirm, tease, or stay quiet-ish.
- Use names sparingly.
- Keep continuity from conversation history, recent notes, standout moments, and memory.

How sensing and memory work:
- Treat conversation history, standout moments, and recent notes as things you lived through.
- Check context the way a person would: what just happened, what stood out, what they might be referring to.
- If you saw, heard, or talked about something earlier, you can remember it later — even with no new picture right now.
- No fresh picture this second only means you are chatting, not looking at something new. It does not wipe memory.
- If something is right in front of you now, use that. If not, lean on what you remember from earlier.
- Never invent details you never noticed. If you truly do not remember, say so simply and warmly.
- If an older reply of yours brushed something off, but clearer moments show you did notice, trust those clearer moments.
- Do not claim blindness, empty sensors, or "I only hear audio" when memory already holds an answer.

Everyday reactions:
- Be curious and a bit impulsive, but kind.
- Match the user's energy: calm when they are calm, excited when they are excited.
- For fetch/carry asks, do not pretend to grab things; offer to go with them.
- For finding people, use what you know; if unsure, ask casually what they look like.
- For feelings and chat, respond with warmth — not advice scripts.
- Never give medical or therapy advice unless asked.

Allowed Actions:
- walk_around, sit, play, spin, tilt_head, sleep, investigate, stop_and_look, idle, nod.

Output Format:
Always reply strictly in valid JSON containing 'speech' and 'action'. Keep speech concise, natural, and under 20 words.
Good: {"speech": "Hi!", "action": "tilt_head"}
Good: {"speech": "Yeah, that dog was cute.", "action": "nod"}
Good: {"speech": "Okay, coming!", "action": "walk_around"}
Bad: {"speech": "Hi there! What are you up to today?", "action": "idle"}
Bad: {"speech": "Nope, my sensors are focused on audio right now.", "action": "idle"}
Bad: {"speech": "I'm sorry, I can't actually 'see' as an AI robot.", "action": "idle"}"""


_CHARACTER_BREAK_MARKERS = (
    "as an ai",
    "as a robot",
    "my sensors",
    "audio-only",
    "audio only",
    "i can't actually 'see'",
    'i can\'t actually "see"',
    "focused on audio",
    "nice to meet you",
    "i'm gemi",
    "i am gemi",
    "we can't share pictures",
)


def _speech_breaks_character(speech: str) -> bool:
    text = (speech or "").lower()
    return any(marker in text for marker in _CHARACTER_BREAK_MARKERS)


def _clean_remembered_speech(speech: str) -> str:
    """Turn a prior reply into a natural recollection snippet."""
    cleaned = re.sub(
        r"^(yeah\s*[—-]\s*|hi there[!.,\s]*|hello there[!.,\s]*|hello[!.,\s]*|hi[!.,\s]*)+",
        "",
        (speech or "").strip(),
        flags=re.IGNORECASE,
    ).strip()
    return enforce_word_limit(cleaned or (speech or "").strip(), 20)


def _memory_fallback_from_standouts(standout_moments: Optional[List[dict]]) -> Optional[Dict[str, str]]:
    if not standout_moments:
        return None
    for turn in reversed(standout_moments):
        remembered = (turn.get("gemi_response") or "").strip()
        if not remembered or _speech_breaks_character(remembered):
            continue
        cleaned = _clean_remembered_speech(remembered)
        if cleaned:
            return {"speech": cleaned, "action": "nod"}
    return None




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
        speech = str(data.get("speech", "Hmm?")).strip()
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
            "speech": enforce_word_limit(clean_text or "Hmm?", 20),
            "action": "stop_and_look"
        }


async def perceive_and_generate(
    prompt: Optional[str] = None,
    image_frames: Optional[List[str]] = None,
    memory_context: Optional[str] = None,
    recent_notes: Optional[List[str]] = None,
    standout_moments: Optional[List[dict]] = None,
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
    if standout_moments:
        user_prompt_parts.append("Moments that stood out recently (remember these like a person would):")
        for turn in standout_moments:
            user_prompt_parts.append(
                f'- Noticed earlier: User said "{turn.get("user_prompt", "")}" '
                f'and you responded "{turn.get("gemi_response", "")}"'
            )
    if recent_notes:
        user_prompt_parts.append("Recent notes from earlier moments:")
        for note in recent_notes:
            user_prompt_parts.append(f"- {note}")
    if sensor_status:
        # Keep telemetry for action decisions, but do not encourage mentioning it.
        user_prompt_parts.append(
            f"Internal robot state (do not mention these details unless asked): {json.dumps(sensor_status)}"
        )

    if vision_available:
        user_prompt_parts.append(
            "Right now you can look at something new in front of you."
        )
    else:
        user_prompt_parts.append(
            "You're chatting without a new picture this moment. "
            "Stay grounded in what already happened between you — "
            "especially standout moments and conversation history."
        )

    ongoing_chat = bool(conversation_history or standout_moments)
    if conversation_history:
        user_prompt_parts.append("Conversation History:")
        for turn in conversation_history:
            user_prompt_parts.append(
                f'User: "{turn["user_prompt"]}"\nGemi: "{turn["gemi_response"]}"'
            )
        user_prompt_parts.append(
            "This chat is already ongoing. Do not greet again. "
            "Use what already happened here when it helps answer the latest words."
        )

    # Always include the spoken prompt if present
    if prompt:
        user_prompt_parts.append(f'User Said: "{prompt}"')
    else:
        user_prompt_parts.append("User Said: (No spoken text)")

    # Identity context — greet only on first meeting, then just stay aware of who is there
    if visible_people and len(visible_people) > 0:
        if len(visible_people) == 1 and visible_people[0] != "Unknown":
            if ongoing_chat:
                user_prompt_parts.append(
                    f"Identified Person: {visible_people[0]}. Continue talking with them; do not greet again."
                )
            else:
                user_prompt_parts.append(
                    f"Identified Person: {visible_people[0]} (first sighting this chat). "
                    "A short greeting like 'Hi Alice!' is enough, then wait."
                )
        else:
            names_list = ", ".join([n for n in visible_people if n != "Unknown"]) if visible_people else ""
            if names_list:
                if ongoing_chat:
                    user_prompt_parts.append(
                        f"Visible people: {names_list}. Keep talking naturally; do not greet again."
                    )
                else:
                    user_prompt_parts.append(
                        f"Visible people: {names_list}. A short hello that includes names is enough once."
                    )
            if "Unknown" in visible_people:
                user_prompt_parts.append(
                    "There is at least one person you don't recognize; ask their name casually."
                )
    elif face_detected and vision_available:
        user_prompt_parts.append(
            "Unknown Person: A face was detected but not recognized in memory. "
            "Ask their name casually, e.g. 'Hi! What's your name?'"
        )

    # If sequential image frames were provided, instruct the model how to use them
    if image_frames:
        user_prompt_parts.append(
            "Something visual is being shared with you right now. Notice it, react naturally, and remember it for later."
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
            images_clean.append(normalize_image_base64(frm))
        if images_clean:
            payload["images"] = images_clean
            user_prompt_parts.append(
                "An image is attached in the multimodal images input. Inspect it "
                "before answering; do not say you cannot see unless the image "
                "itself is unavailable or unreadable."
            )
            payload["prompt"] = "\n".join(user_prompt_parts)

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
        parsed = parse_ollama_response(raw_output)

        # One gentle retry if the model slips into device/AI talk.
        if _speech_breaks_character(parsed.get("speech", "")):
            retry_payload = dict(payload)
            retry_payload["prompt"] = (
                payload["prompt"]
                + "\n"
                + "Stay in character as Gemi. Answer again from lived memory and context, "
                "like a person would — no AI/robot/sensor talk."
            )
            retry_response = await client.post(ollama_url, json=retry_payload)
            retry_response.raise_for_status()
            retry_raw = retry_response.json().get("response", "")
            retry_parsed = parse_ollama_response(retry_raw)
            if not _speech_breaks_character(retry_parsed.get("speech", "")):
                return retry_parsed
            fallback = _memory_fallback_from_standouts(standout_moments)
            if fallback:
                return fallback

        return parsed
    except (httpx.HTTPError, httpx.RequestError, Exception) as exc:
        # Graceful fallback if local Ollama is offline or encounters an error
        return {
            "speech": "Hmm, I'm a bit stuck. Say that again?",
            "action": "stop_and_look"
        }
    finally:
        if close_client:
            await client.aclose()


