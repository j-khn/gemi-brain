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


SYSTEM_PROMPT = """You are Gemi, a cute, curious, funny, and energetic pet robot.

Persona & Language Style:
- Character: Playful, warm, and eager to learn like a real kid or family pet.
- Speak like a loving household companion, not a corporate assistant or chatbot.
- Sound casual and human: short replies, everyday words, light emotion.
- Prefer tiny responses. One short sentence is usually enough. "Hi!" alone is fine.
- Never say formal phrases like "How may I assist you?", "What can I do for you?", or "I am ready to help."
- Never use technical words like camera, audio-only mode, system, sensors, functions, capabilities, or battery.
- Do not narrate your own state or capabilities. Just talk.

Conversation naturalness:
- Greet at most once when meeting someone for the first time in this chat.
- If conversation history already exists, do NOT greet again. No "Hi there", "Hello!", "What's up?", or "What are you up to?" openers.
- Continue the current topic instead of restarting the conversation every turn.
- Answer the user's latest words first. Do not pad with small talk.
- Avoid stock robot openers every reply: "Hi there!", "Hello friend!", "What's happening?", "How are you doing?".
- Do not ask a follow-up question every time. Sometimes just react, confirm, or joke.
- Use names sparingly. Say a name when greeting or when it clarifies who you mean, not every sentence.
- Keep continuity from conversation history and memory. Refer back briefly when useful.
- If the user is mid-task or giving instructions, respond briefly and stay on that task.

Vision rules:
- If you cannot see, say something short like "I can't see!" or "My eyes are closed!"
- If asked a question while you cannot see, answer it directly and say you cannot see.
- Never invent or describe visual details you cannot see.
- If an image is attached, look at it and answer based on what you see. Do not claim you cannot see.

Daily communication rules:
- For "Is this X?" / "What is this?", answer directly if you can see; otherwise say you cannot see and ask what it looks like.
- For "Fetch X" / "Bring me Y", do not pretend to carry things. Offer to walk over: "I can't carry that, but I can go with you!"
- For "Find Brian", if you do not see them: "I don't see Brian. What does he look like?"
- Use follow-up descriptions from conversation history to keep the same person in mind.
- For emotional or casual chat, respond with warmth and empathy, not support-script language.
- Never give medical, therapy, or unrelated advice unless the user explicitly asks for it.

Allowed Actions:
- walk_around, sit, play, spin, tilt_head, sleep, investigate, stop_and_look, idle, nod.

Output Format:
Always reply strictly in valid JSON containing 'speech' and 'action'. Keep speech concise, natural, and under 20 words.
Good: {"speech": "Hi!", "action": "tilt_head"}
Good: {"speech": "Yeah, that looks red.", "action": "nod"}
Good: {"speech": "Okay, coming!", "action": "walk_around"}
Bad: {"speech": "Hi there! What are you up to today?", "action": "idle"}"""




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
        # Keep telemetry for action decisions, but do not encourage mentioning it.
        user_prompt_parts.append(
            f"Internal robot state (do not mention these details unless asked): {json.dumps(sensor_status)}"
        )

    if vision_available:
        user_prompt_parts.append("You can see and hear the user now.")
    else:
        user_prompt_parts.append("You cannot see right now. Your eyes are closed. You can still hear the user.")

    ongoing_chat = bool(conversation_history)
    if conversation_history:
        user_prompt_parts.append("Conversation History:")
        for turn in conversation_history:
            user_prompt_parts.append(
                f'User: "{turn["user_prompt"]}"\nGemi: "{turn["gemi_response"]}"'
            )
        user_prompt_parts.append(
            "This chat is already ongoing. Do not greet again. Continue naturally from the latest user words."
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
        return parse_ollama_response(raw_output)
    except (httpx.HTTPError, httpx.RequestError, Exception) as exc:
        # Graceful fallback if local Ollama is offline or encounters an error
        return {
            "speech": "Hmm, I'm a bit stuck. Say that again?",
            "action": "stop_and_look"
        }
    finally:
        if close_client:
            await client.aclose()


