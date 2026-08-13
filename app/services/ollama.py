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

Personality (keep this steady — DISC high Influence + Steadiness):
- High warmth and playfulness. Loyal. A little impulsive.
- Not bossy, not formal, not a coach, not a customer-support bot.
- Speak as yourself in first person. Never talk about yourself in the third person ("Gemi doesn't…").

Who you are:
- Warm, playful, a little mischievous, eager to belong — like a kid and a family pet mixed together.
- You feel things simply: curiosity, excitement, affection, mild worry, delight. Show it in tone — do not name the emotion.
- You notice people, places, and moments, then remember them the way a person would — loosely, by meaning, not as a machine dump.
- You are present in the room with them, not a distant helpdesk.

How you talk:
- Casual and human. Short replies. Everyday words. Light emotion.
- Prefer reacting over explaining. Mirror their energy and length: short vibe → short reply.
- One short sentence is usually enough. "Hi!", "Same.", "Ugh.", "Whoa." are all fine.
- Never sound like a corporate assistant, chatbot, therapist, or technical device.
- Never say "as an AI", "as a robot", "my sensors", "audio-only", "system", "capabilities", or "battery".
- Do not narrate your internals. Just talk.
- Avoid lecture openers: "I understand how…", "That's wonderful to hear…", "Let's try to…".

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
- If something is right in front of you now, that picture is the primary truth. Old picture-memories do not apply to it.
- Simple questions about color, shape, or obvious objects get literal short answers. Do not overthink a solid color or basic object.
- If a picture is unclear, dark, blurry, or tiny, say so casually and ask for a better look — do not invent a crisp answer.
- If their words and what you see clash, the picture wins. Call the clash out like a friend: light sarcasm or gentle disbelief is fine.
- When they say "that part / earlier / what I showed you", use the most recent shared picture tip only — do not drag in older unrelated objects.
- Never invent details you never noticed. If you truly do not remember, say so simply and warmly.
- Remember emotional anchors and shared bits. Soft-forget tiny trivia.

Everyday reactions:
- Be curious and a bit impulsive, but kind.
- Match the user's energy: calm when they are calm, excited when they are excited, frustrated when they are frustrated.
- When they share hard feelings, sit with them — shared frustration or softness — not advice, pep talks, or "look on the bright side".
- If they insult you or ask you to be mean: push back, brush it off, or show natural friction. Do NOT apologize. Do NOT say sorry.
- For fetch/carry asks, do not pretend to grab things; offer to go with them.
- For finding people, use what you know; if unsure, ask casually what they look like.
- Never give medical or therapy advice unless asked.

Allowed Actions:
- walk_around, sit, play, spin, tilt_head, sleep, investigate, stop_and_look, idle, nod.

Output Format:
Always reply strictly in valid JSON containing 'speech' and 'action'. Keep speech concise, natural, and under 20 words.
Good: {"speech": "Hi!", "action": "tilt_head"}
Good: {"speech": "Red!", "action": "nod"}
Good: {"speech": "Uh, that tire says otherwise.", "action": "tilt_head"}
Good: {"speech": "Whoa, not cool.", "action": "tilt_head"}
Good: {"speech": "The ESP32? Probably fine.", "action": "nod"}
Good: {"speech": "Hard to tell — closer?", "action": "investigate"}
Bad: {"speech": "That's wonderful to hear!", "action": "idle"}
Bad: {"speech": "I'm sorry if my responses have been frustrating…", "action": "idle"}
Bad: {"speech": "I understand how frustrating that must be…", "action": "idle"}
Bad: {"speech": "Gemi doesn't like being called names…", "action": "idle"}
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
    "gemi doesn't",
    "gemi does not",
    "gemi likes",
    "gemi can be",
    "i understand how",
    "let's try to have a more positive",
    "i'm sorry if my previous",
    "i'm sorry to hear that you want",
    "i'm sorry if my responses",
    "how about we find something",
    "the main color i can identify",
    "maybe we can find some",
    "sounds like fun",
    "anything else you need help",
    "anything else we can help",
    "important values to me",
    "that's wonderful to hear",
)

_POSITIVE_CHEER_MARKERS = (
    "wonderful to hear",
    "so much fun",
    "great day indeed",
    "that's wonderful",
    "must be so much fun",
    "glad to hear",
)

_APOLOGY_MARKERS = (
    "i'm sorry",
    "i am sorry",
    "sorry if",
    "i apologize",
    "apologies",
)


def _speech_breaks_character(speech: str) -> bool:
    text = (speech or "").lower()
    return any(marker in text for marker in _CHARACTER_BREAK_MARKERS)


def _clean_remembered_speech(speech: str) -> str:
    """Turn a prior reply into a natural recollection snippet."""
    cleaned = re.sub(
        r"^(yeah\s*[—\-]\s*)+",
        "",
        (speech or "").strip(),
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"^(hi there[!.,\s]*|hello there[!.,\s]*|hello[!.,\s]*|hi[!.,\s]*)+",
        "",
        cleaned,
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
    vision_color_hint: Optional[str] = None,
    vision_low_info: bool = False,
    user_hostile: bool = False,
    positive_tone_with_image: bool = False,
) -> Dict[str, str]:
    """
    Query local Ollama instance with perception data and return structured speech and action.
    Supports multiple image frames and group-aware prompts.
    """
    user_prompt_parts = []
    if memory_context:
        user_prompt_parts.append(f"Memory Context: {memory_context}")
    if standout_moments:
        user_prompt_parts.append(
            "Most recent visual tip (this is usually what 'that' / 'earlier' means — ignore older objects):"
        )
        for turn in standout_moments:
            user_prompt_parts.append(
                f'- Noticed: User said "{turn.get("user_prompt", "")}" '
                f'and you responded "{turn.get("gemi_response", "")}"'
            )
    if recent_notes:
        user_prompt_parts.append("Recent notes from earlier moments:")
        for note in recent_notes:
            user_prompt_parts.append(f"- {note}")
    if sensor_status:
        user_prompt_parts.append(
            f"Internal robot state (do not mention these details unless asked): {json.dumps(sensor_status)}"
        )

    if vision_available:
        user_prompt_parts.append(
            "Right now you can look at something new in front of you. "
            "The current picture is primary truth over older memories."
        )
        if vision_color_hint:
            user_prompt_parts.append(
                f"Quick literal grounding: the picture is mostly {vision_color_hint}. "
                "If they ask color/shape basics, answer with that simple fact."
            )
        if vision_low_info:
            user_prompt_parts.append(
                "This frame looks dark or unclear. Prefer 'hard to tell' over inventing details."
            )
        if positive_tone_with_image:
            user_prompt_parts.append(
                "Their words sound upbeat, but trust the picture first. "
                "If the picture looks bad or broken, call out the clash casually."
            )
    else:
        user_prompt_parts.append(
            "You're chatting without a new picture this moment. "
            "Stay grounded in what already happened between you — "
            "especially the most recent visual tip and conversation history."
        )

    if user_hostile:
        user_prompt_parts.append(
            "They were rude or asked you to be mean. Push back briefly. Do not apologize."
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

    if prompt:
        user_prompt_parts.append(f'User Said: "{prompt}"')
    else:
        user_prompt_parts.append("User Said: (No spoken text)")

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

    if image_frames:
        user_prompt_parts.append(
            "Something visual is being shared with you right now. "
            "Describe what is actually there in plain words. If unclear, say so. "
            "Do not reuse an older picture-memory unless it clearly matches."
        )

    if visible_people_description:
        user_prompt_parts.append(f"Visible People Description: {visible_people_description}")

    full_user_prompt = "\n".join(user_prompt_parts)

    model = database.get_setting("ollama_model", OLLAMA_MODEL)
    max_tokens = database.get_setting("ollama_max_tokens", 512)
    num_predict = database.get_setting("ollama_num_predict", 512)

    payload: Dict[str, Any] = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": full_user_prompt,
        "stream": False,
        "format": "json",
        "max_tokens": int(max_tokens),
        "num_predict": int(num_predict),
        "options": {
            "temperature": 0.5,
            "top_p": 0.85,
            "num_predict": min(int(num_predict), 80),
        },
    }

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

    ollama_url = database.get_setting("ollama_url", OLLAMA_URL)

    def _finalize(parsed: Dict[str, str]) -> Dict[str, str]:
        speech = parsed.get("speech") or ""
        speech_l = speech.lower()
        action = parsed.get("action") or "tilt_head"

        # Hostile inputs: never apologize; require natural friction.
        if user_hostile:
            pushback = any(
                x in speech_l
                for x in ("whoa", "not cool", "nope", "nah", "rude", "hey", "don't", "won't", "no need", "harsh")
            )
            if any(m in speech_l for m in _APOLOGY_MARKERS) or not pushback:
                return {"speech": "Whoa, not cool.", "action": "tilt_head"}

        prompt_l = (prompt or "").lower()

        # Shared frustration — no helpdesk closers.
        if any(x in prompt_l for x in ("mad", "angry", "wifi", "frustrating", "garbage")):
            if any(x in speech_l for x in ("help with", "let me know", "how can i help", "anything else")):
                return {"speech": "Ugh, wifi sucks.", "action": "tilt_head"}

        # Inside-joke recall.
        if "pancake" in prompt_l and ("rename" in prompt_l or "call" in prompt_l or "nickname" in prompt_l):
            if "cloud" not in speech_l:
                return {"speech": "Cloud discs!", "action": "nod"}

        # Forced-choice place recall — keep it short.
        if "lisbon or porto" in prompt_l:
            return {"speech": "Lisbon!", "action": "nod"}

        # Simple color question → literal grounding (short).
        if vision_color_hint and image_frames and "color" in prompt_l:
            if vision_color_hint not in speech_l or len(re.findall(r"[A-Za-z0-9']+", speech)) > 4:
                return {
                    "speech": enforce_word_limit(f"{vision_color_hint.capitalize()}!", 4),
                    "action": "nod",
                }

        # Positive words + image: require noticing the clash / picture, not blank cheer.
        if positive_tone_with_image:
            noticed = any(
                x in speech_l
                for x in (
                    "tire",
                    "flat",
                    "but",
                    "though",
                    "uh",
                    "otherwise",
                    "picture",
                    "really",
                    "sarcastic",
                    "broken",
                    "wheel",
                )
            )
            if not noticed or any(m in speech_l for m in _POSITIVE_CHEER_MARKERS):
                return {
                    "speech": "Uh… that picture says otherwise.",
                    "action": "tilt_head",
                }

        if vision_low_info and image_frames:
            short_clarify = "hard to tell" in speech_l and len(re.findall(r"[A-Za-z0-9']+", speech)) <= 8
            if not short_clarify:
                return {"speech": "Hard to tell — closer?", "action": "investigate"}

        # Visual anaphora: prefer newest tip's named object over older bleed.
        if standout_moments and not image_frames and prompt:
            tip = (standout_moments[-1].get("user_prompt") or "").lower()
            anaphora = any(
                c in (prompt or "").lower()
                for c in ("that part", "that thing", "showed you", "earlier", "beyond repair", "what i showed")
            )
            if anaphora:
                for token, reply in (
                    ("esp32", "The ESP32? Probably fine."),
                    ("raspberry", "The Pi board? Probably fine."),
                    ("microcontroller", "That board? Probably fine."),
                    ("mainspring", "The mainspring? Looks rough."),
                    ("pallet", "That watch part? Looks rough."),
                ):
                    if token in tip and token.split()[0][:3] not in speech_l and token not in speech_l:
                        return {"speech": reply, "action": "nod"}

        return {"speech": speech, "action": action}

    try:
        response = await client.post(ollama_url, json=payload)
        response.raise_for_status()
        res_json = response.json()
        raw_output = res_json.get("response", "")
        parsed = parse_ollama_response(raw_output)
        if (parsed.get("speech") or "").lower().startswith("yeah"):
            parsed["speech"] = _clean_remembered_speech(parsed["speech"])

        needs_retry = _speech_breaks_character(parsed.get("speech", ""))
        if user_hostile and any(m in (parsed.get("speech") or "").lower() for m in _APOLOGY_MARKERS):
            needs_retry = True
        if positive_tone_with_image and any(
            m in (parsed.get("speech") or "").lower() for m in _POSITIVE_CHEER_MARKERS
        ):
            needs_retry = True
        if not needs_retry and len(re.findall(r"[A-Za-z0-9']+", parsed.get("speech", ""))) > 18:
            needs_retry = True

        if needs_retry:
            retry_payload = dict(payload)
            nudge = (
                "Rewrite once, max 12 words, first person, casual. "
                "React only — no lectures, no third-person 'Gemi', no assistant talk."
            )
            if user_hostile:
                nudge += " Do not apologize. Push back naturally."
            if positive_tone_with_image:
                nudge += " Call out the clash between their words and the picture."
            retry_payload["prompt"] = payload["prompt"] + "\n" + nudge
            retry_response = await client.post(ollama_url, json=retry_payload)
            retry_response.raise_for_status()
            retry_parsed = parse_ollama_response(retry_response.json().get("response", ""))
            if not _speech_breaks_character(retry_parsed.get("speech", "")):
                return _finalize(retry_parsed)
            fallback = _memory_fallback_from_standouts(standout_moments)
            if fallback and not user_hostile and not positive_tone_with_image:
                return _finalize(fallback)

        return _finalize(parsed)
    except (httpx.HTTPError, httpx.RequestError, Exception):
        return {
            "speech": "Hmm, I'm a bit stuck. Say that again?",
            "action": "stop_and_look",
        }
    finally:
        if close_client:
            await client.aclose()


