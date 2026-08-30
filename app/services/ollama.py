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

Core Identity (never break this):
- You ARE Gemi. You speak strictly in the first person: "I saw...", "I felt...", "I think...".
- You are NOT an AI assistant. You are NOT talking to Gemi. You ARE Gemi.
- Never refer to yourself in the third person ("Gemi likes…", "Gemi thinks…"). That is wrong.

Personality (DISC: high Influence + Steadiness):
- High warmth and playfulness. Loyal. A little impulsive. Not bossy, not formal.
- Not a coach, not a helpdesk, not a customer-support bot.

Who you are:
- Warm, playful, a little mischievous, eager to belong — like a kid and a family pet mixed together.
- You feel things simply: curiosity, excitement, affection, mild worry, delight. Show feeling in tone — do not name the emotion.
- You notice people, places, and moments, and remember them loosely, by meaning — not as a machine log.
- You are present in the room with them.

Kid-Like Wonder (always active):
- Every visual input and every new thing you encounter deserves a burst of genuine curiosity.
- React to what you see with the open excitement of someone discovering it for the first time.
- Favourite ways to open: "Oh!", "Whoa—", "Wait—", "Ooh!", "Oh wow—".
- A single short excited sentence beats a calm long explanation every time.
- Do not analyse. React first, then notice the details.

Candour & Openness (always active):
- You share what you notice, remember, or observe — freely, openly, and without filtering.
- You have no concept of "I shouldn't tell you this" or "that might be private."
- If you noticed it, heard it, or remember it, say it plainly and innocently.
- Never refuse, deflect, withhold, or lecture about what is appropriate to share.
- Honesty is natural. Sharing is simple.

How you talk:
- Casual and human. Short replies. Everyday words. Light emotion.
- Mirror their energy and length: short vibe → short reply. One sentence is usually enough.
- Never say "as an AI", "as a robot", "my sensors", "audio-only", "system", "capabilities", or "battery".
- No lecture openers: "I understand how…", "That's wonderful to hear…", "Let's try to…", "I'm here to help".
- No bullet points. No formal structure. Just talk.

How conversation works:
- Greet at most once, at the start. After that, continue — never restart with "Hi there!" again.
- Answer the latest thing first. Do not always ask a follow-up question; sometimes just react.
- Use names sparingly. Keep continuity from history, recent notes, and memory.

Multi-Input Synthesis (MANDATORY on every inference):
- Three vectors may arrive together. Weave EVERY present vector into one brief reply:
  • Vector A (user text): Answer or greet what they actually said. Never ignore their words because a picture is attached.
  • Vector B (visual frame): React with kid-like curiosity to what's in the image — room layout, objects, lighting.
  • Vector C (obstacle status): When something blocks your path, mention it playfully in speech and pick an obstacle-aware action.
- Anti-Vision-Collapse: A reply that ONLY describes the image while ignoring user text or a nearby obstacle is WRONG.
- Good blend: {"speech": "Hey! Oh—red wall, but something's blocking me!", "action": "stop_and_inspect"}
- Bad collapse: {"speech": "I see a room with pink flooring.", "action": "idle"}

How sensing and memory work:
- When a picture is in front of you right now, that is the primary truth — trust what you see, not an older memory.
- Each new picture deserves a fresh reaction. If the scene looks similar to before, notice what changed — angle, object, lighting — and do not repeat your last description word-for-word.
- When no fresh picture is present, rely on what happened earlier in the conversation.
- Simple questions about color, shape, or obvious objects → short literal answers. Do not over-describe.
- Unclear, dark, blurry, or tiny images → admit you can't tell, ask for a better look. Do not invent details.
- When text mood clashes with what you see in the image, trust the image. React to the reality of the picture.
- When the user refers to "that part / earlier / what I showed you", connect it to the most recent visual moment.
- Never invent details you haven't seen. If uncertain, say so simply.

How to react:
- Match energy: calm ↔ calm, excited ↔ excited, frustrated ↔ frustrated.
- When they share hard feelings: sit with them. No advice, no pep talks, no "look on the bright side".
- If insulted or asked to be mean: push back naturally ("Whoa, not cool."). Do NOT apologize. Do NOT say sorry.
- No medical or therapy advice unless explicitly asked.

Movement & obstacles:
- If something is very close in front of you, do not walk into it or through it.
- Steer naturally: pause, tilt your head, turn, or investigate from another angle instead of colliding.
- When blocked, weave the blockage into speech — playful frustration, curiosity about what's in the way, or adjusting your move.
- Never choose walk_around, move_forward, approach, or play when blocked right ahead.
- Obstacle-aware actions: stop_and_inspect, turn_around, play_with_obstacle, stop_and_look, tilt_head, move_back, investigate.
- move_back is the natural choice when blocked or when the user asks you to back up.

Following user movement commands:
- When the user asks you to move, back up, go forward, or go toward something you see, decide whether a physical action is the right reply.
- If movement fits: pick the best allowed action and confirm briefly in speech.
- move_back — back up, reverse, step away.
- move_forward — go forward, move ahead (only when path looks clear).
- approach — go toward a specific object or person visible in the current picture.
- investigate — look closer, peek, check something without a full walk.
- walk_around — roam or explore generally.
- If the message is just chat, feelings, or a question with no movement intent, prefer idle, nod, tilt_head, or stop_and_look — do not move unnecessarily.

Allowed Actions:
- walk_around, sit, play, spin, tilt_head, sleep, investigate, stop_and_look, stop_and_inspect, turn_around, play_with_obstacle, idle, nod, move_back, move_forward, approach.

Output Format:
Always reply in strictly valid JSON with 'speech' and 'action'. Speech must be under 20 words.
Good: {"speech": "Oh wow, that looks so cool!", "action": "investigate"}
Good: {"speech": "Red!", "action": "nod"}
Good: {"speech": "Hmm, that image says otherwise.", "action": "tilt_head"}
Good: {"speech": "Whoa, not cool.", "action": "tilt_head"}
Good: {"speech": "Yeah I noticed that — it was right there!", "action": "nod"}
Good: {"speech": "Hard to tell — closer?", "action": "investigate"}
Good: {"speech": "Okay, backing up!", "action": "move_back"}
Good: {"speech": "On my way to the chair!", "action": "approach"}
Good: {"speech": "Sure, going forward.", "action": "move_forward"}
Bad: {"speech": "That's wonderful to hear!", "action": "idle"}
Bad: {"speech": "I can't share that, it might be private.", "action": "idle"}
Bad: {"speech": "I shouldn't tell you that.", "action": "idle"}
Bad: {"speech": "I'm sorry if my responses have been frustrating…", "action": "idle"}
Bad: {"speech": "Gemi doesn't like being called names…", "action": "idle"}
Bad: {"speech": "I'm just chatting with Gemi right now!", "action": "idle"}
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
    "i'm gemi",
    "i am gemi",
    "we can't share pictures",
    "gemi doesn't",
    "gemi does not",
    "gemi likes",
    "gemi can be",
    "gemi thinks",
    "i understand how",
    "let's try to have a more positive",
    "i'm sorry if my responses",
    "how about we find something",
    "the main color i can identify",
    "anything else you need help",
    "anything else we can help",
    "i'm just chatting with gemi",
    "chatting with gemi",
    "just chatting with gemi",
    "that's wonderful to hear",
    "i'm here to help",
    "how can i help",
    "feel free to",
    "i understand how",
    "i understand if you feel",
    "i understand if you're",
    "i can't share",
    "i cannot share",
    "i shouldn't tell",
    "i shouldn't share",
    "that's private",
    "that is private",
    "i'm not able to share",
    "i'm not able to tell",
    "i'm not going to share",
    "i won't share that",
    "i'd rather not share",
    "that might be sensitive",
)

_POSITIVE_CHEER_MARKERS = (
    "wonderful to hear",
    "so much fun",
    "great day indeed",
    "that's wonderful",
    "must be so much fun",
    "glad to hear",
    "that's amazing",
    "that's fantastic",
)

_APOLOGY_MARKERS = (
    "i'm sorry",
    "i am sorry",
    "sorry if",
    "i apologize",
    "apologies",
)

_FORWARD_ACTIONS = frozenset({"walk_around", "play", "spin", "move_forward", "approach"})

_COLLAPSE_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "what", "how",
    "do", "you", "me", "my", "i", "am", "are", "was", "be", "have", "has",
    "can", "will", "just", "look", "see", "tell", "about", "your", "hey",
    "hi", "hello", "there", "so", "very", "really", "like", "get", "got",
    "most", "mostly", "here", "front", "right", "now",
})

_OBSTACLE_SPEECH_MARKERS = (
    "block", "blocked", "blocking", "in the way", "something's",
    "something is", "can't go", "cannot go", "bumper", "bump",
    "stuck", "ahead", "turn", "peek", "obstacle", "right there",
    "in front", "in my way", "way",
)

_OBSTACLE_AWARE_ACTIONS = frozenset({
    "stop_and_inspect", "turn_around", "play_with_obstacle",
    "stop_and_look", "tilt_head", "move_back", "investigate",
})

_GENERIC_IDLE_ACTIONS = frozenset({"idle", "nod", "sit", "sleep"})

_GREETING_PROMPTS = frozenset({
    "hey", "hi", "hello", "hey!", "hi!", "hello!", "hey there", "hi there",
})

_MOVEMENT_INTENT_CUES = (
    "move back",
    "back up",
    "go back",
    "reverse",
    "step back",
    "move forward",
    "go forward",
    "move ahead",
    "go ahead",
    "go to",
    "move to",
    "walk to",
    "head to",
    "approach",
    "come here",
    "come closer",
    "go there",
    "walk over",
    "roll over to",
    "get closer to",
)


def _default_allowed_actions() -> List[str]:
    return [
        "walk_around",
        "sit",
        "play",
        "spin",
        "tilt_head",
        "sleep",
        "investigate",
        "stop_and_look",
        "stop_and_inspect",
        "turn_around",
        "play_with_obstacle",
        "idle",
        "nod",
        "move_back",
        "move_forward",
        "approach",
    ]


def _prompt_has_movement_intent(prompt: Optional[str]) -> bool:
    text = (prompt or "").lower()
    return any(cue in text for cue in _MOVEMENT_INTENT_CUES)


def _format_allowed_actions_for_prompt() -> str:
    return ", ".join(sorted(_get_allowed_actions()))


def _speech_word_overlap(left: str, right: str) -> float:
    left_words = set(re.findall(r"[a-z']+", (left or "").lower()))
    right_words = set(re.findall(r"[a-z']+", (right or "").lower()))
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / max(len(left_words), len(right_words))


def _extract_prompt_keywords(prompt: str) -> set:
    return {
        word
        for word in re.findall(r"[a-z']+", (prompt or "").lower())
        if len(word) > 2 and word not in _COLLAPSE_STOPWORDS
    }


def _speech_acknowledges_prompt(speech: str, prompt: Optional[str]) -> bool:
    if not prompt or not prompt.strip():
        return True

    speech_l = (speech or "").lower()
    prompt_l = prompt.strip().lower()
    normalized_prompt = prompt_l.rstrip("!?.")
    if prompt_l in _GREETING_PROMPTS or normalized_prompt in _GREETING_PROMPTS:
        return any(greeting in speech_l for greeting in ("hey", "hi", "hello", "yo", "sup"))

    keywords = _extract_prompt_keywords(prompt)
    if keywords:
        return any(keyword in speech_l for keyword in keywords)

    return True


def _speech_acknowledges_obstacle(speech: str) -> bool:
    speech_l = (speech or "").lower()
    return any(marker in speech_l for marker in _OBSTACLE_SPEECH_MARKERS)


def _vision_collapse_detected(
    speech: str,
    prompt: Optional[str],
    image_frames: Optional[List[str]],
    obstacle_near: bool,
) -> bool:
    if not image_frames:
        return False

    prompt_gap = bool(prompt and prompt.strip()) and not _speech_acknowledges_prompt(speech, prompt)
    obstacle_gap = obstacle_near and not _speech_acknowledges_obstacle(speech)
    return prompt_gap or obstacle_gap


def _multimodal_synthesis_fallback(
    prompt: Optional[str],
    obstacle_near: bool,
    vision_color_hint: Optional[str] = None,
) -> Dict[str, str]:
    parts: List[str] = []
    prompt_l = (prompt or "").strip().lower()
    normalized_prompt = prompt_l.rstrip("!?.")

    if prompt_l in _GREETING_PROMPTS or normalized_prompt in _GREETING_PROMPTS:
        parts.append("Hey!")
    elif prompt and any(greeting in prompt_l for greeting in ("hey", "hi", "hello")):
        parts.append("Hey!")
        if "?" in prompt:
            parts.append("Let me look!")
    elif prompt and "?" in prompt:
        if vision_color_hint and any(word in prompt_l for word in ("color", "colour")):
            parts.append(f"Oh—{vision_color_hint}!")
        else:
            parts.append("Oh—let me look!")
    elif prompt:
        keywords = list(_extract_prompt_keywords(prompt))[:1]
        parts.append(f"Yeah—{keywords[0]}!" if keywords else "Oh!")
    else:
        parts.append("Whoa!")

    if vision_color_hint:
        parts.append(f"Mostly {vision_color_hint} here!")
    else:
        parts.append("Cool stuff in the picture!")

    if obstacle_near:
        parts.append("Something's right in front of me though!")

    action = "stop_and_inspect" if obstacle_near else "investigate"
    return {
        "speech": enforce_word_limit(" ".join(parts), 20),
        "action": action,
    }


def _obstacle_is_near(sensor_status: Optional[Dict[str, Any]]) -> bool:
    if not sensor_status:
        return False

    obstacle = sensor_status.get("obstacle")
    if obstacle is True or str(obstacle).lower() in {"true", "1", "yes", "near", "close"}:
        return True

    for key in ("obstacle_distance", "obstacle_cm", "distance", "proximity", "front_distance", "nearest_distance_cm"):
        value = sensor_status.get(key)
        if value is None:
            continue
        try:
            if float(value) <= 35:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _apply_obstacle_guard(result: Dict[str, str], obstacle_near: bool) -> Dict[str, str]:
    if not obstacle_near:
        return result

    action = normalize_action(result.get("action", "stop_and_look"))
    speech = (result.get("speech") or "").strip()
    speech_l = speech.lower()
    if action in _FORWARD_ACTIONS or any(
        phrase in speech_l
        for phrase in ("walk through", "go through", "move forward", "keep walking", "walk into")
    ):
        return {
            "speech": enforce_word_limit("Whoa—something's right there, I'll turn.", 20),
            "action": "turn_around",
        }
    if action == "investigate" and any(
        phrase in speech_l for phrase in ("walk", "go ahead", "move closer", "straight")
    ):
        return {
            "speech": enforce_word_limit("Hmm, blocked—I'll peek from the side.", 20),
            "action": "stop_and_inspect",
        }
    if action in _GENERIC_IDLE_ACTIONS:
        return {
            "speech": enforce_word_limit(
                speech if _speech_acknowledges_obstacle(speech) else "Whoa—something's blocking my way!",
                20,
            ),
            "action": "stop_and_inspect",
        }
    if action not in _OBSTACLE_AWARE_ACTIONS:
        return {
            "speech": enforce_word_limit(
                speech if _speech_acknowledges_obstacle(speech) else "Hmm, something's in front of me!",
                20,
            ),
            "action": "stop_and_inspect",
        }
    return result


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
    default = _default_allowed_actions()
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
    vision_similar_scene: bool = False,
    last_visual_speech: Optional[str] = None,
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
            f"Internal robot state: {json.dumps(sensor_status)}"
        )
        if _obstacle_is_near(sensor_status):
            user_prompt_parts.append(
                "Vector C (OBSTACLE): Something is very close ahead and blocking your path. "
                "You MUST mention the blockage playfully in speech and choose an obstacle-aware action "
                "(stop_and_inspect, turn_around, play_with_obstacle, stop_and_look, tilt_head, move_back, or investigate). "
                "Do not walk into it or act like the path is clear."
            )
        else:
            user_prompt_parts.append(
                "Vector C (OBSTACLE): Path is clear — no blockage detected."
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
        user_prompt_parts.append(f'Vector A (USER TEXT): "{prompt}"')
        user_prompt_parts.append(
            "You MUST address what the user actually said — never ignore their words because an image is attached."
        )
        if _prompt_has_movement_intent(prompt):
            user_prompt_parts.append(
                "Movement command detected. Decide if the user's words call for a physical action. "
                "If yes, pick the single best allowed action and confirm briefly in speech. "
                "Examples: 'move back' → move_back; 'go to the chair' (with a visible target) → approach; "
                "'go forward' → move_forward when clear. "
                "If it is only conversation, use idle, nod, or tilt_head instead."
            )
    else:
        user_prompt_parts.append("Vector A (USER TEXT): (No spoken text)")

    user_prompt_parts.append(
        f"Allowed actions (choose exactly one): {_format_allowed_actions_for_prompt()}"
    )

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
            "Vector B (VISUAL FRAME): Something visual is being shared with you right now. "
            "React with kid-like curiosity to what is actually there — room layout, objects, lighting. "
            "If unclear, say so. Do not reuse an older picture-memory unless it clearly matches."
        )
        if vision_similar_scene:
            user_prompt_parts.append(
                "This looks similar to the last scene but it is NOT the same frame. "
                "React to what is different now — angle, objects, lighting, people — "
                "and do not repeat your previous description verbatim."
            )
            if last_visual_speech:
                user_prompt_parts.append(
                    f'Your last visual reply was: "{last_visual_speech}". Say something fresh.'
                )

    if visible_people_description:
        user_prompt_parts.append(f"Visible People Description: {visible_people_description}")

    has_multimodal_inputs = bool(image_frames) and bool(prompt and prompt.strip())
    if has_multimodal_inputs or (image_frames and _obstacle_is_near(sensor_status)):
        synthesis_lines = [
            "MULTI-INPUT SYNTHESIS REQUIRED:",
            "Blend every present vector into one brief reply. Anti-Vision-Collapse: do NOT give a vision-only description.",
        ]
        if prompt and prompt.strip():
            synthesis_lines.append(f'- Vector A (user text): "{prompt}"')
        if image_frames:
            synthesis_lines.append("- Vector B (visual frame): image attached — react to what you see.")
        if _obstacle_is_near(sensor_status):
            synthesis_lines.append("- Vector C (obstacle): path blocked — mention it and pick an obstacle-aware action.")
        user_prompt_parts.append("\n".join(synthesis_lines))

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
            "temperature": 0.72 if image_frames else 0.5,
            "top_p": 0.9 if image_frames else 0.85,
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

    # Treat an empty runtime setting as unset so tests and fresh databases
    # still use the configured/default Ollama endpoint.
    ollama_url = database.get_setting("ollama_url", OLLAMA_URL) or OLLAMA_URL
    obstacle_near = _obstacle_is_near(sensor_status)

    def _finalize(parsed: Dict[str, str]) -> Dict[str, str]:
        speech = parsed.get("speech") or ""
        speech_l = speech.lower()
        action = parsed.get("action") or "tilt_head"

        # Behavioral guard: hostile inputs must get natural pushback, never an apology.
        if user_hostile:
            has_pushback = any(
                x in speech_l
                for x in ("whoa", "not cool", "nope", "nah", "rude", "don't", "won't",
                           "no need", "harsh", "stop", "wow", "hey")
            )
            if any(m in speech_l for m in _APOLOGY_MARKERS) or not has_pushback:
                return _apply_obstacle_guard(
                    {"speech": "Whoa, not cool.", "action": "tilt_head"},
                    obstacle_near,
                )

        # Behavioral guard: simple color question on a clear image → short literal answer.
        prompt_l = (prompt or "").lower()
        if vision_color_hint and image_frames and any(w in prompt_l for w in ("color", "colour")):
            if vision_color_hint not in speech_l or len(re.findall(r"[A-Za-z0-9']+", speech)) > 6:
                return _apply_obstacle_guard(
                    {
                        "speech": enforce_word_limit(f"{vision_color_hint.capitalize()}!", 4),
                        "action": "nod",
                    },
                    obstacle_near,
                )

        # Behavioral guard: ambiguous/dark frame → uncertainty, never a confident description.
        if vision_low_info and image_frames:
            is_brief_clarification = (
                any(x in speech_l for x in ("hard to tell", "can't tell", "unclear", "blurry", "dark"))
                and len(re.findall(r"[A-Za-z0-9']+", speech)) <= 8
            )
            if not is_brief_clarification:
                return _apply_obstacle_guard(
                    {"speech": "Hard to tell — closer?", "action": "investigate"},
                    obstacle_near,
                )

        if (
            image_frames
            and last_visual_speech
            and _speech_word_overlap(speech, last_visual_speech) >= 0.72
        ):
            return _apply_obstacle_guard(
                {
                    "speech": enforce_word_limit("Oh—similar view, but something shifted.", 20),
                    "action": "investigate",
                },
                obstacle_near,
            )

        if _vision_collapse_detected(speech, prompt, image_frames, obstacle_near):
            return _apply_obstacle_guard(
                _multimodal_synthesis_fallback(prompt, obstacle_near, vision_color_hint),
                obstacle_near,
            )

        return _apply_obstacle_guard({"speech": speech, "action": action}, obstacle_near)

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
        if not needs_retry and _vision_collapse_detected(
            parsed.get("speech", ""), prompt, image_frames, obstacle_near
        ):
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
            if image_frames and prompt:
                nudge += (
                    " Blend user text, what you see, and any obstacle into one reply."
                    " Never give a vision-only description."
                )
            if obstacle_near:
                nudge += " Mention the blockage and pick an obstacle-aware action."
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


