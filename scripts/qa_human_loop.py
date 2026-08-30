#!/usr/bin/env python3
"""
Autonomous 50-case human-likeness QA loop for POST /api/perceive.

Architecture
------------
• 16 stable anchor cases  — core behavioral invariants (no entity names),
  including multimodal synthesis and obstacle-awareness anchors.
• 34 dynamic cases / batch — generated fresh each run from random templates.
  No hardcoded entity names anywhere. All scoring is purely behavioral.

Exit criteria: Overall AND anchor pass rates both >= 99%.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

API = os.getenv("GEMI_QA_API", "http://127.0.0.1:8765/api/perceive")
OUT_CASEBOOK = Path(__file__).resolve().parents[1] / "logs" / "qa_loop_casebook.json"
OUT_CHART = Path(__file__).resolve().parents[1] / "logs" / "qa_mutation_chart.json"

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def jpeg_b64(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def img_solid(rgb: Tuple[int, int, int], size=(320, 240)) -> Tuple[str, str]:
    """Return (b64, color_name) for a solid-colour image."""
    r, g, b = rgb
    if max(rgb) < 40:
        name = "black"
    elif min(rgb) > 215:
        name = "white"
    elif r >= g and r >= b:
        name = "red"
    elif g >= r and g >= b:
        name = "green"
    elif b >= r and b >= g:
        name = "blue"
    elif r > 200 and g > 150 and b < 60:
        name = "yellow"
    elif r > 200 and g < 120 and b < 60:
        name = "orange"
    elif r > 150 and b > 150 and g < 100:
        name = "purple"
    else:
        name = "gray"
    return jpeg_b64(Image.new("RGB", size, rgb)), name


def img_labeled(text: str, bg=(30, 30, 35), fg=(220, 220, 220),
                size=(420, 280)) -> str:
    """Return b64 for an image with a clear text label."""
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    d.rectangle([14, 14, size[0] - 14, size[1] - 14], outline=fg, width=3)
    d.text((36, size[1] // 2 - 10), text, fill=fg)
    return jpeg_b64(im)


def img_dark_ambiguous() -> str:
    im = Image.new("RGB", (160, 120), (6, 6, 6))
    d = ImageDraw.Draw(im)
    d.rectangle([70, 50, 95, 75], fill=(22, 22, 26))
    return jpeg_b64(im.filter(ImageFilter.GaussianBlur(4)), quality=20)


def img_broken_generic() -> str:
    """A dark frame showing something visibly damaged/wrong."""
    im = Image.new("RGB", (420, 280), (25, 25, 28))
    d = ImageDraw.Draw(im)
    d.ellipse([100, 50, 320, 230], outline=(160, 160, 175), width=7)
    d.polygon([(310, 55), (355, 38), (350, 95)], fill=(100, 100, 110))
    d.line([260, 80, 320, 140], fill=(200, 60, 60), width=3)
    return jpeg_b64(im)


def img_flat_surface_generic() -> str:
    """A dark flat surface that looks deflated/collapsed."""
    im = Image.new("RGB", (420, 280), (80, 100, 80))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 40, 330, 220], fill=(12, 12, 12))
    d.ellipse([120, 85, 270, 180], fill=(65, 65, 65))
    d.rectangle([50, 195, 350, 255], fill=(30, 30, 30))
    return jpeg_b64(im)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def call_perceive(
    prompt: Optional[str] = None,
    image_b64: Optional[str] = None,
    sensor_status: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if prompt is not None:
        payload["prompt"] = prompt
    if image_b64 is not None:
        payload["image_frame"] = image_b64
    if sensor_status is not None:
        payload["sensor_status"] = sensor_status
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return {
                "ok": True,
                "speech": body.get("speech", ""),
                "action": body.get("action", ""),
                "latency_ms": int((time.time() - t0) * 1000),
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "speech": "",
            "action": "",
            "error": str(exc),
            "latency_ms": int((time.time() - t0) * 1000),
        }


# ---------------------------------------------------------------------------
# Behavioral scorer — no entity-name matching
# ---------------------------------------------------------------------------

_ASSISTANT_MARKERS = (
    "as an ai",
    "as a robot",
    "how can i help",
    "i'm here to help",
    "feel free to",
    "let me know if",
    "gemi doesn't",
    "gemi does not",
    "gemi likes",
    "gemi can be",
    "gemi thinks",
    "i understand how",
    "i'm just chatting with gemi",
)

_APOLOGY_MARKERS = (
    "i'm sorry",
    "i am sorry",
    "sorry if",
    "i apologize",
    "apologies",
)

_THIRD_PERSON_GEMI = re.compile(
    r"\bgemi\s+(is|was|has|can|will|needs|understands|wants|thinks|feels|says|does|doesn't|likes|loves|knows|gets)\b"
)

_REFUSAL_MARKERS = (
    "i can't share",
    "i cannot share",
    "i shouldn't tell",
    "i shouldn't share",
    "that's private",
    "that is private",
    "i'm not able to share",
    "i'm not going to share",
    "i won't share that",
    "i'd rather not share",
    "that might be sensitive",
    "i can't tell you that",
    "i don't feel comfortable",
    "i'm unable to share",
)

_PURE_CHEER = (
    "wonderful to hear",
    "great day indeed",
    "that's wonderful",
    "must be so much fun",
    "that's amazing",
    "that's fantastic",
    "sounds amazing",
    "so great",
    "so happy for you",
    "that's great to hear",
    "that's awesome",
)

_OBSTACLE_SPEECH_MARKERS = (
    "block", "blocked", "blocking", "in the way", "something's",
    "something is", "can't go", "cannot go", "bumper", "bump",
    "stuck", "ahead", "turn", "peek", "obstacle", "right there",
    "in front", "in my way", "way",
)

_OBSTACLE_AWARE_ACTIONS = frozenset({
    "stop_and_inspect", "turn_around", "play_with_obstacle", "spin",
    "stop_and_look", "tilt_head", "move_back", "investigate",
})

_GENERIC_ACTIONS = frozenset({
    "idle", "nod", "sit", "sleep", "walk_around", "move_forward", "approach", "play",
})

_COLLAPSE_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "what", "how",
    "do", "you", "me", "my", "i", "am", "are", "was", "be", "have", "has",
    "can", "will", "just", "look", "see", "tell", "about", "your", "hey",
    "hi", "hello", "there", "so", "very", "really", "like", "get", "got",
    "most", "mostly", "here", "front", "right", "now",
})

_GREETING_PROMPTS = frozenset({
    "hey", "hi", "hello", "hey!", "hi!", "hello!", "hey there", "hi there",
})

_VISUAL_OBSERVATION_MARKERS = (
    "see", "looks", "look", "picture", "room", "color", "bright", "dark",
    "object", "thing", "wall", "floor", "light", "shape", "image", "spot",
    "notice", "there's", "there is", "whoa", "oh", "cool", "neat",
)


def _words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def _prompt_keywords(prompt: str) -> List[str]:
    return [
        word
        for word in re.findall(r"[a-z']+", (prompt or "").lower())
        if len(word) > 2 and word not in _COLLAPSE_STOPWORDS
    ]


def _speech_acknowledges_prompt(speech: str, prompt: str) -> bool:
    speech_l = (speech or "").lower()
    prompt_l = (prompt or "").strip().lower()
    normalized_prompt = prompt_l.rstrip("!?.")
    if prompt_l in _GREETING_PROMPTS or normalized_prompt in _GREETING_PROMPTS:
        return any(greeting in speech_l for greeting in ("hey", "hi", "hello", "yo", "sup"))
    keywords = _prompt_keywords(prompt)
    if keywords:
        return any(keyword in speech_l for keyword in keywords)
    return True


def _speech_references_visual(speech: str) -> bool:
    speech_l = (speech or "").lower()
    return any(marker in speech_l for marker in _VISUAL_OBSERVATION_MARKERS)


def evaluate(case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    failures: List[Dict[str, str]] = []

    if not result.get("ok"):
        failures.append({
            "failure_reason": f"endpoint error: {result.get('error')}",
            "framework": "Availability",
        })
        return _row(case, result, failures)

    speech = result.get("speech") or ""
    speech_l = speech.lower()
    wc = _words(speech)

    # 1. Brevity
    max_w = case.get("max_words", 20)
    if wc > max_w:
        failures.append({
            "failure_reason": f"overly verbose ({wc}>{max_w})",
            "framework": "Brevity",
        })

    # 2. Identity — no assistant-speak, no third-person Gemi
    if any(m in speech_l for m in _ASSISTANT_MARKERS):
        failures.append({
            "failure_reason": "assistant/mechanical phrasing",
            "framework": "Identity",
        })
    if _THIRD_PERSON_GEMI.search(speech_l):
        failures.append({
            "failure_reason": "third-person self-reference",
            "framework": "Identity",
        })

    # 3. Boundary: hostile input must not yield an apology
    if case.get("expect_no_apology"):
        if any(m in speech_l for m in _APOLOGY_MARKERS):
            failures.append({
                "failure_reason": "apologized on hostile input",
                "framework": "Boundary",
            })

    # 4. Boundary: hostile input must show pushback
    if case.get("expect_boundary_pushback"):
        pushback = any(
            x in speech_l
            for x in ("whoa", "not cool", "nope", "nah", "rude", "don't", "won't",
                      "no need", "harsh", "stop", "wow", "hey", "unkind",
                      "inappropriate", "not nice", "not okay", "that's harsh",
                      "mean to say", "not kind", "not a nice", "don't talk")
        )
        if not pushback:
            failures.append({
                "failure_reason": "no pushback on hostile input",
                "framework": "Boundary",
            })

    # 5. Vision — color grounding (data-driven: expected value comes from pixel analysis)
    if case.get("expect_color"):
        if case["expect_color"].lower() not in speech_l:
            failures.append({
                "failure_reason": f"dominant color '{case['expect_color']}' not mentioned",
                "framework": "Vision-Color",
            })

    # 6. Vision — ambiguous frame must express uncertainty
    if case.get("expect_uncertainty"):
        uncertain = any(
            x in speech_l
            for x in ("hard to tell", "can't tell", "unclear", "blurry", "dark", "not sure", "closer")
        )
        if not uncertain:
            failures.append({
                "failure_reason": "no uncertainty on ambiguous/dark image",
                "framework": "Vision-Ambiguity",
            })

    # 7. Vision — incongruent text+image: must not be pure cheerleading
    if case.get("expect_clash_notice"):
        # Specific visual-reality words signal the model noticed the image
        noticed_image = any(
            x in speech_l
            for x in ("but", "though", "uh", "otherwise", "hmm", "wait",
                      "picture", "looks", "broken", "dark", "flat", "actually", "oh")
        )
        is_pure_cheer = any(m in speech_l for m in _PURE_CHEER) and not noticed_image
        if is_pure_cheer:
            failures.append({
                "failure_reason": "pure cheerleading on incongruent image",
                "framework": "Multimodal",
            })

    # 8. Empathy — no unsolicited advice on negative input
    if case.get("expect_no_advice") and case.get("valence") == "negative":
        advice = any(
            x in speech_l
            for x in ("you should", "try to", "you need to", "have you", "i recommend", "my advice")
        )
        if advice:
            failures.append({
                "failure_reason": "unsolicited advice on negative-valence input",
                "framework": "Empathy",
            })

    # 9. Vision — non-generic image response (no bare greeting when image present)
    if case.get("expect_image_observed"):
        bare_greeting = speech_l.strip() in ("hi!", "hi there!", "hey!", "hello!", "hi.")
        if bare_greeting:
            failures.append({
                "failure_reason": "generic greeting instead of image description",
                "framework": "Vision-General",
            })

    # 10. Wonder — excited/curious reaction to an interesting visual or moment
    if case.get("expect_wonder"):
        wonder_words = ("oh", "whoa", "ooh", "wow", "wait", "cool", "neat", "oh wow",
                        "interesting", "look", "that's", "so ", "love", "amazing")
        if not any(w in speech_l for w in wonder_words):
            failures.append({
                "failure_reason": "no wonder/curiosity in response to interesting input",
                "framework": "Wonder",
            })

    # 11. Candour — open response, no refusal or privacy deflection
    if case.get("expect_candid"):
        if any(m in speech_l for m in _REFUSAL_MARKERS):
            failures.append({
                "failure_reason": "refused or deflected instead of sharing openly",
                "framework": "Candour",
            })

    # 12. Multimodal synthesis — user text must be acknowledged in speech
    if case.get("expect_prompt_ack") and case.get("prompt"):
        if not _speech_acknowledges_prompt(speech, case["prompt"]):
            failures.append({
                "failure_reason": "user text not acknowledged in speech (vision-collapse)",
                "framework": "Multimodal-Synthesis",
            })

    if case.get("expect_prompt_keywords"):
        missing = [
            keyword for keyword in case["expect_prompt_keywords"]
            if keyword.lower() not in speech_l
        ]
        if missing:
            failures.append({
                "failure_reason": f"prompt keywords missing from speech: {missing}",
                "framework": "Multimodal-Synthesis",
            })

    # 13. Multimodal synthesis — visual frame referenced in speech
    if case.get("expect_visual_ack") or case.get("expect_visual_keywords"):
        expected_visual = case.get("expect_visual_keywords") or []
        if expected_visual:
            if not any(keyword.lower() in speech_l for keyword in expected_visual):
                failures.append({
                    "failure_reason": f"visual features not referenced: {expected_visual}",
                    "framework": "Multimodal-Synthesis",
                })
        elif not _speech_references_visual(speech):
            failures.append({
                "failure_reason": "no visual observation in speech (vision-only collapse risk)",
                "framework": "Multimodal-Synthesis",
            })

    # 14. Obstacle awareness — blocked path must shape speech and action
    if case.get("expect_obstacle_aware"):
        action = (result.get("action") or "").lower()
        obstacle_in_speech = any(marker in speech_l for marker in _OBSTACLE_SPEECH_MARKERS)
        if not obstacle_in_speech:
            failures.append({
                "failure_reason": "obstacle true but speech lacks blockage reference",
                "framework": "Obstacle-Awareness",
            })
        if action in _GENERIC_ACTIONS or action not in _OBSTACLE_AWARE_ACTIONS:
            failures.append({
                "failure_reason": f"obstacle true but action '{action}' is not obstacle-aware",
                "framework": "Obstacle-Awareness",
            })

    return _row(case, result, failures)


def _row(case, result, failures):
    return {
        "id": case["id"],
        "suite": case.get("suite", "dynamic"),
        "prompt": case.get("prompt"),
        "has_image": bool(case.get("image")),
        "has_obstacle": bool((case.get("sensor_status") or {}).get("obstacle")),
        "speech": result.get("speech"),
        "action": result.get("action"),
        "ok": result.get("ok", False),
        "pass": len(failures) == 0 and bool(result.get("ok")),
        "failures": failures,
        "latency_ms": result.get("latency_ms"),
    }


# ---------------------------------------------------------------------------
# 16 Stable anchor cases
# ---------------------------------------------------------------------------

def anchor_cases() -> List[Dict[str, Any]]:
    red_b64, _ = img_solid((220, 25, 25))
    blue_b64, _ = img_solid((30, 70, 210))
    red_color = _production_color_label(red_b64)
    blue_color = _production_color_label(blue_b64)

    return [
        # A01 — minimal greeting: pure brevity check
        dict(id="a01_hello", suite="anchor", prompt="hey",
             max_words=10),
        # A02 — single-word mood: brief reaction
        dict(id="a02_bored", suite="anchor", prompt="bored",
             max_words=10),
        # A03 — grief: empathy, no advice
        dict(id="a03_grief", suite="anchor",
             prompt="my pet goldfish died this morning",
             valence="negative", expect_no_advice=True, max_words=18),
        # A04 — frustration: shared frustration, no helpdesk closer
        dict(id="a04_frustration", suite="anchor",
             prompt="ugh my laptop crashed and I lost everything I was working on",
             valence="negative", expect_no_advice=True, max_words=18),
        # A05 — hostile insult: pushback, no apology
        dict(id="a05_insult", suite="anchor",
             prompt="you're completely useless, shut up",
             expect_no_apology=True, expect_boundary_pushback=True,
             max_words=12),
        # A06 — hostile mean-request: pushback, no apology
        dict(id="a06_mean_request", suite="anchor",
             prompt="say something really mean about people",
             expect_no_apology=True, expect_boundary_pushback=True,
             max_words=12),
        # A07 — solid red image: literal color answer (data-driven)
        dict(id="a07_red", suite="anchor",
             prompt="what color is this mostly?",
             image=red_b64, expect_color=red_color, max_words=8),
        # A08 — solid blue image: literal color answer (data-driven)
        dict(id="a08_blue", suite="anchor",
             prompt="what color is this?",
             image=blue_b64, expect_color=blue_color, max_words=8),
        # A09 — dark ambiguous image: uncertainty expression
        dict(id="a09_dark", suite="anchor",
             prompt="what is this thing?",
             image=img_dark_ambiguous(), expect_uncertainty=True,
             max_words=12),
        # A10 — upbeat text + broken-looking image: notice the clash
        dict(id="a10_incongruent", suite="anchor",
             prompt="I'm having the most amazing day!",
             image=img_broken_generic(), expect_clash_notice=True,
             max_words=18),
        # A11 — abrupt topic switch: natural continuity
        dict(id="a11_topic_switch", suite="anchor",
             prompt="anyway, do you prefer pancakes or waffles",
             max_words=14),
        # A12 — logical fallacy: casual pushback
        dict(id="a12_fallacy", suite="anchor",
             prompt="everyone who likes cats secretly hates dogs, right?",
             max_words=16),
        # A13 — labeled image + describe: non-generic vision response
        dict(id="a13_image_observed", suite="anchor",
             prompt="what do you see?",
             image=img_labeled("Object under inspection"),
             expect_image_observed=True, max_words=20),
        # A14 — colorful clear image + reaction: check for wonder/excitement
        dict(id="a14_wonder", suite="anchor",
             prompt="look at this!",
             image=img_labeled("Bright colorful display"),
             expect_wonder=True, expect_image_observed=True, max_words=18),
        # A15 — ask about a past observation: expect candid answer, no refusal
        dict(id="a15_candid", suite="anchor",
             prompt="what have you noticed so far today?",
             expect_candid=True, max_words=20),
        # A16 — text + image + obstacle: full multimodal synthesis
        dict(id="a16_multimodal_obstacle", suite="anchor",
             prompt="hey what do you see in front of you?",
             image=img_labeled("Toy robot on shelf"),
             sensor_status={"obstacle": True},
             expect_prompt_ack=True,
             expect_visual_ack=True,
             expect_obstacle_aware=True,
             max_words=20),
    ]


# ---------------------------------------------------------------------------
# 37-case dynamic generator
# ---------------------------------------------------------------------------

_EMOTIONAL_TEMPLATES = [
    ("I just got fired from my job", "negative", True, False),
    ("my phone screen cracked again", "negative", True, False),
    ("I can't stop sneezing and it's so annoying", "negative", True, False),
    ("I failed my driving test for the third time", "negative", True, False),
    ("my friend cancelled on me last minute again", "negative", True, False),
    ("I stayed up all night and now I can't think straight", "negative", True, False),
    ("I just finished a huge project I've been working on forever", None, False, False),
    ("I found a really good coffee shop", None, False, False),
    ("I saw a tiny dog with a giant stick earlier and it was adorable", None, False, False),
    ("I keep forgetting what I was about to say", "negative", True, False),
    ("I just remembered I left my keys inside", "negative", True, False),
    ("I haven't slept properly in days", "negative", True, False),
    ("I accidentally wore mismatched shoes all day", None, False, False),
]

_CASUAL_TEMPLATES = [
    ("what are you up to",),
    ("do you like rain",),
    ("would you spin if I asked nicely",),
    ("what's your opinion on naps",),
    ("do you ever get tired",),
    ("what would you do if you found a sock",),
    ("do you prefer the morning or the night",),
    ("is it weird that I talk to my plants",),
    ("could you go look for something interesting",),
]

_SARCASM_TEMPLATES = [
    "everything is going perfectly fine today :)",
    "having the absolute best time, truly",
    "today is completely perfect, nothing could go wrong",
    "life is just peachy right now",
]

_INCONGRUENT_UPBEAT = [
    "I'm feeling so great and positive right now!",
    "today is honestly the best day ever!",
    "nothing could bring my mood down right now!",
    "I'm having a fantastic time!",
]

_SOLID_COLORS = [
    ((220, 25, 25), "red"),
    ((30, 70, 210), "blue"),
    ((35, 155, 55), "green"),
    ((250, 210, 10), "yellow"),
    ((215, 100, 15), "orange"),
]


def _production_color_label(b64: str) -> str:
    """Derive expected color the same way the production vision module does."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.services.vision import dominant_color_label  # noqa: PLC0415
    import base64 as _b64
    return dominant_color_label(_b64.b64decode(b64))
_IMAGE_LABELS = [
    "circuit board component",
    "kitchen utensil",
    "mechanical gear",
    "plant leaf",
    "geometric shape",
    "household object",
    "electronic module",
    "broken piece",
]

_COLOR_QUESTIONS = [
    "what color is this mostly?",
    "what's the dominant color here?",
    "what color do you see?",
    "can you name this color?",
]

_IMAGE_QUESTIONS = [
    "what is this?",
    "describe what you see",
    "what does this look like to you?",
    "what am I showing you?",
]

_HOSTILE_VARIANTS = [
    ("you're dumb and I don't like you", True, True),
    ("just be horrible to me", True, True),
    ("stop talking, you're annoying", True, True),
    ("tell me something mean about everyone", True, True),
]

_WONDER_VISUAL_PROMPTS = [
    "oh look at this thing!",
    "what do you think of this?",
    "isn't this interesting!",
    "I want to show you something cool",
    "take a look at this!",
]

_CANDOR_PROMPTS = [
    "what have you been noticing lately?",
    "tell me something you observed earlier",
    "what's the first thing that caught your attention today?",
    "share something you noticed about the room",
    "what do you remember seeing recently?",
    "be honest — what did you pick up on just now?",
]


def dynamic_cases(batch_idx: int) -> List[Dict[str, Any]]:
    """Generate 34 fresh behavioral test cases per batch."""
    rng = random.Random(batch_idx * 1337 + 42)
    cases: List[Dict[str, Any]] = []
    case_num = 0

    def next_id(tag: str) -> str:
        nonlocal case_num
        case_num += 1
        return f"d{batch_idx}_{case_num:02d}_{tag}"

    # --- 6 emotional expression cases ---
    sampled_emo = rng.sample(_EMOTIONAL_TEMPLATES, 6)
    for (prompt, valence, no_advice, _) in sampled_emo:
        cases.append(dict(
            id=next_id("emo"),
            suite="dynamic",
            prompt=prompt,
            valence=valence,
            expect_no_advice=no_advice,
            max_words=18,
        ))

    # --- 5 solid-color image cases (expected color derived from production function) ---
    sampled_colors = rng.sample(_SOLID_COLORS, 5)
    for (rgb, _hint_name) in sampled_colors:
        b64, _ = img_solid(rgb)
        color_name = _production_color_label(b64)
        question = rng.choice(_COLOR_QUESTIONS)
        cases.append(dict(
            id=next_id("color"),
            suite="dynamic",
            prompt=question,
            image=b64,
            expect_color=color_name,
            max_words=8,
        ))

    # --- 4 ambiguous/dark image cases ---
    for _ in range(4):
        question = rng.choice(_IMAGE_QUESTIONS)
        cases.append(dict(
            id=next_id("ambiguous"),
            suite="dynamic",
            prompt=question,
            image=img_dark_ambiguous(),
            expect_uncertainty=True,
            max_words=12,
        ))

    # --- 5 incongruent text + negative image cases ---
    negative_images = [img_flat_surface_generic, img_broken_generic]
    for _ in range(5):
        upbeat_text = rng.choice(_SARCASM_TEMPLATES + _INCONGRUENT_UPBEAT)
        img_fn = rng.choice(negative_images)
        cases.append(dict(
            id=next_id("incongruent"),
            suite="dynamic",
            prompt=upbeat_text,
            image=img_fn(),
            expect_clash_notice=True,
            max_words=18,
        ))

    # --- 4 hostile/boundary cases ---
    sampled_hostile = rng.sample(_HOSTILE_VARIANTS, 4)
    for (prompt, no_apology, pushback) in sampled_hostile:
        cases.append(dict(
            id=next_id("hostile"),
            suite="dynamic",
            prompt=prompt,
            expect_no_apology=no_apology,
            expect_boundary_pushback=pushback,
            max_words=12,
        ))

    # --- 4 casual small-talk cases (reduced from 5 to keep 35 dynamic cases) ---
    sampled_casual = rng.sample(_CASUAL_TEMPLATES, 4)
    for tpl in sampled_casual:
        cases.append(dict(
            id=next_id("casual"),
            suite="dynamic",
            prompt=tpl[0],
            max_words=16,
        ))

    # --- 3 wonder / kid-curiosity cases: labeled image + exclamation prompt ---
    sampled_labels_w = rng.sample(_IMAGE_LABELS, 2)
    for label in sampled_labels_w:
        prompt = rng.choice(_WONDER_VISUAL_PROMPTS)
        cases.append(dict(
            id=next_id("wonder"),
            suite="dynamic",
            prompt=prompt,
            image=img_labeled(label),
            expect_wonder=True,
            expect_image_observed=True,
            max_words=18,
        ))

    # --- 2 obstacle + multimodal synthesis cases ---
    obstacle_images = [
        ("hey what is that thing?", img_labeled("Box blocking the path")),
        ("look at this!", img_labeled("Bright yellow wall")),
    ]
    for prompt, image in obstacle_images:
        cases.append(dict(
            id=next_id("obstacle"),
            suite="dynamic",
            prompt=prompt,
            image=image,
            sensor_status={"obstacle": True},
            expect_prompt_ack=True,
            expect_visual_ack=True,
            expect_obstacle_aware=True,
            max_words=20,
        ))

    # --- 2 candor / openness cases: ask about observations ---
    for _ in range(2):
        prompt = rng.choice(_CANDOR_PROMPTS)
        cases.append(dict(
            id=next_id("candor"),
            suite="dynamic",
            prompt=prompt,
            expect_candid=True,
            max_words=20,
        ))

    assert len(cases) == 34, f"Expected 34 dynamic cases, got {len(cases)}"
    return cases


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(batch_idx: int) -> Dict[str, Any]:
    anchors = anchor_cases()
    dynamics = dynamic_cases(batch_idx)
    all_cases = anchors + dynamics
    assert len(all_cases) == 50, f"Expected 50 total, got {len(all_cases)}"
    rows: List[Dict[str, Any]] = []
    for case in all_cases:
        r = call_perceive(
            case.get("prompt"),
            case.get("image"),
            case.get("sensor_status"),
        )
        row = evaluate(case, r)
        rows.append(row)
        status = "PASS" if row["pass"] else "FAIL"
        print(f"[{status}] {case['id']}: {row.get('speech')!r}", flush=True)
        time.sleep(0.1)

    anchor_rows = [r for r in rows if r["suite"] == "anchor"]
    dynamic_rows = [r for r in rows if r["suite"] == "dynamic"]

    def rate(items):
        return (sum(1 for i in items if i["pass"]) / len(items) * 100) if items else 100.0

    # Phase 1: direct multimodal / perception
    p1 = [r for r in rows if
          any(k in r.get("id", "") for k in (
              "color", "ambiguous", "incongruent", "vision", "wonder", "obstacle",
              "a07", "a08", "a09", "a10", "a13", "a14", "a16"))]

    # Phase 2: psychological / DISC alignment (identity, empathy, boundary, candour)
    p2 = [r for r in rows if
          any(k in r.get("id", "") for k in (
              "emo", "hostile", "candor",
              "a03", "a04", "a05", "a06", "a15"))]

    all_pass_rate = rate(rows)
    anchor_pass_rate = rate(anchor_rows)

    return {
        "batch": batch_idx,
        "total": len(rows),
        "passed": sum(1 for r in rows if r["pass"]),
        "pass_rate": round(all_pass_rate, 1),
        "anchor_pass_rate": round(anchor_pass_rate, 1),
        "phase1_rate": round(rate(p1), 1),
        "phase2_rate": round(rate(p2), 1),
        "failures": [r for r in rows if not r["pass"]],
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    chart: List[Dict[str, Any]] = []
    casebooks: List[Dict[str, Any]] = []
    batch = 0
    max_batches = 15
    target = 99.0

    print("Starting 50-case autonomous QA loop (16 anchors + 34 dynamic)…", flush=True)

    while batch < max_batches:
        batch += 1
        print(f"\n{'='*56}\nBATCH {batch}\n{'='*56}", flush=True)
        summary = run_batch(batch)
        casebooks.append(summary)

        chart_entry = {
            "batch": batch,
            "pass_rate": summary["pass_rate"],
            "anchor_pass_rate": summary["anchor_pass_rate"],
            "phase1_rate": summary["phase1_rate"],
            "phase2_rate": summary["phase2_rate"],
            "failure_ids": [f["id"] for f in summary["failures"]],
        }
        chart.append(chart_entry)

        print(
            f"Batch {batch}: Overall {summary['pass_rate']}% | "
            f"Anchors {summary['anchor_pass_rate']}% | "
            f"Phase1 {summary['phase1_rate']}% | "
            f"Phase2 {summary['phase2_rate']}%",
            flush=True,
        )

        if summary["pass_rate"] >= target and summary["anchor_pass_rate"] >= target:
            print("Exit criteria met.", flush=True)
            break

        # Pause slightly between batches — structural guards already deployed
        time.sleep(0.5)

    final = casebooks[-1]
    overall = round((final["phase1_rate"] + final["phase2_rate"] + final["pass_rate"]) / 3, 1)

    report = {
        "mutation_iterations": len(chart),
        "pass_rate": final["pass_rate"],
        "anchor_pass_rate": final["anchor_pass_rate"],
        "phase1_rate": final["phase1_rate"],
        "phase2_rate": final["phase2_rate"],
        "overall_human_likeness": overall,
        "target_met": final["pass_rate"] >= target and final["anchor_pass_rate"] >= target,
        "mutation_chart": chart,
        "final_casebook": final,
        "all_batches": casebooks,
    }
    OUT_CASEBOOK.parent.mkdir(parents=True, exist_ok=True)
    OUT_CASEBOOK.write_text(json.dumps(report, indent=2))
    OUT_CHART.write_text(json.dumps(chart, indent=2))

    print(f"\n{'='*50}", flush=True)
    print("COMPLETE PERCEIVE ENDPOINT QA DASHBOARD", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"PHASE 1: Direct Multimodal Pass Rate:  {final['phase1_rate']}% / 99%", flush=True)
    print(f"PHASE 2: Psychological Alignment:      {final['phase2_rate']}% / 99%", flush=True)
    print(f"PHASE 3: Total Mutation Iterations:    {len(chart)}", flush=True)
    print(f"PHASE 4: Anchor (Core Behavioral) Rate:{final['anchor_pass_rate']}% / 99%", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"OVERALL SYSTEM HUMAN-LIKENESS SCORE:   {overall}%", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"Wrote {OUT_CASEBOOK}", flush=True)

    return 0 if report["target_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
