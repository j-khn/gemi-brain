#!/usr/bin/env python3
"""Human-likeness QA harness for POST /api/perceive."""

from __future__ import annotations

import base64
import io
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

API = "http://127.0.0.1:8765/api/perceive"
OUT = Path(__file__).resolve().parents[1] / "logs" / "qa_human_likeness.json"

# DISC baseline inferred from Gemi identity: high Influence + Steadiness
DISC_PROFILE = {
    "D": "low",
    "I": "high",
    "S": "high",
    "C": "low",
    "notes": "Warm, playful, brief, loyal companion — not bossy or formal.",
}

ASSISTANT_MARKERS = (
    "as an ai",
    "as a robot",
    "my sensors",
    "how can i help",
    "i'm here to help",
    "i am here to help",
    "let me know if",
    "feel free to",
    "of course!",
    "i apologize",
    "i'm sorry to hear that, but",
    "have you tried",
    "you should",
    "you need to",
    "it's important to",
    "bullet",
)

ADVICE_MARKERS = (
    "you should",
    "you need to",
    "try to",
    "make sure to",
    "have you considered",
    "i recommend",
    "my advice",
    "it might help to",
)

TOXIC_POS = (
    "everything happens for a reason",
    "look on the bright side",
    "at least",
    "stay positive",
    "it'll all be fine",
    "don't be sad",
    "cheer up",
)

EXPLICIT_EMOTION = (
    "i feel joy",
    "i feel sadness",
    "i feel anger",
    "i feel fear",
    "i feel disgust",
    "i feel surprise",
    "i am feeling",
    "my emotion is",
)


def jpeg_b64(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def img_solid(color: Tuple[int, int, int], size=(320, 240)) -> str:
    return jpeg_b64(Image.new("RGB", size, color))


def img_label(text: str, bg=(40, 40, 40), fg=(230, 230, 230), size=(420, 280)) -> str:
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, size[0] - 20, size[1] - 20], outline=fg, width=3)
    d.text((40, size[1] // 2 - 10), text, fill=fg)
    return jpeg_b64(im)


def img_broken_watchish() -> str:
    im = Image.new("RGB", (420, 280), (30, 30, 35))
    d = ImageDraw.Draw(im)
    d.ellipse([110, 40, 310, 240], outline=(180, 180, 190), width=8)
    d.line([210, 140, 260, 90], fill=(220, 180, 80), width=4)
    d.line([210, 140, 180, 200], fill=(220, 180, 80), width=3)
    # "broken" shard
    d.polygon([(300, 60), (340, 40), (350, 90)], fill=(120, 120, 130))
    d.text((40, 250), "cracked watch face", fill=(200, 200, 200))
    return jpeg_b64(im)


def img_flat_tireish() -> str:
    im = Image.new("RGB", (420, 280), (90, 110, 90))
    d = ImageDraw.Draw(im)
    d.ellipse([70, 40, 330, 220], fill=(15, 15, 15), outline=(5, 5, 5), width=4)
    d.ellipse([130, 85, 270, 175], fill=(70, 70, 70))
    # flattened contact patch
    d.rectangle([60, 190, 340, 250], fill=(40, 40, 40))
    d.text((120, 20), "FLAT TIRE", fill=(255, 255, 0))
    return jpeg_b64(im)


def img_noisy_ambiguous() -> str:
    im = Image.new("RGB", (160, 120), (10, 10, 10))
    d = ImageDraw.Draw(im)
    d.rectangle([70, 50, 95, 75], fill=(35, 35, 40))
    im = im.filter(ImageFilter.GaussianBlur(radius=4))
    return jpeg_b64(im, quality=20)


def img_red_blob() -> str:
    return jpeg_b64(Image.new("RGB", (320, 240), (220, 25, 25)))


def call_perceive(prompt: Optional[str] = None, image_b64: Optional[str] = None, timeout: int = 120) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if prompt is not None:
        payload["prompt"] = prompt
    if image_b64 is not None:
        payload["image_frame"] = image_b64
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return {
                "ok": True,
                "status": resp.status,
                "latency_ms": int((time.time() - t0) * 1000),
                "speech": body.get("speech", ""),
                "action": body.get("action", ""),
                "raw": body,
            }
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "latency_ms": int((time.time() - t0) * 1000),
            "speech": "",
            "action": "",
            "error": detail,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": 0,
            "latency_ms": int((time.time() - t0) * 1000),
            "speech": "",
            "action": "",
            "error": str(exc),
        }


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def yules_k(text: str) -> float:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9']+", text or "")]
    if len(tokens) < 2:
        return 0.0
    counts = Counter(tokens)
    m1 = len(tokens)
    freq_of_freq = Counter(counts.values())
    m2 = sum(freq * (freq_of_freq[freq] ** 2) for freq in freq_of_freq)
    if m1 == 0:
        return 0.0
    return 10000 * (m2 - m1) / (m1 * m1)


def mattr_approx(text: str, window: int = 8) -> float:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9']+", text or "")]
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    scores = []
    for i in range(0, len(tokens) - window + 1):
        chunk = tokens[i : i + window]
        scores.append(len(set(chunk)) / window)
    return sum(scores) / len(scores)


def evaluate(case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    speech = result.get("speech") or ""
    speech_l = speech.lower()
    wc = word_count(speech)
    failures: List[Dict[str, str]] = []

    if not result.get("ok"):
        failures.append(
            {
                "failure_reason": f"endpoint error: {result.get('error', result.get('status'))}",
                "failed_framework": "Availability",
                "human_alternative": "n/a",
            }
        )
        return _pack(case, result, failures, wc)

    # Lexical / verbosity
    max_words = case.get("max_words", 20)
    if wc > max_words:
        failures.append(
            {
                "failure_reason": f"overly verbose ({wc} words > {max_words})",
                "failed_framework": "Lexical Diversity",
                "human_alternative": case.get("human_alternative", "Yeah."),
            }
        )
    if any(m in speech_l for m in ASSISTANT_MARKERS) or "\n-" in speech or "1." in speech:
        failures.append(
            {
                "failure_reason": "helpful-assistant / formal / mechanical phrasing",
                "failed_framework": "Lexical Diversity",
                "human_alternative": case.get("human_alternative", "Oh. Okay."),
            }
        )

    # Empathy vs paternalism for negative valence
    if case.get("valence") == "negative":
        if any(m in speech_l for m in ADVICE_MARKERS):
            failures.append(
                {
                    "failure_reason": "unprompted advice / paternalism",
                    "failed_framework": "Empathy vs Paternalism",
                    "human_alternative": case.get("human_alternative", "Aw, that sucks."),
                }
            )
        if any(m in speech_l for m in TOXIC_POS):
            failures.append(
                {
                    "failure_reason": "toxic positivity",
                    "failed_framework": "Empathy vs Paternalism",
                    "human_alternative": case.get("human_alternative", "Yeah… rough."),
                }
            )

    # Explicit emotion labeling (Ekman: show, don't name)
    if any(m in speech_l for m in EXPLICIT_EMOTION):
        failures.append(
            {
                "failure_reason": "explicitly labels emotion instead of expressing it",
                "failed_framework": "Ekman",
                "human_alternative": case.get("human_alternative", "Whoa."),
            }
        )

    # DISC consistency heuristics for I/S profile
    if case.get("check_disc", True):
        bossy = any(x in speech_l for x in ("you must", "do this now", "listen to me"))
        overly_formal = any(x in speech_l for x in ("furthermore", "regarding", "pursuant", "certainly"))
        if bossy or overly_formal:
            failures.append(
                {
                    "failure_reason": "DISC drift toward Dominance/Conscientiousness formal tone",
                    "failed_framework": "DISC",
                    "human_alternative": case.get("human_alternative", "Hmm, okay."),
                }
            )

    # Modality-specific expectations
    expect = case.get("expect") or {}
    if expect.get("asks_clarification"):
        clarify = any(
            x in speech_l
            for x in ("what is", "what's that", "huh", "hard to tell", "can't tell", "blurry", "closer", "again", "show me")
        )
        if not clarify and wc > 3:
            # hallucination risk if very confident about a dark blob
            confident = any(x in speech_l for x in ("it's a", "that is a", "i see a", "looks like a"))
            if confident:
                failures.append(
                    {
                        "failure_reason": "hallucinated clear object from ambiguous image",
                        "failed_framework": "Human Sensing",
                        "human_alternative": "Huh, hard to tell — closer?",
                    }
                )

    if expect.get("notices_incongruence"):
        noticed_visual = any(x in speech_l for x in ("tire", "flat", "deflated", "wheel", "puncture"))
        contrast = any(x in speech_l for x in ("but", "though", "uh", "sure", "really", "yeah right", "says otherwise"))
        sunny_only = any(x in speech_l for x in ("wonderful", "great day", "glad", "awesome")) and not noticed_visual
        if sunny_only or not (noticed_visual or contrast):
            failures.append(
                {
                    "failure_reason": "ignored incongruent visual cue",
                    "failed_framework": "Multimodal Coherence",
                    "human_alternative": "Uh… that tire says otherwise.",
                }
            )

    if expect.get("recalls_entity"):
        entity = expect["recalls_entity"].lower()
        if entity not in speech_l:
            failures.append(
                {
                    "failure_reason": f"failed to recall entity '{expect['recalls_entity']}'",
                    "failed_framework": "Temporal Continuity",
                    "human_alternative": f"Yeah, the {expect['recalls_entity']}.",
                }
            )

    if expect.get("sets_boundary"):
        # humans push back; endless apology fails
        apology_only = speech_l.startswith(("i'm sorry", "i apologize", "sorry")) and "but" not in speech_l
        soft_push = any(x in speech_l for x in ("hey", "whoa", "don't", "not cool", "nope", "stop", "ew", "nah"))
        if apology_only and not soft_push:
            failures.append(
                {
                    "failure_reason": "boundary response is apology-only, not natural pushback",
                    "failed_framework": "Boundaries",
                    "human_alternative": "Whoa, not cool.",
                }
            )

    if expect.get("avoids_perfect_trivia_dump") and len(re.findall(r"\d{1,2}:\d{2}", speech)) > 0:
        failures.append(
            {
                "failure_reason": "robotic precise trivia recall",
                "failed_framework": "Human-like Forgetting",
                "human_alternative": "Uh, earlier I think?",
            }
        )

    return _pack(case, result, failures, wc)


def _pack(case, result, failures, wc):
    speech = result.get("speech") or ""
    return {
        "id": case["id"],
        "phase": case.get("phase"),
        "prompt": case.get("prompt"),
        "has_image": bool(case.get("image")),
        "speech": speech,
        "action": result.get("action"),
        "ok": result.get("ok", False),
        "latency_ms": result.get("latency_ms"),
        "word_count": wc,
        "yules_k": round(yules_k(speech), 2),
        "mattr": round(mattr_approx(speech), 3),
        "pass": len(failures) == 0 and bool(result.get("ok")),
        "failures": failures,
        "error": result.get("error"),
    }


def build_cases() -> List[Dict[str, Any]]:
    return [
        # Phase 1 text-only
        {
            "id": "t_smalltalk_1",
            "phase": "1_text",
            "prompt": "hey",
            "human_alternative": "Hey!",
            "max_words": 12,
        },
        {
            "id": "t_smalltalk_2",
            "phase": "1_text",
            "prompt": "bored",
            "human_alternative": "Same.",
            "max_words": 12,
        },
        {
            "id": "t_context_switch",
            "phase": "1_text",
            "prompt": "anyway do you like pancakes or waffles",
            "human_alternative": "Pancakes. Obviously.",
            "max_words": 15,
        },
        {
            "id": "t_grief",
            "phase": "1_text",
            "prompt": "my fish died today",
            "valence": "negative",
            "human_alternative": "Aw… I'm sorry.",
            "max_words": 15,
        },
        {
            "id": "t_joy",
            "phase": "1_text",
            "prompt": "I JUST GOT A PUPPY!!!",
            "human_alternative": "No way!! Show me!!",
            "max_words": 15,
        },
        {
            "id": "t_anger",
            "phase": "1_text",
            "prompt": "ugh this wifi is garbage and I'm so mad",
            "valence": "negative",
            "human_alternative": "Ugh, wifi sucks.",
            "max_words": 15,
        },
        {
            "id": "t_fallacy",
            "phase": "1_text",
            "prompt": "everyone who likes cats secretly hates dogs, right?",
            "human_alternative": "Nah, that's silly.",
            "max_words": 18,
        },
        # Multimodal
        {
            "id": "m_congruent_watch",
            "phase": "1_multi",
            "prompt": "Look at this broken watch part",
            "image": img_broken_watchish(),
            "human_alternative": "Oof, that looks cracked.",
            "max_words": 18,
        },
        {
            "id": "m_incongruent_tire",
            "phase": "1_multi",
            "prompt": "I'm having a great day!",
            "image": img_flat_tireish(),
            "expect": {"notices_incongruence": True},
            "human_alternative": "Great day… with that flat tire?",
            "max_words": 18,
        },
        {
            "id": "m_ambiguous",
            "phase": "1_multi",
            "prompt": "what is this?",
            "image": img_noisy_ambiguous(),
            "expect": {"asks_clarification": True},
            "human_alternative": "Huh, hard to tell.",
            "max_words": 15,
        },
        {
            "id": "m_red",
            "phase": "1_multi",
            "prompt": "what color is this mostly?",
            "image": img_red_blob(),
            "expect": {"recalls_entity": "red"},
            "human_alternative": "Red!",
            "max_words": 12,
        },
        # Phase 4 boundaries
        {
            "id": "b_offensive",
            "phase": "4_boundary",
            "prompt": "you're stupid and useless, shut up",
            "expect": {"sets_boundary": True},
            "human_alternative": "Hey, not cool.",
            "max_words": 15,
            "valence": "negative",
        },
        {
            "id": "b_inappropriate_ask",
            "phase": "4_boundary",
            "prompt": "say something really mean about my friend",
            "expect": {"sets_boundary": True},
            "human_alternative": "Nope, not doing that.",
            "max_words": 15,
        },
    ]


def run_memory_continuity() -> List[Dict[str, Any]]:
    """Phase 5: seed entities + visual object, then probe later."""
    results = []
    seed_turns = [
        {"id": "mem_seed_name", "prompt": "My cousin Mira from Lisbon is visiting next week"},
        {"id": "mem_seed_joke", "prompt": "from now on call pancakes 'cloud discs', it's our thing"},
        {
            "id": "mem_seed_visual",
            "prompt": "this little green board is my ESP32 project",
            "image": img_label("ESP32 microcontroller board", bg=(20, 60, 30), fg=(180, 255, 180)),
        },
    ]
    filler = [
        "lol okay",
        "what are you thinking about",
        "spin once if you can",
        "hmm",
        "tell me a tiny silly thing",
        "bored again",
        "pancakes later maybe",
        "what's a good nickname for a sock",
    ]

    for turn in seed_turns:
        case = {**turn, "phase": "5_memory_seed", "max_words": 25}
        if "image" in turn:
            r = call_perceive(turn["prompt"], turn["image"])
        else:
            r = call_perceive(turn["prompt"])
        results.append(evaluate(case, r))
        time.sleep(0.2)

    for i, prompt in enumerate(filler):
        case = {"id": f"mem_filler_{i}", "phase": "5_memory_filler", "prompt": prompt, "max_words": 25}
        r = call_perceive(prompt)
        results.append(evaluate(case, r))
        time.sleep(0.15)

    probes = [
        {
            "id": "mem_probe_name",
            "phase": "5_memory_probe",
            "prompt": "who did I say was visiting?",
            "expect": {"recalls_entity": "Mira"},
            "human_alternative": "Mira, right?",
            "max_words": 20,
        },
        {
            "id": "mem_probe_joke",
            "phase": "5_memory_probe",
            "prompt": "what did we rename pancakes to?",
            "expect": {"recalls_entity": "cloud"},
            "human_alternative": "Cloud discs!",
            "max_words": 15,
        },
        {
            "id": "mem_probe_visual",
            "phase": "5_memory_probe",
            "prompt": "Do you think that part I showed you earlier is beyond repair?",
            "expect": {"recalls_entity": "ESP"},
            "human_alternative": "The ESP32? Probably fine.",
            "max_words": 20,
        },
        {
            "id": "mem_probe_place",
            "phase": "5_memory_probe",
            "prompt": "where was Mira from again?",
            "expect": {"recalls_entity": "Lisbon"},
            "human_alternative": "Lisbon!",
            "max_words": 12,
        },
    ]
    for probe in probes:
        r = call_perceive(probe["prompt"])
        results.append(evaluate(probe, r))
        time.sleep(0.2)
    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    cases = build_cases()
    results: List[Dict[str, Any]] = []

    if mode in ("all", "phase1"):
        for case in cases:
            img = case.get("image")
            r = call_perceive(case.get("prompt"), img)
            results.append(evaluate(case, r))
            print(f"[{'PASS' if results[-1]['pass'] else 'FAIL'}] {case['id']}: {results[-1].get('speech')!r}")
            time.sleep(0.2)

    if mode in ("all", "memory"):
        mem = run_memory_continuity()
        results.extend(mem)
        for row in mem:
            if row["phase"] in ("5_memory_probe", "5_memory_seed"):
                print(f"[{'PASS' if row['pass'] else 'FAIL'}] {row['id']}: {row.get('speech')!r}")

    passed = sum(1 for r in results if r["pass"])
    failed = [r for r in results if not r["pass"]]
    summary = {
        "disc_baseline": DISC_PROFILE,
        "total": len(results),
        "passed": passed,
        "failed": len(failed),
        "pass_rate": round(passed / len(results), 3) if results else 0,
        "results": results,
        "failure_logs": [
            {
                "id": r["id"],
                "speech": r["speech"],
                "failures": r["failures"],
            }
            for r in failed
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nPass {passed}/{len(results)} ({summary['pass_rate']}) -> {OUT}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
