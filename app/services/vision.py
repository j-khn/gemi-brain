"""face_utils.py — Face encoding and matching utilities.

face_recognition (dlib-based) is imported lazily so the server starts even
when the native library is not compiled yet.  All public functions return
sensible fallback values when the library is unavailable.
"""

import base64
import binascii
import io
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import — dlib/face_recognition may not be installed until required.
# ---------------------------------------------------------------------------
try:
    import face_recognition as _face_recognition  # noqa: F401
    _FACE_RECOGNITION_AVAILABLE = True
except BaseException as exc:
    _face_recognition = None  # type: ignore
    _FACE_RECOGNITION_AVAILABLE = False
    logger.warning(
        "face_recognition module not available (%s). "
        "Face recognition features will be disabled. "
        "Run: brew install cmake libpng && pip install face_recognition",
        exc,
    )


def is_face_recognition_available() -> bool:
    """Return True if face_recognition is installed and importable."""
    return _FACE_RECOGNITION_AVAILABLE


def check_face_recognition_health() -> tuple[bool, str]:
    """Report face recognition status for readiness checks.

    Face recognition is required for visual mode and must be installed.
    """
    if _FACE_RECOGNITION_AVAILABLE:
        return True, "ok"
    return False, "missing face_recognition library"


def normalize_image_base64(image_str: str) -> str:
    """Validate and canonicalize a base64 image for both vision paths.

    Accepts raw base64 or a ``data:image/...;base64,`` value, tolerates
    transport whitespace, and returns canonical padded base64. Invalid input
    raises ``ValueError`` instead of being silently sent to Ollama.
    """
    if not isinstance(image_str, str) or not image_str.strip():
        raise ValueError("image_frame must contain a non-empty base64 string.")

    b64 = image_str.split(",", 1)[1] if "," in image_str else image_str
    b64 = "".join(b64.split())
    b64 += "=" * ((4 - len(b64) % 4) % 4)
    try:
        decoded = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_frame is not valid base64.") from exc

    if not decoded:
        raise ValueError("image_frame contains no image bytes.")
    return base64.b64encode(decoded).decode("ascii")


def decode_image_base64(image_str: str) -> bytes:
    """Decode a validated base64 image, stripping data-URL/whitespace."""
    return base64.b64decode(normalize_image_base64(image_str))


def extract_face_embedding(image_bytes: bytes) -> List[float]:
    """Extract a 128-d face embedding vector from raw image bytes.

    Returns a list of floats.
    Raises RuntimeError if face_recognition is not available.
    Raises ValueError if no face is detected in the image.
    """
    if not _FACE_RECOGNITION_AVAILABLE:
        raise RuntimeError(
            "face_recognition is not installed. Install dlib to enable face recognition."
        )
    image = _face_recognition.load_image_file(io.BytesIO(image_bytes))
    encodings = _face_recognition.face_encodings(image)
    if not encodings:
        raise ValueError("No face detected in the provided image.")
    return encodings[0].tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors (range –1 to 1)."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def match_face(
    query_embedding: List[float],
    stored: List[Tuple[int, str, List[List[float]]]],
) -> Optional[Tuple[int, str, float, float]]:
    """Find the best-matching person for the query embedding.

    Returns (person_id, name, cosine_sim, euclidean_dist) or None.
    """
    if not query_embedding or not stored:
        return None

    best_match: Optional[Tuple[int, str, float, float]] = None
    best_cosine = -1.0
    best_euclid = float("inf")

    for person_id, name, embeddings in stored:
        for emb in embeddings:
            cos = cosine_similarity(query_embedding, emb)
            euc = euclidean_distance(query_embedding, emb)
            if cos > best_cosine or (abs(cos - best_cosine) < 1e-6 and euc < best_euclid):
                best_cosine = cos
                best_euclid = euc
                best_match = (person_id, name, cos, euc)

    return best_match


def dominant_color_label(image_bytes: bytes) -> str:
    """Return a simple everyday color word for the image's average color."""
    try:
        from PIL import Image
    except ImportError:
        return ""

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))
    pixels = list(image.getdata())
    if not pixels:
        return ""
    r = sum(p[0] for p in pixels) / len(pixels)
    g = sum(p[1] for p in pixels) / len(pixels)
    b = sum(p[2] for p in pixels) / len(pixels)
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 40:
        return "black"
    if mn > 210:
        return "white"
    if mx - mn < 25:
        return "white" if mx > 160 else "gray"
    # Yellow: both R and G high, B clearly lower
    if r > 160 and g > 130 and b < 80 and min(r, g) > b * 2.5:
        return "yellow"
    # Orange: R high, G moderate, B low
    if r > 160 and 60 < g < 145 and b < 60:
        return "orange"
    # Purple: R and B both elevated, G lower
    if r > 110 and b > 110 and g < (r + b) * 0.45:
        return "purple"
    if r >= g and r >= b:
        return "red" if r > 90 else "brown"
    if g >= r and g >= b:
        return "green"
    return "blue"


def image_fingerprint(image_bytes: bytes, hash_size: int = 8) -> str:
    """Compact perceptual fingerprint for spotting similar (not identical) scenes."""
    try:
        from PIL import Image
    except ImportError:
        return ""

    gray = Image.open(io.BytesIO(image_bytes)).convert("L").resize((hash_size, hash_size))
    pixels = list(gray.getdata())
    if not pixels:
        return ""
    mean = sum(pixels) / len(pixels)
    return "".join("1" if value > mean else "0" for value in pixels)


def fingerprint_similarity(left: str, right: str) -> float:
    """Return fraction of matching bits between two fingerprints (0.0–1.0)."""
    if not left or not right or len(left) != len(right):
        return 0.0
    matches = sum(a == b for a, b in zip(left, right))
    return matches / len(left)


def looks_low_information(image_bytes: bytes) -> bool:
    """Heuristic: very dark frames are likely ambiguous to a person.

    Solid-colour bright images are NOT ambiguous — only truly dark/underexposed
    frames should trigger the uncertainty guard.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return False
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((32, 32))
    arr = np.asarray(image, dtype=np.float32)
    return float(arr.mean()) < 35
