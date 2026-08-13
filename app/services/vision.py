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
