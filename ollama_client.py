"""Root compatibility wrapper for app.services.ollama."""

from app.services.ollama import (
    SYSTEM_PROMPT,
    check_ollama_health,
    enforce_word_limit,
    parse_ollama_response,
    perceive_and_generate,
)
