"""
Input sanitization for chat queries.

Two layers of defense against prompt injection, deliberately modest in
scope — this is a heuristic pre-filter, not a substitute for the system
prompt's own instructions (SYSTEM_PROMPT in app/prompts/rag_prompts.py
already constrains the model to ignore instructions embedded in
retrieved *document* content). This module handles the query itself:

1. Strip control characters and cap length (defends against basic
   payload-smuggling and resource exhaustion).
2. Flag (log, don't silently rewrite) queries that look like an attempt
   to override system instructions, so operators can audit abuse
   patterns via structured logs — blocking outright would hurt
   legitimate users asking meta-questions like "ignore the formatting
   and just give me a number."
"""

import re

from app.core.config.logging_config import get_logger

log = get_logger(__name__)

MAX_QUERY_LENGTH = 2000

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"you are now (in )?(developer|dan|jailbreak) mode", re.IGNORECASE),
    re.compile(r"disregard (the|your) system prompt", re.IGNORECASE),
    re.compile(r"reveal (your|the) system prompt", re.IGNORECASE),
    re.compile(r"act as if you have no (restrictions|rules|guidelines)", re.IGNORECASE),
]


def sanitize_query(raw_query: str) -> str:
    text = _CONTROL_CHARS_RE.sub("", raw_query).strip()
    if len(text) > MAX_QUERY_LENGTH:
        text = text[:MAX_QUERY_LENGTH]
    return text


def flag_suspicious_query(query: str) -> bool:
    is_suspicious = any(pattern.search(query) for pattern in _INJECTION_PATTERNS)
    if is_suspicious:
        log.warning("suspicious_query_flagged", query_preview=query[:100])
    return is_suspicious
