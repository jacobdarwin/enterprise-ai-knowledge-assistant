"""
Cleaning pass applied to raw extracted text before chunking.

Kept deliberately conservative — we normalize whitespace and strip junk
characters that come from PDF extraction (form-feed, null bytes, repeated
page-break artifacts) but we do NOT lowercase, strip stopwords, or stem.
This is a RAG pipeline, not a classic NLP/IR pipeline — the LLM needs the
original casing and structure to generate a good answer.
"""

import re
import unicodedata

# Runs of 3+ blank lines collapse to a single blank line
_MULTI_BLANK_LINE_RE = re.compile(r"\n\s*\n\s*\n+")
# Runs of horizontal whitespace collapse to one space
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
# Common PDF extraction artifacts
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Hyphenated line-wrap artifacts from PDFs: "informa-\ntion" -> "information"
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def clean_text(raw: str) -> str:
    if not raw:
        return ""

    text = unicodedata.normalize("NFKC", raw)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()
