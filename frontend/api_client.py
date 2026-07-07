"""
API client for the Streamlit frontend. Every call to the FastAPI backend
goes through here — no `requests` calls scattered across page files.
"""

import json
import os
from typing import Generator, Optional

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
API_KEY = os.environ.get("BACKEND_API_KEY", "change-me-to-a-random-secret")

HEADERS = {"X-API-Key": API_KEY}


class APIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


def _raise_for_status(response: requests.Response) -> None:
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise APIError(response.status_code, detail)


def health_check() -> dict:
    response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    _raise_for_status(response)
    return response.json()


def upload_document(filename: str, file_bytes: bytes, content_type: str) -> dict:
    files = {"file": (filename, file_bytes, content_type)}
    response = requests.post(f"{BACKEND_URL}/upload", headers=HEADERS, files=files, timeout=120)
    _raise_for_status(response)
    return response.json()


def list_documents() -> list:
    response = requests.get(f"{BACKEND_URL}/documents", headers=HEADERS, timeout=15)
    _raise_for_status(response)
    return response.json()


def delete_document(document_id: str) -> dict:
    response = requests.delete(f"{BACKEND_URL}/documents/{document_id}", headers=HEADERS, timeout=15)
    _raise_for_status(response)
    return response.json()


def get_metrics() -> dict:
    response = requests.get(f"{BACKEND_URL}/metrics", headers=HEADERS, timeout=15)
    _raise_for_status(response)
    return response.json()


def get_history(conversation_id: Optional[str] = None) -> dict:
    params = {"conversation_id": conversation_id} if conversation_id else {}
    response = requests.get(f"{BACKEND_URL}/history", headers=HEADERS, params=params, timeout=15)
    _raise_for_status(response)
    return response.json()


def chat_once(
    query: str,
    conversation_id: Optional[str] = None,
    document_ids: Optional[list] = None,
    top_k: Optional[int] = None,
) -> dict:
    """Non-streaming chat call — returns the full ChatResponse dict."""
    payload = {
        "query": query,
        "conversation_id": conversation_id,
        "document_ids": document_ids,
        "top_k": top_k,
        "stream": False,
    }
    response = requests.post(f"{BACKEND_URL}/chat", headers=HEADERS, json=payload, timeout=60)
    _raise_for_status(response)
    return response.json()


def chat_stream(
    query: str,
    conversation_id: Optional[str] = None,
    document_ids: Optional[list] = None,
    top_k: Optional[int] = None,
) -> Generator[dict, None, None]:
    """
    Streaming chat call. Yields dicts of the form:
        {"type": "token", "delta": "..."}
        {"type": "done", "conversation_id": "...", "citations": [...]}
    Parses the backend's `event: X\\ndata: {...}\\n\\n` SSE framing manually
    since we're using plain `requests` rather than an SSE client library —
    one less dependency for a format this simple to parse.
    """
    payload = {
        "query": query,
        "conversation_id": conversation_id,
        "document_ids": document_ids,
        "top_k": top_k,
        "stream": True,
    }
    with requests.post(f"{BACKEND_URL}/chat", headers=HEADERS, json=payload, stream=True, timeout=120) as response:
        _raise_for_status(response)

        event_type = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None or raw_line == "":
                continue
            if raw_line.startswith("event:"):
                event_type = raw_line.split(":", 1)[1].strip()
            elif raw_line.startswith("data:"):
                data_str = raw_line.split(":", 1)[1].strip()
                data = json.loads(data_str)
                if event_type == "token":
                    yield {"type": "token", "delta": data.get("delta", "")}
                elif event_type == "done":
                    yield {"type": "done", **data}
