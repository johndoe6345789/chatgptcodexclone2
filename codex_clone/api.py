from __future__ import annotations

from typing import Iterable, List, Dict

import json
import urllib.request
import urllib.error

from .config import Config


class CodexError(RuntimeError):
    """Error raised when the HTTP API fails."""


def _build_payload(
    messages: Iterable[Dict[str, str]],
    config: Config,
) -> bytes:
    payload = {
        "model": config.model,
        "messages": list(messages),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    text = json.dumps(payload)
    return text.encode("utf-8")


def _build_request(
    payload: bytes,
    config: Config,
) -> urllib.request.Request:
    url = config.base_url.rstrip("/") + "/v1/chat/completions"
    request = urllib.request.Request(url, data=payload)
    request.add_header("Content-Type", "application/json")
    if config.api_key:
        request.add_header("Authorization", f"Bearer {config.api_key}")
    return request


def _parse_response(data: bytes) -> str:
    try:
        obj = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise CodexError("Invalid JSON from server") from exc
    choices = obj.get("choices") or []
    if not choices:
        raise CodexError("Response contains no choices")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if not isinstance(content, str):
        raise CodexError("Assistant content is not a string")
    return content


def send_chat(
    messages: List[Dict[str, str]],
    config: Config,
) -> str:
    payload = _build_payload(messages, config)
    request = _build_request(payload, config)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read()
    except urllib.error.URLError as exc:
        raise CodexError(str(exc)) from exc
    return _parse_response(body)
