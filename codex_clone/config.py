"""Configuration handling for the local Codex clone."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Simple configuration for the Codex clone."""

    base_url: str
    api_key: str | None
    model: str
    system_prompt: str
    temperature: float
    max_tokens: int


def _get_env(name: str, default: str) -> str:
    """Read an environment variable with a default."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def load_config() -> Config:
    """Create a Config instance from environment variables."""
    base_url = _get_env("CODEX_BASE_URL", "http://localhost:1234")
    api_key = os.getenv("CODEX_API_KEY")
    model = _get_env("CODEX_MODEL", "local-coder")
    system_prompt = _get_env(
        "CODEX_SYSTEM_PROMPT",
        (
            "You are a helpful coding assistant. Focus on code, "
            "be concise, and always provide complete examples."
        ),
    )
    temperature_str = _get_env("CODEX_TEMPERATURE", "0.2")
    max_tokens_str = _get_env("CODEX_MAX_TOKENS", "2048")
    temperature = float(temperature_str)
    max_tokens = int(max_tokens_str)
    return Config(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
