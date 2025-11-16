"""Plain terminal REPL (fallback / debugging)."""

from __future__ import annotations

from typing import Dict, List

import sys

from .config import load_config, Config
from .api import send_chat, CodexError


def _print_banner() -> None:
    """Show a short greeting message."""
    text = (
        "Local Codex clone - chat-only, code-focused assistant\n"
        "Type 'exit' or 'quit' to leave. Use empty line to send."
    )
    print(text)


def _build_initial_messages(config: Config) -> List[Dict[str, str]]:
    """Create the initial system message list."""
    return [
        {"role": "system", "content": config.system_prompt},
    ]


def _read_user_block() -> str:
    """Read a possibly multi-line user message from stdin."""
    lines: List[str] = []
    while True:
        try:
            line = input("you> ")
        except EOFError:
            return ""
        if line.strip().lower() in {"exit", "quit"}:
            return "__EXIT__"
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _print_assistant_reply(text: str) -> None:
    """Print the assistant reply with a small prefix."""
    separator = "-" * 60
    print(separator)
    print("codex>")
    print(text)
    print(separator)


def _handle_error(error: CodexError) -> None:
    """Print a human-friendly error message."""
    print(f"[error] {error}", file=sys.stderr)


def run_repl() -> None:
    """Run the interactive chat loop."""
    config = load_config()
    messages = _build_initial_messages(config)
    _print_banner()
    while True:
        user = _read_user_block()
        if user == "__EXIT__" or user == "":
            break
        messages.append({"role": "user", "content": user})
        try:
            reply = send_chat(messages, config)
        except CodexError as error:
            _handle_error(error)
            continue
        messages.append({"role": "assistant", "content": reply})
        _print_assistant_reply(reply)


if __name__ == "__main__":
    run_repl()
