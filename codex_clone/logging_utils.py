from __future__ import annotations

from pathlib import Path
import threading
import datetime


_log_lock = threading.Lock()
_LOG_PATH = Path(__file__).resolve().parent.parent / "codex.log"


def log_line(text: str) -> None:
    """Append a timestamped line to the shared log file."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {text}"
    with _log_lock:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
