from __future__ import annotations

import json
import socket
import threading
import subprocess
import sys
from typing import Callable, Optional

from .config import load_config
from .api import send_chat, CodexError
from .logging_utils import log_line


HOST = "127.0.0.1"
PORT = 56789


class BackendProcessManager:
    """Manage the llama backend helper process and stream its logs."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        with self._lock:
            if self._proc is None:
                return False
            return self._proc.poll() is None

    def start(self, log: Callable[[str], None]) -> None:
        with self._lock:
            if self.is_running():
                log("[daemon] Backend already running.")
                return
            cmd = [sys.executable, "-m", "codex_clone.backend_helper"]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception as exc:
                log(f"[daemon] ERROR: could not start backend_helper: {exc}")
                return
            self._proc = proc
        log("[daemon] backend_helper launched.")
        threading.Thread(
            target=self._log_reader, args=(proc, log), daemon=True
        ).start()

    def _log_reader(self, proc: subprocess.Popen, log: Callable[[str], None]) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            log(line.rstrip())
        code = proc.wait()
        log(f"[daemon] backend_helper exited with code {code}.")

    def stop(self, log: Callable[[str], None]) -> None:
        with self._lock:
            if not self.is_running():
                log("[daemon] No backend process to stop.")
                return
            assert self._proc is not None
            proc = self._proc
            self._proc = None
        log("[daemon] Terminating backend_helper...")
        proc.terminate()


class SocketBackendServer:
    """JSON-over-TCP daemon for the PyQt frontend."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._backend = BackendProcessManager()
        self._config = load_config()
        self._stop_event = threading.Event()
        self._listener: Optional[socket.socket] = None

    def serve_forever(self) -> None:
        log_line(f"[daemon] Socket backend starting on {self._host}:{self._port}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self._host, self._port))
            s.listen(5)
            self._listener = s
            log_line("[daemon] Socket backend listening")
            while not self._stop_event.is_set():
                try:
                    conn, addr = s.accept()
                except OSError:
                    break
                t = threading.Thread(
                    target=self._handle_client, args=(conn, addr), daemon=True
                )
                t.start()
        log_line("[daemon] Socket backend exiting serve_forever")

    def _request_shutdown(self, log: Callable[[str], None]) -> None:
        if not self._stop_event.is_set():
            log("[daemon] Shutdown requested, closing listener socket")
            self._stop_event.set()
            if self._listener is not None:
                try:
                    self._listener.close()
                except OSError:
                    pass

    def _handle_client(self, conn: socket.socket, addr) -> None:
        f_in = conn.makefile("r", encoding="utf-8")
        f_out = conn.makefile("w", encoding="utf-8")
        lock = threading.Lock()

        def send(obj: dict) -> None:
            text = json.dumps(obj, ensure_ascii=False)
            with lock:
                try:
                    f_out.write(text + "\n")
                    f_out.flush()
                except OSError:
                    return

        def log_fn(text: str) -> None:
            log_line(text)
            send({"type": "backend_log", "message": text})

        send({"type": "hello", "message": "socket backend ready"})
        try:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mtype = msg.get("type")
                if mtype == "ping":
                    send({"type": "pong"})
                elif mtype == "start_backend":
                    threading.Thread(
                        target=self._backend.start, args=(log_fn,), daemon=True
                    ).start()
                    send({"type": "start_backend_ack"})
                elif mtype == "stop_backend":
                    threading.Thread(
                        target=self._backend.stop, args=(log_fn,), daemon=True
                    ).start()
                    send({"type": "stop_backend_ack"})
                elif mtype == "status":
                    running = self._backend.is_running()
                    send({"type": "status", "running": running})
                elif mtype == "chat":
                    messages = msg.get("messages") or []
                    req_id = msg.get("id", "")

                    def chat_worker() -> None:
                        try:
                            reply = send_chat(messages, self._config)
                            send(
                                {
                                    "type": "chat_reply",
                                    "id": req_id,
                                    "ok": True,
                                    "content": reply,
                                }
                            )
                        except CodexError as exc:
                            log_line(f"[daemon] chat error: {exc}")
                            send(
                                {
                                    "type": "chat_reply",
                                    "id": req_id,
                                    "ok": False,
                                    "error": str(exc),
                                }
                            )

                    threading.Thread(target=chat_worker, daemon=True).start()
                elif mtype == "shutdown":
                    log_fn("[daemon] Shutdown message received from client")
                    self._backend.stop(log_fn)
                    send({"type": "shutdown_ack"})
                    self._request_shutdown(log_fn)
                    break
        finally:
            try:
                conn.close()
            except Exception:
                pass


def main() -> int:
    server = SocketBackendServer(HOST, PORT)
    server.serve_forever()
    log_line("[daemon] main() returning, process exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
