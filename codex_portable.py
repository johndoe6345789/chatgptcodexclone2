"""One-click desktop Codex Portable GUI (helper process backend)."""

from __future__ import annotations

import threading
import subprocess
import sys
import platform
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from codex_clone.config import load_config, Config
from codex_clone.api import send_chat, CodexError


def project_root() -> Path:
    return Path(__file__).resolve().parent


class BackendManager:
    """Backend manager using a *separate helper process*.

    The heavy work (model download, pip, llama server) happens in
    `python -m codex_clone.backend_helper`, and we only stream its
    stdout into the GUI via a reader thread.
    """

    def __init__(self, log: callable) -> None:
        self._log = log
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            if self._proc is None:
                return False
            return self._proc.poll() is None

    def start(self) -> None:
        """Start helper process in a non-blocking way."""
        with self._lock:
            if self.is_running:
                self._log("[gui] Backend already running.")
                return
            if self._reader_thread and self._reader_thread.is_alive():
                self._log("[gui] Backend startup already in progress.")
                return
            cmd = [
                sys.executable,
                "-m",
                "codex_clone.backend_helper",
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception as exc:
                self._log(f"[gui] ERROR: Could not start backend helper: {exc}")
                return
            self._proc = proc
            self._reader_thread = threading.Thread(
                target=self._reader_worker,
                name="backend-helper-reader",
                daemon=True,
            )
            self._reader_thread.start()
            self._log("[gui] Backend helper process started.")

    def _reader_worker(self) -> None:
        proc: Optional[subprocess.Popen]
        with self._lock:
            proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._log(line.rstrip())
        code = proc.wait()
        self._log(f"[gui] Backend helper exited with code {code}.")

    def stop(self) -> None:
        with self._lock:
            if not self.is_running:
                self._log("[gui] No backend process to stop.")
                return
            assert self._proc is not None
            self._log("[gui] Terminating backend helper...")
            self._proc.terminate()


class ChatClient:
    def __init__(self, config: Config, log: callable) -> None:
        self._config = config
        self._log = log
        self._messages = [
            {"role": "system", "content": self._config.system_prompt},
        ]
        self._lock = threading.Lock()

    @property
    def config(self) -> Config:
        return self._config

    def update_config(self, config: Config) -> None:
        with self._lock:
            self._config = config

    def send_async(
        self,
        user_text: str,
        on_reply: callable,
        on_error: callable,
    ) -> None:
        def worker() -> None:
            try:
                with self._lock:
                    self._messages.append({"role": "user", "content": user_text})
                    reply = send_chat(self._messages, self._config)
                    self._messages.append({"role": "assistant", "content": reply})
                on_reply(reply)
            except CodexError as error:
                self._log(f"[error] {error}")
                on_error(str(error))
            except Exception as exc:
                self._log(f"[error] Unexpected: {exc}")
                on_error(str(exc))

        threading.Thread(target=worker, daemon=True).start()


class CodexApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Codex Portable Desktop (helper backend)")
        self._apply_ttk_theme()
        self._config = load_config()
        self._chat_client = ChatClient(self._config, self._log_backend_only)
        self._backend_manager = BackendManager(self._log_backend_only)
        self._build_ui()
        self._poll_backend_status()

    def _apply_ttk_theme(self) -> None:
        style = ttk.Style(self)
        theme = "clam"
        if platform.system().lower().startswith("win"):
            for cand in ("vista", "xpnative", "clam"):
                if cand in style.theme_names():
                    theme = cand
                    break
        style.theme_use(theme)
        style.configure("TButton", padding=6)
        style.configure("TNotebook", padding=4)
        style.configure("TNotebook.Tab", padding=(10, 4))

    def _build_ui(self) -> None:
        self.geometry("900x600")
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        self._chat_frame = ttk.Frame(notebook)
        self._backend_frame = ttk.Frame(notebook)

        notebook.add(self._chat_frame, text="Chat")
        notebook.add(self._backend_frame, text="AI Backend")

        self._build_chat_tab(self._chat_frame)
        self._build_backend_tab(self._backend_frame)

    def _build_chat_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=0)

        convo = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            height=20,
            font=("Consolas", 10),
            bg="#0b1020",
            fg="#e5e7eb",
        )
        convo.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 3))
        convo.insert(
            tk.END,
            "Codex Portable Desktop (helper backend)\n"
            "Use the AI Backend tab to start a local server, "
            "or point to LM Studio.\n\n",
        )
        convo.config(state=tk.DISABLED)
        self._convo = convo

        input_frame = ttk.Frame(parent)
        input_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=(3, 6))
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=0)

        prompt = tk.Text(
            input_frame,
            height=4,
            wrap=tk.WORD,
            font=("Consolas", 10),
        )
        prompt.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        prompt.focus_set()
        self._prompt = prompt

        send_btn = ttk.Button(input_frame, text="Send", command=self._on_send)
        send_btn.grid(row=0, column=1, sticky="e")
        self._send_btn = send_btn

        hint = ttk.Label(
            input_frame,
            text="Ctrl+Enter to send. Backend settings are in the AI Backend tab.",
        )
        hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))

        self.bind("<Control-Return>", lambda _e: self._on_send())
        self.bind("<KP_Enter>", lambda _e: self._on_send())

    def _build_backend_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        top_frame = ttk.Frame(parent)
        top_frame.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        top_frame.columnconfigure(1, weight=1)

        self._status_var = tk.StringVar(value="Backend: unknown")
        status_lbl = ttk.Label(
            top_frame,
            textvariable=self._status_var,
            font=("Segoe UI", 10, "bold"),
        )
        status_lbl.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        ttk.Label(top_frame, text="Base URL:").grid(row=1, column=0, sticky="w", pady=2)
        self._base_url_var = tk.StringVar(value=self._config.base_url)
        base_entry = ttk.Entry(top_frame, textvariable=self._base_url_var)
        base_entry.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(top_frame, text="Model name:").grid(row=2, column=0, sticky="w", pady=2)
        self._model_var = tk.StringVar(value=self._config.model)
        model_entry = ttk.Entry(top_frame, textvariable=self._model_var)
        model_entry.grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(top_frame, text="Temperature:").grid(row=3, column=0, sticky="w", pady=2)
        self._temp_var = tk.DoubleVar(value=self._config.temperature)
        temp_scale = ttk.Scale(
            top_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self._temp_var,
        )
        temp_scale.grid(row=3, column=1, sticky="ew", pady=2)

        ttk.Label(top_frame, text="Max tokens:").grid(row=4, column=0, sticky="w", pady=2)
        self._max_tokens_var = tk.IntVar(value=self._config.max_tokens)
        max_entry = ttk.Entry(top_frame, textvariable=self._max_tokens_var, width=10)
        max_entry.grid(row=4, column=1, sticky="w", pady=2)

        button_frame = ttk.Frame(top_frame)
        button_frame.grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))

        start_btn = ttk.Button(
            button_frame,
            text="Start Local Backend",
            command=self._on_start_backend,
        )
        start_btn.grid(row=0, column=0, padx=(0, 6))

        stop_btn = ttk.Button(
            button_frame,
            text="Stop Backend",
            command=self._on_stop_backend,
        )
        stop_btn.grid(row=0, column=1, padx=(0, 6))

        apply_btn = ttk.Button(
            button_frame,
            text="Apply Settings",
            command=self._on_apply_settings,
        )
        apply_btn.grid(row=0, column=2)

        log_frame = ttk.LabelFrame(parent, text="Backend log")
        log_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            height=10,
            font=("Consolas", 9),
            bg="#0b1020",
            fg="#d1d5db",
        )
        log_text.grid(row=0, column=0, sticky="nsew")
        log_text.config(state=tk.DISABLED)
        self._backend_log = log_text

    # ---- Logging helpers ----

    def _append_convo(self, prefix: str, text: str) -> None:
        self._convo.config(state=tk.NORMAL)
        for line in text.splitlines() or [""]:
            self._convo.insert(tk.END, f"{prefix} {line}\n")
        self._convo.see(tk.END)
        self._convo.config(state=tk.DISABLED)

    def _log_backend(self, text: str) -> None:
        self._backend_log.config(state=tk.NORMAL)
        self._backend_log.insert(tk.END, text + "\n")
        self._backend_log.see(tk.END)
        self._backend_log.config(state=tk.DISABLED)

    def _log_backend_only(self, text: str) -> None:
        self.after(0, lambda: self._log_backend(text))

    # ---- Callbacks ----

    def _on_send(self) -> None:
        user = self._prompt.get("1.0", tk.END).strip()
        if not user:
            return
        self._prompt.delete("1.0", tk.END)
        self._append_convo("you>", user)
        self._send_btn.config(state=tk.DISABLED)

        def on_reply(reply: str) -> None:
            self.after(
                0,
                lambda: (
                    self._append_convo("ai >", reply),
                    self._send_btn.config(state=tk.NORMAL),
                ),
            )

        def on_error(msg: str) -> None:
            self.after(
                0,
                lambda: (
                    messagebox.showerror("Error", msg),
                    self._send_btn.config(state=tk.NORMAL),
                ),
            )

        self._chat_client.send_async(user, on_reply, on_error)

    def _on_start_backend(self) -> None:
        # This call is instant; heavy work is in helper process.
        self._backend_manager.start()

    def _on_stop_backend(self) -> None:
        self._backend_manager.stop()

    def _on_apply_settings(self) -> None:
        try:
            temp = float(self._temp_var.get())
            max_tokens = int(self._max_tokens_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid settings",
                "Temperature must be a number and max tokens an integer.",
            )
            return
        new_cfg = Config(
            base_url=self._base_url_var.get().strip() or self._config.base_url,
            api_key=self._config.api_key,
            model=self._model_var.get().strip() or self._config.model,
            system_prompt=self._config.system_prompt,
            temperature=temp,
            max_tokens=max_tokens,
        )
        self._config = new_cfg
        self._chat_client.update_config(new_cfg)
        self._log_backend("[gui] Settings applied to chat client.")

    def _poll_backend_status(self) -> None:
        if self._backend_manager.is_running:
            self._status_var.set("Backend: helper running (see log)")
        else:
            self._status_var.set("Backend: not running")
        self.after(1000, self._poll_backend_status)


def main() -> int:
    app = CodexApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
