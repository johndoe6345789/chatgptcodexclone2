"""One-click desktop Codex Portable GUI (non-blocking backend)."""

from __future__ import annotations

import threading
import subprocess
import sys
import platform
from pathlib import Path
from typing import Final, Callable, Optional

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from codex_clone.config import load_config, Config
from codex_clone.api import send_chat, CodexError


HF_REPO: Final[str] = "TheBloke/deepseek-coder-6.7B-instruct-GGUF"
HF_FILE: Final[str] = "deepseek-coder-6.7b-instruct.Q4_K_M.gguf"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def models_dir() -> Path:
    directory = project_root() / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_huggingface_hub(log: Callable[[str], None]) -> None:
    try:
        import huggingface_hub  # noqa: F401
        return
    except Exception:
        pass
    log("[info] Installing huggingface_hub...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "huggingface_hub>=0.25.0"],
        check=False,
    )


def download_model(log: Callable[[str], None]) -> Path:
    ensure_huggingface_hub(log)
    from huggingface_hub import hf_hub_download

    dest_dir = models_dir()
    local_path = dest_dir / HF_FILE
    if local_path.exists():
        log(f"[info] Model already present: {local_path}")
        return local_path
    log("[info] Downloading model from Hugging Face... (first run)")
    actual = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    local_path = Path(actual)
    log(f"[info] Model downloaded to: {local_path}")
    return local_path


def have_llama_server() -> bool:
    try:
        import llama_cpp.server  # type: ignore[unused-ignore]  # noqa: F401
        return True
    except Exception:
        return False


def ensure_llama_cpp(log: Callable[[str], None]) -> bool:
    if have_llama_server():
        return True
    log("[info] Installing llama-cpp-python[server] via pip...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "llama-cpp-python[server]"],
        check=False,
    )
    if result.returncode == 0 and have_llama_server():
        log("[info] llama-cpp-python[server] is ready.")
        return True
    log(
        "[warn] Could not install llama-cpp-python[server]. "
        "Use LM Studio or another backend if needed."
    )
    return False


class BackendManager:
    """Non-blocking backend manager.

    All slow work (download, install, server start, log reading)
    happens in background threads so the Tkinter UI never blocks.
    """

    def __init__(self, log: Callable[[str], None]) -> None:
        self._log = log
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._start_thread: Optional[threading.Thread] = None
        self._log_thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            if self._process is None:
                return False
            return self._process.poll() is None

    def start_in_background(self) -> None:
        with self._lock:
            if self.is_running:
                self._log("[info] Backend already running.")
                return
            if self._start_thread and self._start_thread.is_alive():
                self._log("[info] Backend start already in progress.")
                return
            self._start_thread = threading.Thread(
                target=self._start_worker,
                name="backend-start",
                daemon=True,
            )
            self._start_thread.start()

    def _start_worker(self) -> None:
        try:
            model_path = download_model(self._log)
            if not ensure_llama_cpp(self._log):
                self._log("[warn] Backend server will not be started.")
                self._log("[hint] Run LM Studio or another backend on 127.0.0.1:1234.")
                return
            cmd = [
                sys.executable,
                "-m",
                "llama_cpp.server",
                "--model",
                str(model_path),
                "--model_alias",
                "local-coder",
                "--host",
                "127.0.0.1",
                "--port",
                "1234",
                "--n_ctx",
                "8192",
            ]
            self._log("[info] Starting llama_cpp.server on 127.0.0.1:1234...")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with self._lock:
                self._process = proc
            # Launch a separate log reader thread so this worker can exit.
            self._log_thread = threading.Thread(
                target=self._log_reader,
                name="backend-logs",
                daemon=True,
            )
            self._log_thread.start()
        except Exception as exc:
            self._log(f"[error] Backend start failed: {exc}")

    def _log_reader(self) -> None:
        proc: Optional[subprocess.Popen]
        with self._lock:
            proc = self._process
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._log("[backend] " + line.rstrip())
        self._log("[info] Backend process exited.")

    def stop(self) -> None:
        with self._lock:
            if not self.is_running:
                self._log("[info] No backend process to stop.")
                return
            assert self._process is not None
            self._log("[info] Stopping backend process...")
            self._process.terminate()


class ChatClient:
    def __init__(self, config: Config, log: Callable[[str], None]) -> None:
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
        on_reply: Callable[[str], None],
        on_error: Callable[[str], None],
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
        self.title("Codex Portable Desktop (Non-blocking)")
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
            "Codex Portable Desktop (non-blocking backend)\n"
            "Use the AI Backend tab to start a local server, or point to LM Studio.\n\n",
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
        self._backend_manager.start_in_background()

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
        self._log_backend("[info] Settings applied to chat client.")

    def _poll_backend_status(self) -> None:
        if self._backend_manager.is_running:
            self._status_var.set("Backend: running on 127.0.0.1:1234")
        else:
            self._status_var.set("Backend: not running")
        self.after(1000, self._poll_backend_status)


def main() -> int:
    app = CodexApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
