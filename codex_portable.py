from __future__ import annotations

import sys
import subprocess
import threading
import queue
from pathlib import Path
from typing import Optional

from codex_clone.config import load_config, Config
from codex_clone.api import send_chat, CodexError


class BackendManager:
    """Helper-process backend manager with log queue."""

    def __init__(self, log_queue: "queue.Queue[str]") -> None:
        self._log_queue = log_queue
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
        with self._lock:
            if self.is_running:
                self._log_queue.put("[gui] Backend already running.")
                return
            if self._reader_thread and self._reader_thread.is_alive():
                self._log_queue.put("[gui] Backend startup already in progress.")
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
                self._log_queue.put(f"[gui] ERROR: Could not start backend helper: {exc}")
                return
            self._proc = proc
            self._reader_thread = threading.Thread(
                target=self._reader_worker,
                name="backend-helper-reader",
                daemon=True,
            )
            self._reader_thread.start()
            self._log_queue.put("[gui] Backend helper process started.")

    def _reader_worker(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._log_queue.put(line.rstrip())
        code = proc.wait()
        self._log_queue.put(f"[gui] Backend helper exited with code {code}.")

    def stop(self) -> None:
        with self._lock:
            if not self.is_running:
                self._log_queue.put("[gui] No backend process to stop.")
                return
            assert self._proc is not None
            self._log_queue.put("[gui] Terminating backend helper...")
            self._proc.terminate()


class ChatClient:
    """Background chat client using queues."""

    def __init__(self, config: Config, log_queue: "queue.Queue[str]") -> None:
        self._config = config
        self._log_queue = log_queue
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
        reply_queue: "queue.Queue[tuple[str, str]]",
    ) -> None:
        def worker() -> None:
            try:
                with self._lock:
                    self._messages.append({"role": "user", "content": user_text})
                    reply = send_chat(self._messages, self._config)
                    self._messages.append({"role": "assistant", "content": reply})
                reply_queue.put(("reply", reply))
            except CodexError as error:
                self._log_queue.put(f"[error] {error}")
                reply_queue.put(("error", str(error)))
            except Exception as exc:
                self._log_queue.put(f"[error] Unexpected: {exc}")
                reply_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()


# ---------------------- PyQt6 modern desktop UI ---------------------------


def _apply_modern_style(app: "QtWidgets.QApplication") -> None:  # type: ignore[name-defined]
    from PyQt6 import QtGui

    dark = QtGui.QPalette()
    dark.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#0f172a"))
    dark.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#e5e7eb"))
    dark.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#020617"))
    dark.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#020617"))
    dark.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#1f2937"))
    dark.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#e5e7eb"))
    dark.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#e5e7eb"))
    dark.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#111827"))
    dark.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#e5e7eb"))
    dark.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ffffff"))
    dark.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#5865F2"))
    dark.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#f9fafb"))
    app.setPalette(dark)

    # Discord-ish styling
    app.setStyleSheet(
        """
        QMainWindow {
            background-color: #0f172a;
        }
        QTabWidget::pane {
            border-top: 1px solid #111827;
            background: #020617;
        }
        QTabBar::tab {
            background: #020617;
            color: #9ca3af;
            padding: 6px 14px;
            border-radius: 6px 6px 0 0;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #111827;
            color: #e5e7eb;
        }
        QPlainTextEdit, QTextEdit {
            background-color: #020617;
            color: #e5e7eb;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 6px;
        }
        QLineEdit, QSpinBox {
            background-color: #020617;
            color: #e5e7eb;
            border: 1px solid #1f2937;
            border-radius: 6px;
            padding: 4px 6px;
            selection-background-color: #3b82f6;
        }
        QLabel {
            color: #9ca3af;
        }
        QPushButton {
            background-color: #111827;
            color: #e5e7eb;
            border-radius: 8px;
            padding: 6px 14px;
            border: 1px solid #1f2937;
        }
        QPushButton:hover {
            background-color: #1f2937;
        }
        QPushButton:pressed {
            background-color: #020617;
        }
        QPushButton#primaryButton {
            background-color: #5865F2;
            border: none;
        }
        QPushButton#primaryButton:hover {
            background-color: #4f5ee8;
        }
        QPushButton#primaryButton:pressed {
            background-color: #4b56cf;
        }
        """
    )


def _load_app_icon() -> "Optional[object]":  # QIcon, but avoid importing in sig
    from PyQt6 import QtGui

    icon_path = Path(__file__).with_name("icon.svg")
    if not icon_path.exists():
        return None
    icon = QtGui.QIcon(str(icon_path))
    if icon.isNull():
        return None
    return icon


def run_pyqt_app() -> int:
    from PyQt6 import QtWidgets, QtCore, QtGui

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Codex Portable Desktop")
            self.resize(1000, 650)

            self.backend_log_queue: "queue.Queue[str]" = queue.Queue()
            self.chat_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()

            self.config: Config = load_config()
            self.backend_manager = BackendManager(self.backend_log_queue)
            self.chat_client = ChatClient(self.config, self.backend_log_queue)

            self._build_ui()

            self._backend_timer = QtCore.QTimer(self)
            self._backend_timer.timeout.connect(self._drain_backend_log)
            self._backend_timer.start(100)

            self._chat_timer = QtCore.QTimer(self)
            self._chat_timer.timeout.connect(self._drain_chat_queue)
            self._chat_timer.start(80)

            self._status_timer = QtCore.QTimer(self)
            self._status_timer.timeout.connect(self._poll_backend_status)
            self._status_timer.start(1000)

        def _build_ui(self) -> None:
            tabs = QtWidgets.QTabWidget()
            self.setCentralWidget(tabs)

            chat_tab = QtWidgets.QWidget()
            tabs.addTab(chat_tab, "Chat")
            self._build_chat_tab(chat_tab)

            backend_tab = QtWidgets.QWidget()
            tabs.addTab(backend_tab, "AI Backend")
            self._build_backend_tab(backend_tab)

        def _build_chat_tab(self, parent: QtWidgets.QWidget) -> None:
            layout = QtWidgets.QVBoxLayout(parent)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            self.convo = QtWidgets.QPlainTextEdit()
            self.convo.setReadOnly(True)
            self.convo.appendPlainText(
                "Codex Portable Desktop (PyQt6)\n"
                "Use the AI Backend tab to start a local server or point to LM Studio.\n"
            )
            layout.addWidget(self.convo, stretch=1)

            bottom = QtWidgets.QHBoxLayout()
            bottom.setSpacing(8)
            layout.addLayout(bottom)

            self.prompt = QtWidgets.QTextEdit()
            self.prompt.setPlaceholderText("Ask for code, refactors, or explanations...")
            self.prompt.setFixedHeight(90)
            bottom.addWidget(self.prompt, stretch=1)

            self.send_button = QtWidgets.QPushButton("Send")
            self.send_button.setObjectName("primaryButton")
            self.send_button.clicked.connect(self._on_send)
            bottom.addWidget(self.send_button)

            shortcut = QtGui.QShortcut(
                QtGui.QKeySequence("Ctrl+Return"), parent
            )
            shortcut.activated.connect(self._on_send)

        def _build_backend_tab(self, parent: QtWidgets.QWidget) -> None:
            outer = QtWidgets.QVBoxLayout(parent)
            outer.setContentsMargins(12, 12, 12, 12)
            outer.setSpacing(10)

            form = QtWidgets.QGridLayout()
            form.setHorizontalSpacing(10)
            form.setVerticalSpacing(8)
            outer.addLayout(form)

            self.status_label = QtWidgets.QLabel("Backend: unknown")
            font = self.status_label.font()
            font.setBold(True)
            self.status_label.setFont(font)
            form.addWidget(self.status_label, 0, 0, 1, 2)

            form.addWidget(QtWidgets.QLabel("Base URL:"), 1, 0)
            self.base_url_edit = QtWidgets.QLineEdit(self.config.base_url)
            form.addWidget(self.base_url_edit, 1, 1)

            form.addWidget(QtWidgets.QLabel("Model name:"), 2, 0)
            self.model_edit = QtWidgets.QLineEdit(self.config.model)
            form.addWidget(self.model_edit, 2, 1)

            form.addWidget(QtWidgets.QLabel("Temperature:"), 3, 0)
            self.temp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.temp_slider.setMinimum(0)
            self.temp_slider.setMaximum(100)
            self.temp_slider.setValue(int(self.config.temperature * 100))
            form.addWidget(self.temp_slider, 3, 1)

            form.addWidget(QtWidgets.QLabel("Max tokens:"), 4, 0)
            self.max_tokens_spin = QtWidgets.QSpinBox()
            self.max_tokens_spin.setMinimum(128)
            self.max_tokens_spin.setMaximum(32768)
            self.max_tokens_spin.setValue(self.config.max_tokens)
            form.addWidget(self.max_tokens_spin, 4, 1)

            button_row = QtWidgets.QHBoxLayout()
            button_row.setSpacing(8)
            outer.addLayout(button_row)

            self.start_button = QtWidgets.QPushButton("Start Local Backend")
            self.start_button.setObjectName("primaryButton")
            self.start_button.clicked.connect(self._on_start_backend)
            button_row.addWidget(self.start_button)

            self.stop_button = QtWidgets.QPushButton("Stop Backend")
            self.stop_button.clicked.connect(self._on_stop_backend)
            button_row.addWidget(self.stop_button)

            self.apply_button = QtWidgets.QPushButton("Apply Settings")
            self.apply_button.clicked.connect(self._on_apply_settings)
            button_row.addWidget(self.apply_button)
            button_row.addStretch(1)

            self.backend_log = QtWidgets.QPlainTextEdit()
            self.backend_log.setReadOnly(True)
            outer.addWidget(self.backend_log, stretch=1)

        def _drain_backend_log(self) -> None:
            while True:
                try:
                    line = self.backend_log_queue.get_nowait()
                except queue.Empty:
                    break
                self.backend_log.appendPlainText(line)
                self.backend_log.verticalScrollBar().setValue(
                    self.backend_log.verticalScrollBar().maximum()
                )

        def _drain_chat_queue(self) -> None:
            while True:
                try:
                    kind, text = self.chat_queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "reply":
                    self.convo.appendPlainText("ai > " + text)
                else:
                    QtWidgets.QMessageBox.critical(self, "Error", text)
                    self.convo.appendPlainText("err> " + text)
                self.send_button.setEnabled(True)
                self.prompt.setFocus()
            self.convo.verticalScrollBar().setValue(
                self.convo.verticalScrollBar().maximum()
            )

        def _poll_backend_status(self) -> None:
            if self.backend_manager.is_running:
                self.status_label.setText("Backend: helper running (see log)")
            else:
                self.status_label.setText("Backend: not running")


        def _on_send(self) -> None:
            user = self.prompt.toPlainText().strip()
            if not user:
                return
            self.prompt.clear()
            self.convo.appendPlainText("you> " + user)
            self.send_button.setEnabled(False)
            self.chat_client.send_async(user, self.chat_queue)

        def _on_start_backend(self) -> None:
            self.backend_manager.start()

        def _on_stop_backend(self) -> None:
            self.backend_manager.stop()

        def _on_apply_settings(self) -> None:
            temp = self.temp_slider.value() / 100.0
            max_tokens = int(self.max_tokens_spin.value())
            base_url = self.base_url_edit.text().strip() or self.config.base_url
            model = self.model_edit.text().strip() or self.config.model
            new_cfg = Config(
                base_url=base_url,
                api_key=self.config.api_key,
                model=model,
                system_prompt=self.config.system_prompt,
                temperature=temp,
                max_tokens=max_tokens,
            )
            self.config = new_cfg
            self.chat_client.update_config(new_cfg)
            self.backend_log.appendPlainText("[gui] Settings applied to chat client.")

    from PyQt6 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    _apply_modern_style(app)
    icon = _load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    window = MainWindow()
    if icon is not None:
        window.setWindowIcon(icon)
    window.show()
    return app.exec()


# ------------------ Tkinter / curses PyQt6 auto-installer -----------------


def run_tk_installer() -> int:
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except Exception:
        return run_curses_installer()

    log_queue: "queue.Queue[str]" = queue.Queue()

    def installer_thread() -> None:
        cmd = [sys.executable, "-m", "pip", "install", "PyQt6"]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            log_queue.put(f"ERROR: could not start pip: {exc}")
            return
        assert proc.stdout is not None
        for line in proc.stdout:
            log_queue.put(line.rstrip())
        code = proc.wait()
        log_queue.put(f"Installer exited with code {code}.")
        if code == 0:
            log_queue.put("PyQt6 installed successfully. Close this window and restart.")
        else:
            log_queue.put("PyQt6 install failed. See output above.")

    root = tk.Tk()
    root.title("Codex Portable - PyQt6 Auto-Installer (Tkinter)")
    root.geometry("700x400")

    text = scrolledtext.ScrolledText(root, wrap=tk.WORD)
    text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
    text.insert(
        tk.END,
        "PyQt6 is not installed.\n\n"
        "An automatic install has been started using pip.\n"
        "Progress will appear here.\n\n",
    )
    text.config(state=tk.DISABLED)

    def append_log(line: str) -> None:
        text.config(state=tk.NORMAL)
        text.insert(tk.END, line + "\n")
        text.see(tk.END)
        text.config(state=tk.DISABLED)

    def poll_queue() -> None:
        while True:
            try:
                line = log_queue.get_nowait()
            except queue.Empty:
                break
            append_log(line)
        root.after(100, poll_queue)

    threading.Thread(target=installer_thread, daemon=True).start()
    poll_queue()
    root.mainloop()
    return 0


def run_curses_installer() -> int:
    try:
        import curses  # type: ignore[unused-ignore]
    except Exception:
        cmd = [sys.executable, "-m", "pip", "install", "PyQt6"]
        return subprocess.call(cmd)

    def _main(stdscr: "curses._CursesWindow") -> int:  # type: ignore[name-defined]
        curses.curs_set(0)
        stdscr.clear()
        stdscr.addstr(0, 0, "PyQt6 is not installed. Installing automatically...")
        stdscr.refresh()
        cmd = [sys.executable, "-m", "pip", "install", "PyQt6"]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        y = 2
        for line in proc.stdout:
            if y >= curses.LINES - 1:
                stdscr.scroll(1)
                y = curses.LINES - 2
            stdscr.addstr(y, 0, line[: curses.COLS - 1])
            y += 1
            stdscr.refresh()
        code = proc.wait()
        stdscr.addstr(y, 0, f"Installer exited with code {code}. Press any key.")
        stdscr.getch()
        return code

    import curses  # type: ignore[redefined-outer-name]

    return curses.wrapper(_main)


def main() -> int:
    try:
        import PyQt6  # noqa: F401
    except Exception:
        return run_tk_installer()
    return run_pyqt_app()


if __name__ == "__main__":
    raise SystemExit(main())
