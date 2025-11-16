from __future__ import annotations

import sys
import json
import socket
import threading
import queue
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from codex_clone.config import load_config, Config
from codex_clone.logging_utils import log_line


HOST = "127.0.0.1"
PORT = 56789


class SocketBackendClient:
    """JSON-over-TCP client for the socket backend.

    - Keeps a persistent connection.
    - Background reader thread pushes messages into queues for the UI.
    - Auto-launches the daemon if the connection fails.
    - Provides a best-effort shutdown() method to tell the daemon to exit.
    """

    def __init__(
        self,
        backend_log_queue: "queue.Queue[str]",
        chat_queue: "queue.Queue[Tuple[str, str]]",
    ) -> None:
        self._backend_log_queue = backend_log_queue
        self._chat_queue = chat_queue
        self._sock: Optional[socket.socket] = None
        self._writer: Optional[Any] = None  # Text IO
        self._reader_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._connected = False
        self._req_counter = 0
        self._ever_connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def ensure_connected_async(self) -> None:
        """Ensure the daemon is running and we have a connection (background)."""

        def worker() -> None:
            if self._connected:
                return
            if not self._try_connect():
                # Attempt to start daemon and reconnect
                self._start_daemon()
                if not self._try_connect():
                    msg = "[gui] Could not connect to socket backend."
                    self._backend_log_queue.put(msg)
                    log_line(msg)
                    return
            msg = "[gui] Connected to socket backend."
            self._backend_log_queue.put(msg)
            log_line(msg)
            self._ever_connected = True

        threading.Thread(target=worker, daemon=True).start()

    def _try_connect(self) -> bool:
        try:
            sock = socket.create_connection((HOST, PORT), timeout=2.0)
        except OSError as exc:
            log_line(f"[gui] socket connect failed: {exc}")
            return False
        # Clear timeout so reads block normally; shutdown will close the socket.
        sock.settimeout(None)
        f_out = sock.makefile("w", encoding="utf-8")
        with self._lock:
            self._sock = sock
            self._writer = f_out
            self._connected = True
        self._reader_thread = threading.Thread(
            target=self._reader_worker, name="socket-backend-reader", daemon=True
        )
        self._reader_thread.start()
        return True

    def _start_daemon(self) -> None:
        try:
            subprocess.Popen(
                [sys.executable, "-m", "codex_clone.socket_backend"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log_line("[gui] Launched socket backend daemon process")
        except Exception as exc:
            msg = f"[gui] ERROR: failed to start socket backend daemon: {exc}"
            self._backend_log_queue.put(msg)
            log_line(msg)

    def _reader_worker(self) -> None:
        with self._lock:
            sock = self._sock
        if sock is None:
            return
        f_in = sock.makefile("r", encoding="utf-8")
        try:
            while True:
                try:
                    line = f_in.readline()
                except Exception as exc:  # includes TimeoutError, connection reset, etc.
                    err = f"[gui] socket reader error: {exc}"
                    self._backend_log_queue.put(err)
                    log_line(err)
                    # Surface as GUI dialog via chat_queue.
                    self._chat_queue.put(("error", err))
                    break
                if not line:
                    # EOF from daemon
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._handle_message(msg)
        finally:
            with self._lock:
                self._connected = False
            msg = "[gui] Disconnected from socket backend."
            self._backend_log_queue.put(msg)
            log_line(msg)

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "backend_log":
            text = str(msg.get("message", ""))
            self._backend_log_queue.put(text)
            log_line(text)
        elif mtype == "chat_reply":
            ok = bool(msg.get("ok", False))
            content = str(msg.get("content", ""))
            error = str(msg.get("error", ""))
            if ok:
                self._chat_queue.put(("reply", content))
                log_line("[gui] chat_reply ok")
            else:
                self._chat_queue.put(("error", error or "Chat request failed."))
                log_line(f"[gui] chat_reply error: {error}")
        elif mtype == "hello":
            text = str(msg.get("message", ""))
            line = f"[daemon] {text}"
            self._backend_log_queue.put(line)
            log_line(line)
        elif mtype == "status":
            running = bool(msg.get("running", False))
            if running:
                line = "[daemon] Backend helper is running."
            else:
                line = "[daemon] Backend helper is not running."
            self._backend_log_queue.put(line)
            log_line(line)
        elif mtype == "shutdown_ack":
            line = "[daemon] Shutdown acknowledged."
            self._backend_log_queue.put(line)
            log_line(line)

    def send_start_backend(self) -> None:
        self._send_async({"type": "start_backend"})

    def send_stop_backend(self) -> None:
        self._send_async({"type": "stop_backend"})

    def send_status(self) -> None:
        self._send_async({"type": "status"})

    def send_chat(self, messages: list[dict[str, str]]) -> None:
        req_id = self._next_req_id()
        payload = {"type": "chat", "id": req_id, "messages": messages}
        self._send_async(payload)

    def send_shutdown(self) -> None:
        """Best-effort shutdown request to daemon.

        Sends a 'shutdown' message and then closes the socket.
        """
        with self._lock:
            if not self._connected or self._writer is None:
                return
            try:
                text = json.dumps({"type": "shutdown"}, ensure_ascii=False)
                self._writer.write(text + "\n")
                self._writer.flush()
            except OSError:
                pass
            # Close socket; daemon will see EOF after handling shutdown.
            try:
                if self._sock is not None:
                    self._sock.close()
            except OSError:
                pass
            self._connected = False

    def _next_req_id(self) -> str:
        self._req_counter += 1
        return f"req-{self._req_counter}"

    def _send_async(self, obj: Dict[str, Any]) -> None:
        def worker() -> None:
            with self._lock:
                if not self._connected or self._writer is None:
                    msg = "[gui] Cannot send, socket backend not connected."
                    self._backend_log_queue.put(msg)
                    log_line(msg)
                    return
                try:
                    text = json.dumps(obj, ensure_ascii=False)
                    self._writer.write(text + "\n")
                    self._writer.flush()
                except OSError as exc:
                    msg = f"[gui] Send failed: {exc}"
                    self._backend_log_queue.put(msg)
                    log_line(msg)

        threading.Thread(target=worker, daemon=True).start()


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

    app.setStyleSheet(
        "QMainWindow {"
        "background-color: #0f172a;"
        "}"
        "QTabWidget::pane {"
        "border-top: 1px solid #111827;"
        "background: #020617;"
        "}"
        "QTabBar::tab {"
        "background: #020617;"
        "color: #9ca3af;"
        "padding: 6px 14px;"
        "border-radius: 6px 6px 0 0;"
        "margin-right: 2px;"
        "}"
        "QTabBar::tab:selected {"
        "background: #111827;"
        "color: #e5e7eb;"
        "}"
        "QPlainTextEdit, QTextEdit {"
        "background-color: #020617;"
        "color: #e5e7eb;"
        "border: 1px solid #1f2937;"
        "border-radius: 8px;"
        "padding: 6px;"
        "}"
        "QLineEdit, QSpinBox {"
        "background-color: #020617;"
        "color: #e5e7eb;"
        "border: 1px solid #1f2937;"
        "border-radius: 6px;"
        "padding: 4px 6px;"
        "selection-background-color: #3b82f6;"
        "}"
        "QLabel {"
        "color: #9ca3af;"
        "}"
        "QPushButton {"
        "background-color: #111827;"
        "color: #e5e7eb;"
        "border-radius: 8px;"
        "padding: 6px 14px;"
        "border: 1px solid #1f2937;"
        "}"
        "QPushButton:hover {"
        "background-color: #1f2937;"
        "}"
        "QPushButton:pressed {"
        "background-color: #020617;"
        "}"
        "QPushButton#primaryButton {"
        "background-color: #5865F2;"
        "border: none;"
        "}"
        "QPushButton#primaryButton:hover {"
        "background-color: #4f5ee8;"
        "}"
        "QPushButton#primaryButton:pressed {"
        "background-color: #4b56cf;"
        "}"
    )


def _load_app_icon() -> "Optional[object]":
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
            self.setWindowTitle("Codex Portable Desktop (Socket Backend, Clean)")
            self.resize(1000, 650)

            # Queues used by the socket client to feed data back to the UI.
            self.backend_log_queue: "queue.Queue[str]" = queue.Queue()
            self.chat_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()

            # Config + socket backend client.
            self.config: Config = load_config()
            self.socket_client = SocketBackendClient(
                self.backend_log_queue, self.chat_queue
            )
            self.socket_client.ensure_connected_async()

            # Status bar like a code editor: services + backend + last log line.
            self._init_status_bar()

            # Build tabs and timers.
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

        def _init_status_bar(self) -> None:
            from PyQt6 import QtWidgets as QtW

            bar = self.statusBar()
            bar.setSizeGripEnabled(False)

            self.sb_services = QtW.QLabel("Daemon: starting...")
            self.sb_backend = QtW.QLabel("Backend: idle")
            self.sb_lastlog = QtW.QLabel("Last log: (none)")

            for lbl in (self.sb_services, self.sb_backend, self.sb_lastlog):
                lbl.setStyleSheet("color: #9ca3af; padding: 0 8px;")

            bar.addPermanentWidget(self.sb_services)
            bar.addPermanentWidget(self.sb_backend, 1)
            bar.addPermanentWidget(self.sb_lastlog, 3)

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
                "Codex Portable Desktop (Socket Backend)\n"
                "The backend daemon handles HTTP + model server; this UI just renders.\n"
            )
            layout.addWidget(self.convo, stretch=1)

            bottom = QtWidgets.QHBoxLayout()
            bottom.setSpacing(8)
            layout.addLayout(bottom)

            self.prompt = QtWidgets.QTextEdit()
            self.prompt.setPlaceholderText(
                "Ask for code, refactors, or explanations..."
            )
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

            self.status_label = QtWidgets.QLabel("Backend: idle")
            font = self.status_label.font()
            font.setBold(True)
            self.status_label.setFont(font)
            outer.addWidget(self.status_label)

            # Progress/feedback widget.
            from PyQt6 import QtWidgets as QtW
            self.backend_progress = QtW.QProgressBar()
            self.backend_progress.setMinimum(0)
            self.backend_progress.setMaximum(1)
            self.backend_progress.setValue(0)
            self.backend_progress.setTextVisible(True)
            self.backend_progress.setFormat("Idle")
            outer.addWidget(self.backend_progress)

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

            self.status_button = QtWidgets.QPushButton("Check Status")
            self.status_button.clicked.connect(self._on_check_status)
            button_row.addWidget(self.status_button)

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
                # Update backend progress + status bar last-log entry.
                self._update_backend_progress_from_log(line)
                self.sb_lastlog.setText(f"Last log: {line.strip()}")
                self.backend_log.verticalScrollBar().setValue(
                    self.backend_log.verticalScrollBar().maximum()
                )

        def _update_backend_progress_from_log(self, line: str) -> None:
            text = line.strip()

            if "[gui] Start Local Backend clicked" in text:
                self.backend_progress.setRange(0, 0)
                self.backend_progress.setFormat("Starting backend...")
                self.status_label.setText("Backend: starting (launching helper)")
                self.sb_backend.setText("Backend: starting")
            elif "[daemon] backend_helper launched." in text:
                self.backend_progress.setRange(0, 0)
                self.backend_progress.setFormat("Backend helper launched, waiting for server...")
                self.status_label.setText("Backend: helper running (starting server)")
                self.sb_backend.setText("Backend: helper running")
            elif "[helper] Downloading model from Hugging Face" in text:
                self.backend_progress.setRange(0, 0)
                self.backend_progress.setFormat("Downloading model (first run may take a while)...")
                self.status_label.setText("Backend: downloading model")
                self.sb_backend.setText("Backend: downloading model")
            elif "[helper] Model already present" in text:
                self.backend_progress.setRange(0, 0)
                self.backend_progress.setFormat("Model present, starting server...")
                self.status_label.setText("Backend: starting server")
                self.sb_backend.setText("Backend: starting server")
            elif "[helper] Installing huggingface_hub" in text or "Installing llama-cpp-python" in text:
                self.backend_progress.setRange(0, 0)
                self.backend_progress.setFormat("Installing backend dependencies...")
                self.status_label.setText("Backend: installing dependencies")
                self.sb_backend.setText("Backend: installing deps")
            elif "[llama] " in text and "HTTP server listening" in text:
                self.backend_progress.setRange(0, 1)
                self.backend_progress.setValue(1)
                self.backend_progress.setFormat("Backend running")
                self.status_label.setText("Backend: running")
                self.sb_backend.setText("Backend: running")
            elif "backend_helper exited with code" in text:
                self.backend_progress.setRange(0, 1)
                self.backend_progress.setValue(0)
                self.backend_progress.setFormat("Backend stopped")
                self.status_label.setText("Backend: stopped")
                self.sb_backend.setText("Backend: stopped")
            elif "Backend helper is not running" in text:
                self.backend_progress.setRange(0, 1)
                self.backend_progress.setValue(0)
                self.backend_progress.setFormat("Idle")
                self.status_label.setText("Backend: idle (helper not running)")
                self.sb_backend.setText("Backend: idle")
            elif "ERROR" in text or "error" in text:
                self.backend_progress.setRange(0, 1)
                self.backend_progress.setValue(0)
                self.backend_progress.setFormat("Error")
                self.status_label.setText("Backend: error (see log)")
                self.sb_backend.setText("Backend: error")

            # Update services label for daemon/connection hints.
            if "Socket backend starting on" in text:
                self.sb_services.setText("Daemon: starting")
            elif "Socket backend listening" in text:
                self.sb_services.setText("Daemon: listening")
            elif "Socket backend exiting serve_forever" in text:
                self.sb_services.setText("Daemon: stopped")
            elif "[gui] Connected to socket backend." in text:
                self.sb_services.setText("Daemon: connected")
            elif "[gui] Disconnected from socket backend." in text:
                self.sb_services.setText("Daemon: disconnected")

        def _drain_chat_queue(self) -> None:
            while True:
                try:
                    kind, text = self.chat_queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "reply":
                    self.convo.appendPlainText("ai > " + text)
                else:
                    from PyQt6 import QtWidgets as QtW
                    QtW.QMessageBox.critical(self, "Error", text)
                    self.convo.appendPlainText("err> " + text)
                self.send_button.setEnabled(True)
                self.prompt.setFocus()
            self.convo.verticalScrollBar().setValue(
                self.convo.verticalScrollBar().maximum()
            )

        def _poll_backend_status(self) -> None:
            # Auto-healing: if we were ever connected and are now disconnected,
            # keep trying to reconnect in the background.
            if self.socket_client.is_connected:
                self.status_label.setText(
                    "Backend: socket daemon connected (see log)"
                )
                self.sb_services.setText("Daemon: connected")
            else:
                self.status_label.setText("Backend: not connected (auto-reconnect)")
                self.sb_services.setText("Daemon: reconnecting...")
                if self.socket_client._ever_connected:
                    self.socket_client.ensure_connected_async()

        def _on_send(self) -> None:
            user = self.prompt.toPlainText().strip()
            if not user:
                return
            self.prompt.clear()
            self.convo.appendPlainText("you> " + user)
            self.send_button.setEnabled(False)
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user},
            ]
            log_line("[gui] sending chat via socket backend")
            self.socket_client.send_chat(messages)

        def _on_start_backend(self) -> None:
            log_line("[gui] Start Local Backend clicked")
            self.socket_client.ensure_connected_async()
            self.socket_client.send_start_backend()

        def _on_stop_backend(self) -> None:
            log_line("[gui] Stop Backend clicked")
            self.socket_client.send_stop_backend()

        def _on_check_status(self) -> None:
            log_line("[gui] Check Status clicked")
            self.socket_client.send_status()

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[name-defined]
            log_line("[gui] closeEvent: sending shutdown to daemon")
            try:
                self.socket_client.send_shutdown()
            except Exception as exc:
                log_line(f"[gui] closeEvent shutdown error: {exc}")
            super().closeEvent(event)

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
    log_line("[gui] PyQt application started")
    rc = app.exec()
    log_line("[gui] PyQt application exiting")
    return rc


def main() -> int:
    try:
        import PyQt6  # noqa: F401
    except Exception as exc:
        print("PyQt6 is required for this frontend:", exc)
        log_line(f"[gui] PyQt6 import failed: {exc}")
        return 1
    return run_pyqt_app()


if __name__ == "__main__":
    raise SystemExit(main())
