from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final

from .logging_utils import log_line


HF_REPO: Final[str] = "TheBloke/deepseek-coder-6.7B-instruct-GGUF"
HF_FILE: Final[str] = "deepseek-coder-6.7b-instruct.Q4_K_M.gguf"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    directory = project_root() / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def log(msg: str) -> None:
    print(msg, flush=True)
    log_line(msg)


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
        return
    except Exception:
        pass
    log("[helper] Installing huggingface_hub...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "huggingface_hub>=0.25.0"],
        check=False,
    )


def download_model() -> Path:
    ensure_huggingface_hub()
    from huggingface_hub import hf_hub_download

    dest_dir = models_dir()
    local_path = dest_dir / HF_FILE
    if local_path.exists():
        log(f"[helper] Model already present: {local_path}")
        return local_path
    log("[helper] Downloading model from Hugging Face... (first run)")
    actual = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    local_path = Path(actual)
    log(f"[helper] Model downloaded to: {local_path}")
    return local_path


def have_llama_server() -> bool:
    try:
        import llama_cpp.server  # type: ignore[unused-ignore]  # noqa: F401
        return True
    except Exception:
        return False


def ensure_llama_cpp() -> bool:
    if have_llama_server():
        return True
    log("[helper] Installing llama-cpp-python[server] via pip...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "llama-cpp-python[server]"],
        check=False,
    )
    if result.returncode == 0 and have_llama_server():
        log("[helper] llama-cpp-python[server] is ready.")
        return True
    log(
        "[helper] WARNING: Could not install llama-cpp-python[server].\n"
        "[helper] You can still run LM Studio or another backend manually."
    )
    return False


def run_llama_server(model_path: Path) -> int:
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
    log("[helper] Starting llama_cpp.server on 127.0.0.1:1234...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        msg = "[llama] " + line.rstrip()
        print(msg, flush=True)
        log_line(msg)
    code = proc.wait()
    log(f"[helper] llama_cpp.server exited with code {code}.")
    return code


def main() -> int:
    log("[helper] backend_helper starting up")
    try:
        model_path = download_model()
    except Exception as exc:
        log(f"[helper] ERROR: model download failed: {exc}")
        return 1
    if not ensure_llama_cpp():
        log("[helper] Exiting without starting backend.")
        return 0
    rc = run_llama_server(model_path)
    log("[helper] backend_helper shutting down")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
