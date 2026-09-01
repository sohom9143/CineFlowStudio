"""
CineFlow-AI: Native Desktop Application Launcher
================================================
Embeds the CineFlow-AI Gradio WebUI inside a native Windows desktop window using pywebview.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Set up root directories whether running as a standard script or bundled by PyInstaller
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    BASE_DIR = Path(sys._MEIPASS).resolve()
    APP_DIR = Path(sys.executable).parent.resolve()
else:
    BASE_DIR = Path(__file__).resolve().parent
    APP_DIR = BASE_DIR

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Ensure required package metadata/version files exist in frozen PyInstaller bundles
if getattr(sys, "frozen", False):
    critical_version_files = {
        "groovy": "0.1.2",
        "safehttpx": "0.1.7",
    }
    candidate_dirs = [BASE_DIR, APP_DIR, APP_DIR / "_internal"]
    for pkg_name, default_ver in critical_version_files.items():
        for candidate in candidate_dirs:
            target_pkg_dir = candidate / pkg_name
            try:
                target_pkg_dir.mkdir(parents=True, exist_ok=True)
                ver_file = target_pkg_dir / "version.txt"
                if not ver_file.exists():
                    ver_file.write_text(default_ver, encoding="utf-8")
            except Exception:
                pass

os.chdir(APP_DIR)

import webview
import webbrowser
from app import CineFlowApp, build_gradio_ui

COLAB_NOTEBOOK_URL = "https://colab.research.google.com/github/sohom9143/CineFlowStudio/blob/main/CineFlow_Colab_FreeTier.ipynb"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CineFlow.Desktop")


def find_free_port(start_port: int = 7860) -> int:
    """Finds an available TCP port starting from start_port."""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Polls server URL until it responds with HTTP 200/OK or timeout expires."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status in (200, 302, 307):
                    return True
        except Exception:
            time.sleep(0.4)
    return False


def open_colab_notebook() -> None:
    """Opens the Google Colab Free Tier notebook in the default web browser."""
    try:
        logger.info(f"Opening Google Colab runtime: {COLAB_NOTEBOOK_URL}")
        webbrowser.open(COLAB_NOTEBOOK_URL)
    except Exception as e:
        logger.warning(f"Could not open Google Colab in browser automatically: {e}")


def start_gradio_server(port: int, config_path: str = "configs/colab_t4_config.yaml") -> None:
    """Initializes and starts the Gradio backend in a daemon thread."""
    try:
        # Check config in APP_DIR or BASE_DIR
        resolved_config = config_path
        if not os.path.exists(resolved_config):
            alt_path = os.path.join(str(BASE_DIR), config_path)
            if os.path.exists(alt_path):
                resolved_config = alt_path

        logger.info(f"Initializing CineFlow-AI Studio backend with '{resolved_config}'...")
        app = CineFlowApp(config_path=resolved_config)
        demo = build_gradio_ui(app)
        logger.info(f"Starting local server at http://127.0.0.1:{port} ...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            inbrowser=False,
            prevent_thread_lock=True,
            show_error=True,
            quiet=True,
        )
    except Exception as e:
        logger.error(f"Error starting Gradio server: {e}", exc_info=True)


def main() -> None:
    """Main launcher for Desktop Application."""
    port = find_free_port(7860)
    server_url = f"http://127.0.0.1:{port}"

    config_file = "configs/colab_t4_config.yaml"
    if not os.path.exists(config_file):
        configs = list(Path("configs").glob("*.yaml"))
        if configs:
            config_file = str(configs[0])

    # Auto-open Google Colab in browser when launching the desktop app
    auto_colab = os.getenv("CINEFLOW_AUTO_OPEN_COLAB", "1").strip().lower() not in ("0", "false", "no")
    if auto_colab:
        colab_thread = threading.Thread(target=open_colab_notebook, daemon=True)
        colab_thread.start()

    # Start Gradio in a background thread
    server_thread = threading.Thread(
        target=start_gradio_server,
        args=(port, config_file),
        daemon=True,
    )
    server_thread.start()

    logger.info("Waiting for CineFlow-AI interface to be ready...")
    if not wait_for_server(server_url, timeout=30.0):
        logger.warning("Server took longer than expected to respond, opening window anyway...")

    logger.info("Opening CineFlow-AI Studio Desktop Window...")
    
    # Create native desktop window using pywebview
    window = webview.create_window(
        title="🎬 CineFlow-AI Studio - Cinematic AI Video Suite",
        url=server_url,
        width=1360,
        height=880,
        resizable=True,
        min_size=(960, 640),
        confirm_close=False,
        background_color="#111827",
    )

    # Start the desktop window event loop
    webview.start(debug=False)
    logger.info("CineFlow-AI Studio Desktop closed.")


if __name__ == "__main__":
    main()

