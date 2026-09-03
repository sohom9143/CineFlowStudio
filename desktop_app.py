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
from app import CineFlowApp, build_gradio_ui, build_fastapi_app, FASTAPI_AVAILABLE

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


def get_saved_colab_url() -> Optional[str]:
    """Reads saved Colab URL from environment or configs/colab_connection.json."""
    env_url = os.getenv("CINEFLOW_COLAB_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")

    candidates = [
        APP_DIR / "configs" / "colab_connection.json",
        BASE_DIR / "configs" / "colab_connection.json",
        Path(os.getcwd()) / "configs" / "colab_connection.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                import json
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    url = data.get("colab_url", "").strip()
                    if url:
                        return url.rstrip("/")
            except Exception:
                pass
    return None


def verify_colab_url(url: str, timeout: float = 3.5) -> bool:
    """Checks if remote Colab endpoint is reachable and healthy."""
    if not url:
        return False
    health_url = f"{url.rstrip('/')}/api/health"
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "CineFlow-Desktop"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return True
    except Exception:
        try:
            req = urllib.request.Request(url.rstrip("/"), headers={"User-Agent": "CineFlow-Desktop"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status in (200, 302, 307):
                    return True
        except Exception:
            return False
    return False


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
        if FASTAPI_AVAILABLE:
            import uvicorn
            fastapi_app = build_fastapi_app(app)
            logger.info(f"Starting local Studio server at http://127.0.0.1:{port} ...")
            config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=port, log_level="warning")
            server = uvicorn.Server(config)
            server.run()
        else:
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
    """Main launcher for Desktop Application with Colab Cloud GPU Bridge."""
    port = find_free_port(7860)
    local_server_url = f"http://127.0.0.1:{port}"

    config_file = "configs/colab_t4_config.yaml"
    if not os.path.exists(config_file):
        configs = list(Path("configs").glob("*.yaml"))
        if configs:
            config_file = str(configs[0])

    # 1. Start local backend in background thread (handles connection portal and offline fallback)
    server_thread = threading.Thread(
        target=start_gradio_server,
        args=(port, config_file),
        daemon=True,
    )
    server_thread.start()

    logger.info("Waiting for CineFlow-AI local server to initialize...")
    wait_for_server(local_server_url, timeout=20.0)

    # 2. Check for previously saved or configured Colab Cloud GPU URL
    saved_url = get_saved_colab_url()
    target_url = f"{local_server_url}/connect"
    window_title = "🎬 CineFlow-AI Studio - Cloud GPU Bridge"

    if saved_url:
        logger.info(f"Checking saved Colab URL: {saved_url} ...")
        if verify_colab_url(saved_url, timeout=3.0):
            logger.info(f"✅ Verified Colab Cloud GPU backend online at {saved_url}!")
            target_url = saved_url
            window_title = "🎬 CineFlow-AI Studio [Cloud GPU Active - Colab T4]"
        else:
            logger.info(f"Saved Colab session ({saved_url}) is inactive. Loading Connection Portal...")
            target_url = f"{local_server_url}/connect"
    else:
        logger.info("No active Colab URL found. Loading Cloud GPU Connection Portal...")
        target_url = f"{local_server_url}/connect"

    # Auto-open Google Colab in browser when starting in portal mode
    auto_colab = os.getenv("CINEFLOW_AUTO_OPEN_COLAB", "1").strip().lower() not in ("0", "false", "no")
    if auto_colab and target_url == f"{local_server_url}/connect":
        colab_thread = threading.Thread(target=open_colab_notebook, daemon=True)
        colab_thread.start()

    logger.info(f"Opening CineFlow-AI Studio Desktop Window at: {target_url}")

    # Create native desktop window using pywebview
    window = webview.create_window(
        title=window_title,
        url=target_url,
        width=1360,
        height=880,
        resizable=True,
        min_size=(960, 640),
        confirm_close=False,
        background_color="#0f131c",
    )

    # Start the desktop window event loop
    webview.start(debug=False)
    logger.info("CineFlow-AI Studio Desktop closed.")


if __name__ == "__main__":
    main()


