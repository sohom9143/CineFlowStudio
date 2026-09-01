"""
CineFlow-AI: PyInstaller Executable Builder
==========================================
Builds a standalone or folder-based Windows .exe application for CineFlow-AI,
properly bundling all required dependencies, static templates, safehttpx data,
and Gradio / PyWebView assets.
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def build():
    print("=" * 65)
    print("  Building CineFlow-AI Desktop Executable (.exe)")
    print("=" * 65)

    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name", "CineFlow-AI-Studio",
        "--add-data", f"configs{os.pathsep}configs",
        "--add-data", f"character_profiles{os.pathsep}character_profiles",
        "--add-data", f"modules{os.pathsep}modules",
        "--collect-all", "groovy",
        "--collect-all", "safehttpx",
        "--collect-all", "gradio",
        "--collect-all", "gradio_client",
        "--collect-all", "hf_gradio",
        "--collect-all", "webview",
        "--collect-all", "uvicorn",
        "--collect-all", "fastapi",
        "--collect-all", "starlette",
        "--collect-all", "pydantic",
        "--collect-all", "jinja2",
        "--collect-all", "huggingface_hub",
        "--collect-all", "multipart",
        "--collect-all", "tomlkit",
        "--collect-all", "typer",
        "--collect-all", "semantic_version",
        "--collect-all", "pydub",
        "--collect-all", "anyio",
        "--hidden-import", "clr",
        "--hidden-import", "yaml",
        "--hidden-import", "PIL",
        "--hidden-import", "numpy",
        "--hidden-import", "modules.model_downloader",
        "--hidden-import", "modules.memory_manager",
        "--hidden-import", "modules.character_engine",
        "--hidden-import", "modules.video_engine",
        "--hidden-import", "modules.lipsync_engine",
        "--hidden-import", "modules.post_processing",
        "desktop_app.py",
    ]

    print(f"Running PyInstaller in: {BASE_DIR}")
    print("Command:", " ".join(pyinstaller_cmd))
    
    result = subprocess.run(pyinstaller_cmd, cwd=str(BASE_DIR))
    if result.returncode == 0:
        dist_dir = BASE_DIR / "dist" / "CineFlow-AI-Studio"
        dist_internal = dist_dir / "_internal"
        dist_path = dist_dir / "CineFlow-AI-Studio.exe"

        # Post-build safeguard: ensure critical version and resource files exist
        for pkg_name, default_ver in [("groovy", "0.1.2"), ("safehttpx", "0.1.7")]:
            for target_base in [dist_dir, dist_internal]:
                target_pkg = target_base / pkg_name
                try:
                    target_pkg.mkdir(parents=True, exist_ok=True)
                    v_file = target_pkg / "version.txt"
                    if not v_file.exists():
                        v_file.write_text(default_ver, encoding="utf-8")
                except Exception as ex:
                    print(f"  [Warning] Failed to write {pkg_name}/version.txt in {target_base}: {ex}")

        print("\n" + "=" * 65)
        print("  BUILD SUCCESSFUL!")
        print(f"  Executable location: {dist_path}")
        print("=" * 65)
    else:
        print("\n" + "=" * 65)
        print(f"  BUILD FAILED with return code {result.returncode}")
        print("=" * 65)

if __name__ == "__main__":
    build()
