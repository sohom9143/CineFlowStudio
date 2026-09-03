"""
CineFlow-AI: PyInstaller Executable Builder
==========================================
Builds a standalone or folder-based Windows .exe application for CineFlow-AI,
properly bundling all required dependencies, static templates, safehttpx data,
FastAPI / Uvicorn server, and Gradio / PyWebView assets.
"""

import os
import shutil
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
        "--add-data", f"static{os.pathsep}static",
        "--add-data", f"database{os.pathsep}database",
        "--add-data", f"code.html{os.pathsep}.",
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
        "--hidden-import", "modules.character_database",
        "--hidden-import", "modules.video_engine",
        "--hidden-import", "modules.lipsync_engine",
        "--hidden-import", "modules.post_processing",
        "--hidden-import", "email.mime.multipart",
        "--hidden-import", "email.mime.text",
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

        # Ensure static, database, and character_profiles exist in dist_dir as well
        for folder_name in ["static", "database", "configs"]:
            src = BASE_DIR / folder_name
            for dst_base in [dist_dir, dist_internal]:
                dst = dst_base / folder_name
                if src.exists() and not dst.exists():
                    try:
                        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
                        print(f"  Synced {folder_name} to {dst}")
                    except Exception as ex:
                        print(f"  [Warning] Could not copy {folder_name} to {dst}: {ex}")

        # Copy code.html to dist_dir
        for dst_base in [dist_dir, dist_internal]:
            src_code = BASE_DIR / "code.html"
            dst_code = dst_base / "code.html"
            if src_code.exists():
                try:
                    shutil.copy2(str(src_code), str(dst_code))
                except Exception:
                    pass

        # Purge any legacy character folders from dist
        for base in [dist_dir, dist_internal]:
            char_base = base / "character_profiles"
            if char_base.exists():
                for legacy in ["dev", "neel", "meghla", "cha_kaku"]:
                    leg_dir = char_base / legacy
                    if leg_dir.exists():
                        shutil.rmtree(str(leg_dir), ignore_errors=True)
                        print(f"  Purged legacy character '{legacy}' from {leg_dir}")

        # Rebuild root Launcher executable
        print("\n" + "=" * 65)
        print("  Building Root Launcher: CineFlow-AI-Studio.exe...")
        print("=" * 65)
        launcher_cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--name", "CineFlow-AI-Studio-Launcher",
            "launcher.py",
        ]
        res_launcher = subprocess.run(launcher_cmd, cwd=str(BASE_DIR))
        if res_launcher.returncode == 0:
            launcher_src = BASE_DIR / "dist" / "CineFlow-AI-Studio-Launcher.exe"
            root_launcher = BASE_DIR / "CineFlow-AI-Studio-Launcher.exe"
            root_exe = BASE_DIR / "CineFlow-AI-Studio.exe"
            if launcher_src.exists():
                shutil.copy2(str(launcher_src), str(root_launcher))
                shutil.copy2(str(launcher_src), str(root_exe))
                print(f"  Successfully updated root executables:")
                print(f"    - {root_launcher}")
                print(f"    - {root_exe}")

        print("\n" + "=" * 65)
        print("  BUILD SUCCESSFUL!")
        print(f"  Standalone Studio Executable: {dist_path}")
        print(f"  Root Launcher Executable:     {BASE_DIR / 'CineFlow-AI-Studio.exe'}")
        print("=" * 65)
    else:
        print("\n" + "=" * 65)
        print(f"  BUILD FAILED with return code {result.returncode}")
        print("=" * 65)

if __name__ == "__main__":
    build()
