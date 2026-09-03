import os
import subprocess
import sys
from pathlib import Path

def main():
    if getattr(sys, 'frozen', False):
        app_dir = Path(sys.executable).parent.resolve()
    else:
        app_dir = Path(__file__).resolve().parent

    compiled_exe = app_dir / "dist" / "CineFlow-AI-Studio" / "CineFlow-AI-Studio.exe"
    bat_file = app_dir / "run_app.bat"
    py_app = app_dir / "desktop_app.py"

    print("=" * 65)
    print("  🎬 CineFlow-AI Studio: Launching Cinematic Desktop Application...")
    print("=" * 65)

    if compiled_exe.exists():
        print(f"  Starting compiled studio: {compiled_exe}")
        subprocess.run([str(compiled_exe)], cwd=str(compiled_exe.parent))
    elif bat_file.exists():
        print(f"  Starting via run_app.bat...")
        subprocess.run(["cmd.exe", "/c", str(bat_file)], cwd=str(app_dir))
    elif py_app.exists():
        print(f"  Starting via python desktop_app.py...")
        subprocess.run([sys.executable, str(py_app)], cwd=str(app_dir))

if __name__ == '__main__':
    main()
