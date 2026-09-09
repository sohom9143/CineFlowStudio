import os
import subprocess
import sys
from pathlib import Path

def main():
    if getattr(sys, 'frozen', False):
        app_dir = Path(sys.executable).parent.resolve()
    else:
        app_dir = Path(__file__).resolve().parent

    py_exe = app_dir / "dist" / "CineFlow-AI-Studio" / "CineFlow-AI-Studio.exe"
    py_script = app_dir / "desktop_app.py"

    print("=" * 65)
    print("  🎬 CineFlow-AI Studio: Launching Cinematic Desktop Application...")
    print("=" * 65)

    if py_exe.exists():
        print(f"  Starting CineFlow-AI Studio: {py_exe}")
        subprocess.run([str(py_exe)], cwd=str(py_exe.parent))
    elif py_script.exists():
        print(f"  Starting via python desktop_app.py...")
        subprocess.run([sys.executable, str(py_script)], cwd=str(app_dir))

if __name__ == '__main__':
    main()
