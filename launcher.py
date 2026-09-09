import os
import subprocess
import sys
from pathlib import Path

def main():
    if getattr(sys, 'frozen', False):
        app_dir = Path(sys.executable).parent.resolve()
    else:
        app_dir = Path(__file__).resolve().parent

    csharp_exe = app_dir / "dist" / "CineFlowStudio-Desktop" / "CineFlowStudio.Desktop.exe"
    bat_file = app_dir / "run_desktop_csharp.bat"
    old_exe = app_dir / "dist" / "CineFlow-AI-Studio" / "CineFlow-AI-Studio.exe"

    print("=" * 65)
    print("  🎬 CineFlow-AI Studio: Launching Enterprise C# Desktop Edition...")
    print("=" * 65)

    if csharp_exe.exists():
        print(f"  Starting Enterprise C# Studio: {csharp_exe}")
        subprocess.run([str(csharp_exe)], cwd=str(csharp_exe.parent))
    elif bat_file.exists():
        print(f"  Starting via run_desktop_csharp.bat...")
        subprocess.run(["cmd.exe", "/c", str(bat_file)], cwd=str(app_dir))
    elif old_exe.exists():
        subprocess.run([str(old_exe)], cwd=str(old_exe.parent))

if __name__ == '__main__':
    main()
