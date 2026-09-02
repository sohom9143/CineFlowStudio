import os
import subprocess
import sys
from pathlib import Path

def main():
    if getattr(sys, 'frozen', False):
        app_dir = Path(sys.executable).parent.resolve()
    else:
        app_dir = Path(__file__).resolve().parent

    bat_file = app_dir / 'run_app.bat'

    print('=' * 65)
    print('  CineFlow-AI Studio: Launching Live Backend via run_app.bat...')
    print('=' * 65)

    if bat_file.exists():
        subprocess.run(['cmd.exe', '/c', str(bat_file)], cwd=str(app_dir))
    else:
        py_app = app_dir / 'desktop_app.py'
        subprocess.run([sys.executable, str(py_app)], cwd=str(app_dir))

if __name__ == '__main__':
    main()
