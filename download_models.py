"""
CineFlow-AI: Standalone Model Weights Downloader & Verifier
===========================================================
Executes pre-flight model check and downloads all neural network weights.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.model_downloader import check_model_status, download_all_models

def main():
    print("=" * 70)
    print("  🎬 CineFlow-AI Studio: Model Checkpoints & Face Bank Downloader")
    print("=" * 70)
    
    print("\n[1/2] Checking existing model status...")
    models = check_model_status()
    for m in models:
        size_info = f"{m['size_mb']} MB" if m['size_mb'] > 0 else "Auto-Cached"
        print(f"  - [{m['status']:<10}] {m['name']:<30} ({m['category']}) - {size_info}")

    print("\n[2/2] Downloading & verifying required model weights...")
    def print_progress(filename: str, pct: float, status_text: str):
        print(f"\r  --> {status_text}", end="", flush=True)

    results = download_all_models(include_optional=False, progress_callback=print_progress)
    print("\n")
    
    print("=" * 70)
    print("  Summary:")
    for name, success in results.items():
        status_str = "✅ Ready" if success else "⚠️ Fallback Active"
        print(f"  - {name}: {status_str}")
    print("=" * 70)
    print("All weights configured. You can now launch CineFlow-AI-Studio.exe!")

if __name__ == "__main__":
    main()
