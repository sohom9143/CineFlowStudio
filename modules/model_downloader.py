"""
CineFlow-AI: Automated Model Checkpoint Downloader & Hub
========================================================
Manages discovery, verification, and automatic downloading of all required
neural network checkpoints, face recognition models, and diffusion weights.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("CineFlow.ModelDownloader")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Base models directory
MODELS_DIR = Path("models").resolve()

# Direct HTTPS Checkpoint Registry
CHECKPOINT_REGISTRY: Dict[str, Dict[str, str]] = {
    "RealESRGAN_x4plus.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "category": "Post-Processing (Super Resolution)",
        "dest": "models/RealESRGAN_x4plus.pth",
        "min_size_bytes": 60 * 1024 * 1024,  # ~67 MB
    },
    "wav2lip_gan.pth": {
        "url": "https://huggingface.co/numz/wav2lip_gan/resolve/main/wav2lip_gan.pth",
        "category": "Lip-Sync Engine",
        "dest": "models/wav2lip_gan.pth",
        "min_size_bytes": 400 * 1024 * 1024,  # ~435 MB
    },
    "wav2lip.pth": {
        "url": "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip.pth",
        "category": "Lip-Sync Engine (Fallback)",
        "dest": "models/wav2lip.pth",
        "min_size_bytes": 400 * 1024 * 1024,  # ~435 MB
    },
    "buffalo_l.zip": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "category": "Character Face Recognition (InsightFace)",
        "dest": "models/buffalo_l.zip",
        "min_size_bytes": 200 * 1024 * 1024,  # ~280 MB
        "extract_to": "models/buffalo_l",
    },
}

# HuggingFace Repositories required for full GPU execution
HUGGINGFACE_MODELS: Dict[str, Dict[str, str]] = {
    "InstantX/SDXL-InstantID": {
        "category": "Character Studio (InstantID SDXL)",
        "type": "huggingface_repo",
    },
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {
        "category": "Video Engine (Wan 2.1 1.3B DiT)",
        "type": "huggingface_repo",
    },
    "Lightricks/LTX-Video": {
        "category": "Video Engine (LTX-Video DiT Fallback)",
        "type": "huggingface_repo",
    },
    "KwaiVGI/LivePortrait": {
        "category": "Lip-Sync & Expressive Portrait Engine",
        "type": "huggingface_repo",
    },
}


class DownloadProgressTracker:
    """Tracks byte streaming progress and invokes user progress callbacks."""
    def __init__(self, filename: str, callback: Optional[Callable[[str, float, str], None]] = None):
        self.filename = filename
        self.callback = callback
        self.total_size = 0
        self.downloaded = 0
        self.start_time = time.time()
        self.last_update = 0.0

    def hook(self, block_num: int, block_size: int, total_size: int) -> None:
        self.total_size = total_size
        self.downloaded = block_num * block_size
        current_time = time.time()
        
        # Throttle progress updates to ~4 times per second
        if current_time - self.last_update > 0.25 or (total_size > 0 and self.downloaded >= total_size):
            self.last_update = current_time
            if total_size > 0:
                percent = min(100.0, (self.downloaded / total_size) * 100.0)
                mb_down = self.downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                status_text = f"{self.filename}: {mb_down:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)"
            else:
                mb_down = self.downloaded / (1024 * 1024)
                status_text = f"{self.filename}: {mb_down:.1f}MB downloaded"
                percent = 50.0

            if self.callback:
                self.callback(self.filename, percent, status_text)


def check_model_status() -> List[Dict[str, Any]]:
    """Returns a list describing the presence and size of each model."""
    status_list = []
    
    # 1. Direct Checkpoints
    for name, info in CHECKPOINT_REGISTRY.items():
        dest = Path(info["dest"])
        exists = dest.exists()
        size_mb = (dest.stat().st_size / (1024 * 1024)) if exists else 0.0
        min_size = info.get("min_size_bytes", 1024) / (1024 * 1024)
        is_valid = exists and (size_mb >= min_size * 0.8)
        
        status_list.append({
            "name": name,
            "category": info["category"],
            "dest": str(dest),
            "status": "Ready" if is_valid else ("Missing" if not exists else "Incomplete"),
            "size_mb": round(size_mb, 1),
            "is_valid": is_valid,
            "type": "checkpoint",
        })

    # 2. Hugging Face Models
    for repo_id, info in HUGGINGFACE_MODELS.items():
        status_list.append({
            "name": repo_id,
            "category": info["category"],
            "dest": f"huggingface_hub/{repo_id}",
            "status": "HF Hub (Auto-cached)",
            "size_mb": 0.0,
            "is_valid": True,
            "type": "hf_repo",
        })

    return status_list


def download_file(
    url: str,
    output_path: str,
    min_size_bytes: int = 1024,
    extract_to: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float, str], None]] = None,
) -> bool:
    """Downloads a file from url to output_path with progress callback and extraction support."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if valid file already exists
    if out_file.exists() and out_file.stat().st_size >= min_size_bytes:
        logger.info(f"Model already present: {out_file.name} ({out_file.stat().st_size / (1024*1024):.1f} MB)")
        if extract_to and not Path(extract_to).exists():
            _extract_archive(out_file, extract_to)
        return True

    filename = out_file.name
    logger.info(f"Starting download: {filename} from {url} ...")
    
    tracker = DownloadProgressTracker(filename, callback=progress_callback)
    
    # Download with custom user agent and timeout
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CineFlow-AI/1.0")]
    urllib.request.install_opener(opener)
    
    try:
        urllib.request.urlretrieve(url, filename=str(out_file), reporthook=tracker.hook)
        logger.info(f"Successfully downloaded: {filename}")
        
        if extract_to:
            _extract_archive(out_file, extract_to)
        return True
    except Exception as e:
        logger.warning(f"Download failed for {filename}: {e}. High-order fallback will be active.")
        if out_file.exists() and out_file.stat().st_size < min_size_bytes:
            try:
                out_file.unlink()
            except OSError:
                pass
        return False


def _extract_archive(archive_path: Path, target_dir: str) -> None:
    """Extracts zip archive to target directory."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(str(archive_path), "r") as zip_ref:
            zip_ref.extractall(str(target))
        logger.info(f"Extracted {archive_path.name} to {target_dir}")
    except Exception as e:
        logger.error(f"Failed to extract {archive_path.name}: {e}")


def download_all_models(
    include_optional: bool = False,
    progress_callback: Optional[Callable[[str, float, str], None]] = None,
) -> Dict[str, bool]:
    """Downloads all essential model checkpoints."""
    results = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download direct checkpoints
    for name, info in CHECKPOINT_REGISTRY.items():
        if not include_optional and "Fallback" in info.get("category", ""):
            continue
        success = download_file(
            url=info["url"],
            output_path=info["dest"],
            min_size_bytes=info.get("min_size_bytes", 1024),
            extract_to=info.get("extract_to"),
            progress_callback=progress_callback,
        )
        results[name] = success

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  CineFlow-AI Model Downloader Diagnostic")
    print("=" * 60)
    models_status = check_model_status()
    for item in models_status:
        print(f"[{item['status']:<10}] {item['name']:<30} ({item['category']}) - {item['size_mb']} MB")
    print("\nDownloading essential weights...")
    download_all_models(include_optional=False)
    print("\nDone.")
