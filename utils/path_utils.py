from __future__ import annotations

from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def list_video_files(root: Path, recursive: bool = True) -> list[Path]:
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    files = [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    return sorted(files)


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in path.stem.lower())
