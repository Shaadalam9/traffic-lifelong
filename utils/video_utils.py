from __future__ import annotations

import shutil
from pathlib import Path

import cv2


def get_video_metadata(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


def copy_video(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_preview(src: Path, dst: Path, seconds: float = 20.0) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not open video for preview: {src}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))
    max_frames = int(seconds * fps)
    written = 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()


def sample_frame(src: Path, ratio: float, dst: Path, jpeg_quality: int = 95) -> bool:
    cap = cv2.VideoCapture(str(src))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_idx = max(0, min(frame_count - 1, int(frame_count * ratio))) if frame_count > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    return True
