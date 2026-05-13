#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from utils.base import PipelineStage

# ============================================================
# Edit only these values
# ============================================================
INPUT_PATH = "/Volumes/Alam/Ubicomb"
PROJECT_ROOT = "."
RECURSIVE = True

# Main outputs
INVENTORY_CSV_PATH = "metadata/video_inventory.csv"
TIME_BOUNDS_CSV_PATH = "metadata/video_time_bounds.csv"
CLIP_MANIFEST_CSV_PATH = "metadata/clip_manifest.csv"
SCENE_FRAME_MANIFEST_CSV_PATH = "metadata/scene_frame_manifest.csv"

# Output folders
STANDARDIZED_VIDEO_DIR = "standardized_videos"
PREVIEW_DIR = "metadata/previews"
SCENE_FRAME_DIR = "frames_for_scene_setup"

# Pipeline toggles
RUN_OCR_TIME_BOUNDS = True
RUN_VIDEO_INVENTORY = True
RUN_STANDARDIZE_AND_SPLIT = True
RUN_PREVIEW_CLIPS = True
RUN_SCENE_FRAME_SAMPLING = True

# Clip and preview settings
CLIP_DURATION_SECONDS = 1800
PREVIEW_DURATION_SECONDS = 20
OVERWRITE_EXISTING_OUTPUTS = False
KEEP_AUDIO = False

# Standardisation settings
TARGET_CONTAINER_SUFFIX = ".mp4"
TARGET_VIDEO_CODEC = "libx264"
TARGET_PRESET = "medium"
TARGET_CRF = 18
TARGET_FPS = 10.0
TARGET_WIDTH = None
TARGET_HEIGHT = None
PIXEL_FORMAT = "yuv420p"

# Deduplication and safe output writing
DEDUPE_OVERLAP_TOLERANCE_SECONDS = 1.0
MIN_OUTPUT_SEGMENT_SECONDS = 1.0
SKIP_VIDEOS_WITHOUT_TRUSTED_TIME_RANGE = True
CLEAN_STALE_TEMP_OUTPUTS_ON_START = True
TEMP_OUTPUT_SUFFIX = ".tmp"

# Scene frame sampling
SCENE_FRAME_SAMPLE_RATIOS = [0.05, 0.25, 0.50, 0.75, 0.95]
SCENE_FRAME_JPEG_QUALITY = 95

# Timestamp area in the top left corner
CROP_X = 0.015
CROP_Y = 0.020
CROP_W = 0.310
CROP_H = 0.080

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpeg", ".mpg",
    ".wmv", ".asf", ".ts", ".mts", ".m2ts", ".3gp", ".3g2", ".f4v",
    ".flv", ".dv", ".ogv", ".vob", ".mxf", ".dav", ".h264", ".h265",
    ".hevc",
}

FRAME_OFFSETS = [0, 2, 5, 10]
THRESHOLDS = [140, 170, 200, 225]
OCR_TIMEOUT_SECONDS = 1.5
TESSERACT_CONFIGS = [
    "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:/- APMapm",
    "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:/- APMapm",
]
TIME_ALIGNMENT_TOLERANCE_SECONDS = 5.0


# ============================================================
# Utilities
# ============================================================
def ensure_parent_folder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def path_to_posix(path: Path) -> str:
    return path.as_posix()


def bool_to_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "yes" if value else "no"


def safe_float(value) -> float | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value) -> int | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def format_datetime(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def parse_iso_datetime(text: str) -> datetime | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def seconds_to_hms(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def slugify_for_id(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text)
    cleaned = cleaned.strip("_")
    return cleaned.lower() or "item"


def run_subprocess(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def now_temp_token() -> str:
    return f"{os.getpid()}_{uuid.uuid4().hex}"


def load_existing_csv_rows(csv_path: Path, key_field: str = "video_id") -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}

    rows: dict[str, dict[str, str]] = {}
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                key = str(row.get(key_field, "") or "").strip()
                if not key:
                    continue
                rows[key] = {str(k): "" if v is None else str(v) for k, v in row.items()}
    except OSError:
        return {}
    return rows


def choose_manual_or_auto_datetime(row: dict[str, str], manual_key: str, auto_key: str) -> tuple[datetime | None, str]:
    manual_value = parse_iso_datetime(row.get(manual_key, ""))
    if manual_value is not None:
        return manual_value, manual_key

    auto_value = parse_iso_datetime(row.get(auto_key, ""))
    if auto_value is not None:
        return auto_value, auto_key

    return None, ""


def apply_existing_manual_values(
    inventory_rows: list[dict[str, str]],
    existing_time_bounds_rows: dict[str, dict[str, str]],
    existing_inventory_rows: dict[str, dict[str, str]],
) -> None:
    manual_fields = [
        "checked_start_time",
        "checked_end_time",
        "location_overlay_visible_manual",
        "camera_view_changes_manual",
        "approximate_lighting_manual",
        "notes_manual",
    ]

    for row in inventory_rows:
        existing_sources = [
            existing_inventory_rows.get(row["video_id"], {}),
            existing_time_bounds_rows.get(row["video_id"], {}),
        ]
        for field in manual_fields:
            for existing in existing_sources:
                value = str(existing.get(field, "") or "").strip()
                if value:
                    row[field] = value
                    break


# ============================================================
# Safe temporary output handling
# ============================================================
def is_temporary_output_path(path: Path) -> bool:
    name = path.name
    return TEMP_OUTPUT_SUFFIX in name


def cleanup_stale_temp_outputs(project_root: Path) -> int:
    removed = 0
    candidate_roots = [
        project_root / STANDARDIZED_VIDEO_DIR,
        project_root / PREVIEW_DIR,
    ]

    for root in candidate_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if not is_temporary_output_path(path):
                continue
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def make_temp_output_path(final_path: Path) -> Path:
    token = now_temp_token()
    suffix = final_path.suffix
    stem = final_path.stem
    if suffix:
        temp_name = f"{stem}{TEMP_OUTPUT_SUFFIX}.{token}{suffix}"
    else:
        temp_name = f"{final_path.name}{TEMP_OUTPUT_SUFFIX}.{token}"
    return final_path.with_name(temp_name)


def cleanup_temporary_variants_for_target(final_path: Path) -> None:
    parent = final_path.parent
    if not parent.exists():
        return

    suffix = final_path.suffix
    stem = final_path.stem
    patterns = [f"{final_path.name}{TEMP_OUTPUT_SUFFIX}*"]
    if suffix:
        patterns.append(f"{stem}{TEMP_OUTPUT_SUFFIX}.*{suffix}")

    seen: set[Path] = set()
    for pattern in patterns:
        for path in parent.glob(pattern):
            if path in seen:
                continue
            seen.add(path)
            if not path.is_file():
                continue
            try:
                path.unlink()
            except OSError:
                continue


# ============================================================
# Video discovery
# ============================================================
def list_video_files(folder: Path, recursive: bool) -> list[Path]:
    if not folder.exists():
        return []
    if folder.is_file():
        return [folder] if folder.suffix.lower() in VIDEO_EXTENSIONS else []

    pattern = "**/*" if recursive else "*"
    files: list[Path] = []
    for path in folder.glob(pattern):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(path)
    return sorted(files)


def make_video_id(video_path: Path, input_root: Path) -> str:
    try:
        relative = video_path.relative_to(input_root)
    except ValueError:
        relative = video_path.name
        return slugify_for_id(str(relative))

    stemmed = relative.with_suffix("")
    return slugify_for_id(path_to_posix(stemmed))


# ============================================================
# OCR for burned in timestamp
# ============================================================
def normalise_ocr_text(text: str) -> str:
    cleaned = text.strip().replace("\n", " ").replace("\r", " ")
    cleaned = cleaned.replace("—", "-").replace("–", "-").replace("/", "-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.translate(str.maketrans({
        "O": "0",
        "o": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        ",": ":",
        ";": ":",
    }))
    cleaned = re.sub(r"(\d{1,2}:\d{2}:\d{2})(\d{1,2}-\d{1,2}-\d{2,4})", r"\1 \2", cleaned)
    cleaned = re.sub(r"[^0-9:\- APMapm]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_datetime_from_pattern(cleaned_text: str) -> datetime | None:
    patterns = [
        r"(?P<time>\d{1,2}:\d{2}:\d{2})\s+(?P<date>\d{1,2}-\d{1,2}-\d{4})(?:\s*(?P<ampm>AM|PM|am|pm))?",
        r"(?P<date>\d{1,2}-\d{1,2}-\d{4})\s+(?P<time>\d{1,2}:\d{2}:\d{2})(?:\s*(?P<ampm>AM|PM|am|pm))?",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned_text)
        if match is None:
            continue

        date_part = match.group("date")
        time_part = match.group("time")
        ampm = match.groupdict().get("ampm")

        month, day, year = date_part.split("-")
        assembled = f"{month.zfill(2)}-{day.zfill(2)}-{year} {time_part}"

        formats = ["%m-%d-%Y %H:%M:%S", "%m-%d-%Y %I:%M:%S"]
        if ampm:
            assembled = f"{assembled} {ampm.upper()}"
            formats = ["%m-%d-%Y %I:%M:%S %p"]

        for fmt in formats:
            try:
                parsed = datetime.strptime(assembled, fmt)
            except ValueError:
                continue
            if 2000 <= parsed.year <= 2099:
                return parsed
    return None


def parse_datetime_from_digits(cleaned_text: str) -> datetime | None:
    digits = re.sub(r"\D", "", cleaned_text)
    if len(digits) < 14:
        return None

    seen = set()
    for index in range(len(digits) - 13):
        token = digits[index:index + 14]
        if token in seen:
            continue
        seen.add(token)

        month = int(token[0:2])
        day = int(token[2:4])
        year = int(token[4:8])
        hour = int(token[8:10])
        minute = int(token[10:12])
        second = int(token[12:14])

        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        if not (2000 <= year <= 2099):
            continue
        if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
            continue

        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            continue
    return None


def parse_overlay_datetime(raw_text: str) -> datetime | None:
    cleaned = normalise_ocr_text(raw_text)
    parsed = parse_datetime_from_pattern(cleaned)
    if parsed is not None:
        return parsed
    return parse_datetime_from_digits(cleaned)


def crop_overlay(frame):
    height, width = frame.shape[:2]
    x1 = max(0, int(round(width * CROP_X)))
    y1 = max(0, int(round(height * CROP_Y)))
    x2 = min(width, int(round(width * (CROP_X + CROP_W))))
    y2 = min(height, int(round(height * (CROP_Y + CROP_H))))

    if x2 <= x1 or y2 <= y1:
        return frame.copy()
    return frame[y1:y2, x1:x2].copy()


def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    enlarged = cv2.GaussianBlur(enlarged, (3, 3), 0)

    outputs = [enlarged]
    for threshold in THRESHOLDS:
        binary = cv2.threshold(enlarged, threshold, 255, cv2.THRESH_BINARY)[1]
        padded = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        outputs.append(padded)
    return outputs


def extract_timestamp_from_frame(frame) -> datetime | None:
    crop = crop_overlay(frame)

    for processed in preprocess_crop(crop):
        for tesseract_cfg in TESSERACT_CONFIGS:
            try:
                raw_text = pytesseract.image_to_string(
                    processed,
                    config=tesseract_cfg,
                    timeout=OCR_TIMEOUT_SECONDS,
                )
            except Exception:
                continue

            parsed = parse_overlay_datetime(raw_text)
            if parsed is not None:
                return parsed
    return None


def decode_image_bytes(image_bytes: bytes):
    if not image_bytes:
        return None
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    if array.size == 0:
        return None
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return frame


def extract_frame_with_ffmpeg(video_path: Path, timestamp_seconds: float):
    if not ffmpeg_exists():
        return None

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-err_detect",
        "ignore_err",
        "-ss",
        format_float(timestamp_seconds, 3),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        return None
    return decode_image_bytes(result.stdout)


def read_frame(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return frame if ok else None


def read_frame_robust(
    video_path: Path,
    cap: cv2.VideoCapture,
    frame_index: int,
    fps: float,
):
    timestamp_seconds = frame_index / fps if fps > 0 else 0.0
    frame = extract_frame_with_ffmpeg(video_path, timestamp_seconds)
    if frame is not None:
        return frame
    return read_frame(cap, frame_index)


def extract_edge_timestamp(video_path: Path, cap: cv2.VideoCapture, frame_count: int,
                           fps: float, edge: str) -> datetime | None:
    for offset in FRAME_OFFSETS:
        frame_index = offset if edge == "start" else max(0, frame_count - 1 - offset)
        frame = read_frame_robust(video_path, cap, frame_index, fps)
        if frame is None:
            continue

        parsed = extract_timestamp_from_frame(frame)
        if parsed is not None:
            return parsed
    return None


def extract_video_bounds(video_path: Path) -> dict[str, str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "ocr_start_time": "",
            "ocr_end_time": "",
            "ocr_status": "video_open_failed",
            "timestamp_overlay_visible_auto": "",
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    if frame_count <= 0:
        cap.release()
        return {
            "ocr_start_time": "",
            "ocr_end_time": "",
            "ocr_status": "empty_video",
            "timestamp_overlay_visible_auto": "",
        }

    start_dt = extract_edge_timestamp(video_path, cap, frame_count, fps, "start")
    end_dt = extract_edge_timestamp(video_path, cap, frame_count, fps, "end")
    cap.release()

    status = "ok"
    if start_dt is None or end_dt is None:
        status = "needs_review"
    elif end_dt < start_dt:
        status = "needs_review"

    overlay_visible = start_dt is not None or end_dt is not None
    return {
        "ocr_start_time": format_datetime(start_dt),
        "ocr_end_time": format_datetime(end_dt),
        "ocr_status": status,
        "timestamp_overlay_visible_auto": bool_to_text(overlay_visible),
    }


# ============================================================
# Metadata and inventory
# ============================================================
def parse_fraction(text: str | None) -> float | None:
    if not text:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    if "/" in cleaned:
        numerator, denominator = cleaned.split("/", 1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            return float(numerator) / denominator_value
        except ValueError:
            return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def ffprobe_video_metadata(video_path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = run_subprocess(command)
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def cv2_video_metadata(video_path: Path) -> dict[str, object]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0
    duration_seconds = frame_count / fps if fps and frame_count else None
    cap.release()

    return {
        "duration_seconds": duration_seconds,
        "fps": fps if fps else None,
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "codec_name": "",
        "container_format": "",
    }


def get_video_metadata(video_path: Path) -> dict[str, object]:
    if ffmpeg_exists():
        probe = ffprobe_video_metadata(video_path)
    else:
        probe = {}

    video_stream = {}
    if probe:
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

    format_info = probe.get("format", {}) if probe else {}

    duration_seconds = safe_float(format_info.get("duration"))
    fps = parse_fraction(video_stream.get("avg_frame_rate")) or parse_fraction(video_stream.get("r_frame_rate"))
    width = safe_int(video_stream.get("width"))
    height = safe_int(video_stream.get("height"))
    codec_name = str(video_stream.get("codec_name", "") or "")
    container_format = str(format_info.get("format_name", "") or "")

    if duration_seconds is None or fps is None or width is None or height is None:
        fallback = cv2_video_metadata(video_path)
        duration_seconds = duration_seconds if duration_seconds is not None else fallback.get("duration_seconds")
        fps = fps if fps is not None else fallback.get("fps")
        width = width if width is not None else fallback.get("width")
        height = height if height is not None else fallback.get("height")
        if not codec_name:
            codec_name = str(fallback.get("codec_name", "") or "")
        if not container_format:
            container_format = str(fallback.get("container_format", "") or "")

    return {
        "duration_seconds": duration_seconds,
        "fps": fps,
        "width": width,
        "height": height,
        "codec_name": codec_name,
        "container_format": container_format,
    }


def build_inventory_row(video_path: Path, input_root: Path) -> dict[str, str]:
    metadata = get_video_metadata(video_path)
    ocr_bounds = extract_video_bounds(video_path) if RUN_OCR_TIME_BOUNDS else {
        "ocr_start_time": "",
        "ocr_end_time": "",
        "ocr_status": "",
        "timestamp_overlay_visible_auto": "",
    }

    try:
        relative_path = video_path.relative_to(input_root)
    except ValueError:
        relative_path = video_path

    video_id = make_video_id(video_path, input_root)
    file_size_bytes = video_path.stat().st_size
    width = metadata.get("width")
    height = metadata.get("height")
    resolution = f"{width}x{height}" if width and height else ""

    row = {
        "video_id": video_id,
        "video_name": video_path.name,
        "source_path": path_to_posix(video_path.resolve()),
        "relative_source_path": path_to_posix(relative_path),
        "parent_folder": path_to_posix(relative_path.parent) if relative_path.parent != Path(".") else "",
        "file_size_bytes": str(file_size_bytes),
        "duration_seconds": format_float(safe_float(metadata.get("duration_seconds")), 3),
        "duration_hms": seconds_to_hms(safe_float(metadata.get("duration_seconds"))),
        "fps": format_float(safe_float(metadata.get("fps")), 3),
        "width": str(width or ""),
        "height": str(height or ""),
        "resolution": resolution,
        "codec_name": str(metadata.get("codec_name", "") or ""),
        "container_format": str(metadata.get("container_format", "") or ""),
        "ocr_start_time": ocr_bounds.get("ocr_start_time", ""),
        "ocr_end_time": ocr_bounds.get("ocr_end_time", ""),
        "checked_start_time": "",
        "checked_end_time": "",
        "effective_start_time": "",
        "effective_end_time": "",
        "ocr_status": ocr_bounds.get("ocr_status", ""),
        "timestamp_overlay_visible_auto": ocr_bounds.get("timestamp_overlay_visible_auto", ""),
        "trusted_interval_start": "",
        "trusted_interval_end": "",
        "trusted_interval_source": "",
        "trusted_interval_duration_sec": "",
        "dedupe_status": "pending",
        "selected_for_output": "",
        "dedupe_notes": "",
        "location_overlay_visible_manual": "",
        "camera_view_changes_manual": "",
        "approximate_lighting_manual": "",
        "notes_manual": "",
    }
    return row


# ============================================================
# Trusted time ranges and duplicate coverage planning
# ============================================================
def infer_trusted_video_interval(inventory_row: dict[str, str]) -> tuple[datetime | None, datetime | None,
                                                                         str, datetime | None, datetime | None]:
    duration_seconds = safe_float(inventory_row.get("duration_seconds"))
    effective_start_time, start_source = choose_manual_or_auto_datetime(
        inventory_row,
        manual_key="checked_start_time",
        auto_key="ocr_start_time",
    )
    effective_end_time, end_source = choose_manual_or_auto_datetime(
        inventory_row,
        manual_key="checked_end_time",
        auto_key="ocr_end_time",
    )

    if effective_start_time is not None and effective_end_time is not None and effective_end_time >= effective_start_time:
        source_name = f"{start_source}_{end_source}".strip("_") or "manual_or_ocr_start_end"
        return effective_start_time, effective_end_time, source_name, effective_start_time, effective_end_time
    if effective_start_time is not None and duration_seconds is not None and duration_seconds > 0:
        source_name = f"{start_source}_plus_duration".strip("_") or "manual_or_ocr_start_plus_duration"
        return effective_start_time, effective_start_time + timedelta(seconds=duration_seconds),
    source_name, effective_start_time, effective_end_time

    if effective_end_time is not None and duration_seconds is not None and duration_seconds > 0:
        source_name = f"{end_source}_minus_duration".strip("_") or "manual_or_ocr_end_minus_duration"
        return effective_end_time - timedelta(seconds=duration_seconds), effective_end_time, source_name,
    effective_start_time, effective_end_time
    return None, None, "insufficient_time_mapping", effective_start_time, effective_end_time


def resolution_area(row: dict[str, str]) -> int:
    width = safe_int(row.get("width")) or 0
    height = safe_int(row.get("height")) or 0
    return width * height


def interval_duration_seconds(start_time: datetime, end_time: datetime) -> float:
    return max(0.0, (end_time - start_time).total_seconds())


def sort_key_for_coverage(row: dict[str, str]) -> tuple:
    start_time = parse_iso_datetime(row.get("trusted_interval_start", ""))
    end_time = parse_iso_datetime(row.get("trusted_interval_end", ""))
    duration_seconds = 0.0
    if start_time is not None and end_time is not None:
        duration_seconds = interval_duration_seconds(start_time, end_time)
    return (
        start_time or datetime.max,
        -duration_seconds,
        -resolution_area(row),
        row.get("relative_source_path", ""),
    )


Interval = tuple[datetime, datetime]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []

    tolerance = timedelta(seconds=DEDUPE_OVERLAP_TOLERANCE_SECONDS)
    ordered = sorted(intervals, key=lambda item: item[0])
    merged: list[Interval] = [ordered[0]]

    for start_time, end_time in ordered[1:]:
        last_start, last_end = merged[-1]
        if start_time <= last_end + tolerance:
            merged[-1] = (last_start, max(last_end, end_time))
        else:
            merged.append((start_time, end_time))
    return merged


def subtract_covered_intervals(base_interval: Interval, covered_intervals: list[Interval]) -> list[Interval]:
    tolerance = timedelta(seconds=DEDUPE_OVERLAP_TOLERANCE_SECONDS)
    min_segment = timedelta(seconds=MIN_OUTPUT_SEGMENT_SECONDS)
    remaining: list[Interval] = [base_interval]

    for covered_start, covered_end in covered_intervals:
        updated: list[Interval] = []
        for segment_start, segment_end in remaining:
            if covered_end <= segment_start + tolerance or covered_start >= segment_end - tolerance:
                updated.append((segment_start, segment_end))
                continue

            left_end = min(segment_end, covered_start)
            right_start = max(segment_start, covered_end)

            if left_end - segment_start >= min_segment:
                updated.append((segment_start, left_end))
            if segment_end - right_start >= min_segment:
                updated.append((right_start, segment_end))
        remaining = updated
        if not remaining:
            break

    return remaining


def annotate_inventory_with_trusted_intervals(inventory_rows: list[dict[str, str]]) -> None:
    for row in inventory_rows:
        start_time, end_time, source_name, effective_start_time, effective_end_time = infer_trusted_video_interval(row)
        row["effective_start_time"] = format_datetime(effective_start_time)
        row["effective_end_time"] = format_datetime(effective_end_time)
        row["trusted_interval_start"] = format_datetime(start_time)
        row["trusted_interval_end"] = format_datetime(end_time)
        row["trusted_interval_source"] = source_name
        if start_time is not None and end_time is not None:
            row["trusted_interval_duration_sec"] = format_float(interval_duration_seconds(start_time, end_time), 3)
        else:
            row["trusted_interval_duration_sec"] = ""


SegmentPlan = dict[str, object]


def build_deduplicated_segment_plan(inventory_rows: list[dict[str, str]]) -> list[SegmentPlan]:
    annotate_inventory_with_trusted_intervals(inventory_rows)

    for row in inventory_rows:
        row["dedupe_status"] = "pending"
        row["selected_for_output"] = "no"
        row["dedupe_notes"] = ""

    covered_intervals: list[Interval] = []
    segment_plan: list[SegmentPlan] = []

    rows_with_time = []
    rows_without_time = []
    for row in inventory_rows:
        start_time = parse_iso_datetime(row.get("trusted_interval_start", ""))
        end_time = parse_iso_datetime(row.get("trusted_interval_end", ""))
        if start_time is None or end_time is None or end_time <= start_time:
            rows_without_time.append(row)
        else:
            rows_with_time.append(row)

    rows_with_time.sort(key=sort_key_for_coverage)

    for row in rows_with_time:
        start_time = parse_iso_datetime(row["trusted_interval_start"])
        end_time = parse_iso_datetime(row["trusted_interval_end"])
        assert start_time is not None
        assert end_time is not None

        base_interval = (start_time, end_time)
        uncovered_intervals = subtract_covered_intervals(base_interval, covered_intervals)

        if not uncovered_intervals:
            row["dedupe_status"] = "fully_covered_duplicate"
            row["selected_for_output"] = "no"
            row["dedupe_notes"] = "All wall clock coverage already present in earlier selected footage."
            continue

        if len(uncovered_intervals) == 1 and uncovered_intervals[0] == base_interval:
            row["dedupe_status"] = "unique_full_interval"
            row["dedupe_notes"] = "Entire trusted interval kept for output."
        else:
            row["dedupe_status"] = "partial_overlap_trimmed"
            row["dedupe_notes"] = "Only uncovered wall clock ranges kept to avoid duplicate footage."

        row["selected_for_output"] = "yes"

        for segment_index, (segment_start, segment_end) in enumerate(uncovered_intervals, start=1):
            segment_plan.append({
                "video_id": row["video_id"],
                "source_path": row["source_path"],
                "relative_source_path": row["relative_source_path"],
                "segment_index": segment_index,
                "segment_start_time": segment_start,
                "segment_end_time": segment_end,
                "segment_duration_sec": interval_duration_seconds(segment_start, segment_end),
                "trusted_video_start": start_time,
                "trusted_video_end": end_time,
                "dedupe_status": row["dedupe_status"],
            })

        covered_intervals.extend(uncovered_intervals)
        covered_intervals = merge_intervals(covered_intervals)

    for row in rows_without_time:
        if SKIP_VIDEOS_WITHOUT_TRUSTED_TIME_RANGE:
            row["dedupe_status"] = "skipped_no_trusted_time_range"
            row["selected_for_output"] = "no"
            row["dedupe_notes"] = "Skipped so output clips can remain guaranteed non duplicate by timestamp."
        else:
            row["dedupe_status"] = "kept_without_trusted_time_range"
            row["selected_for_output"] = "yes"
            row["dedupe_notes"] = "Included without timestamp dedupe guarantee."
            duration_seconds = safe_float(row.get("duration_seconds"))
            if duration_seconds is None or duration_seconds <= 0:
                continue
            segment_plan.append({
                "video_id": row["video_id"],
                "source_path": row["source_path"],
                "relative_source_path": row["relative_source_path"],
                "segment_index": 1,
                "segment_start_time": None,
                "segment_end_time": None,
                "segment_duration_sec": duration_seconds,
                "trusted_video_start": None,
                "trusted_video_end": None,
                "dedupe_status": row["dedupe_status"],
            })

    segment_plan.sort(
        key=lambda item: (
            item["segment_start_time"] if item["segment_start_time"] is not None else datetime.max,
            item["relative_source_path"],
            item["segment_index"],
        )
    )
    return segment_plan


# ============================================================
# Standardisation and clip manifest
# ============================================================
def build_scale_filter() -> str:
    if TARGET_WIDTH is None and TARGET_HEIGHT is None:
        return ""
    if TARGET_WIDTH is not None and TARGET_HEIGHT is not None:
        return f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}"
    if TARGET_WIDTH is not None:
        return f"scale={TARGET_WIDTH}:-2"
    return f"scale=-2:{TARGET_HEIGHT}"


def build_ffmpeg_video_filters() -> str:
    filters: list[str] = []
    if TARGET_FPS:
        filters.append(f"fps={TARGET_FPS}")
    scale_filter = build_scale_filter()
    if scale_filter:
        filters.append(scale_filter)
    return ",".join(filters)


def ffmpeg_common_output_args() -> list[str]:
    args = [
        "-map", "0:v:0",
        "-c:v", TARGET_VIDEO_CODEC,
        "-preset", TARGET_PRESET,
        "-crf", str(TARGET_CRF),
        "-pix_fmt", PIXEL_FORMAT,
        "-movflags", "+faststart",
    ]
    video_filters = build_ffmpeg_video_filters()
    if video_filters:
        args.extend(["-vf", video_filters])
    if KEEP_AUDIO:
        args.extend(["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"])
    else:
        args.append("-an")
    return args


def ffmpeg_muxer_args_for_path(output_path: Path) -> list[str]:
    suffix = output_path.suffix.lower()
    if suffix == ".mp4":
        return ["-f", "mp4"]
    if suffix == ".mov":
        return ["-f", "mov"]
    if suffix == ".mkv":
        return ["-f", "matroska"]
    if suffix == ".avi":
        return ["-f", "avi"]
    return []


def run_ffmpeg_to_temp(command_prefix: list[str], final_path: Path) -> tuple[bool, str]:
    ensure_parent_folder(final_path)
    cleanup_temporary_variants_for_target(final_path)
    temp_path = make_temp_output_path(final_path)

    command = [
        *command_prefix,
        *ffmpeg_muxer_args_for_path(final_path),
        "-y",
        str(temp_path),
    ]
    result = run_subprocess(command)
    if result.returncode != 0:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        return False, result.stderr.strip() or "ffmpeg_failed"

    if not temp_path.exists():
        return False, "ffmpeg_missing_temp_output"

    try:
        temp_path.replace(final_path)
    except OSError as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        return False, f"rename_failed:{exc}"

    return True, "ok"


def make_preview_clip(source_path: Path, preview_path: Path, start_seconds: float,
                      duration_seconds: float) -> tuple[bool, str]:
    if preview_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
        return True, "existing"

    duration_seconds = max(0.0, duration_seconds)
    command_prefix = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", format_float(start_seconds, 3),
        "-i", str(source_path),
        "-t", format_float(duration_seconds, 3),
        *ffmpeg_common_output_args(),
    ]
    return run_ffmpeg_to_temp(command_prefix, preview_path)


def standardize_clip(source_path: Path, output_path: Path, start_seconds: float,
                     duration_seconds: float) -> tuple[bool, str]:
    if output_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
        return True, "existing"

    duration_seconds = max(0.0, duration_seconds)
    command_prefix = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", format_float(start_seconds, 3),
        "-i", str(source_path),
        "-t", format_float(duration_seconds, 3),
        *ffmpeg_common_output_args(),
    ]
    return run_ffmpeg_to_temp(command_prefix, output_path)


def infer_absolute_clip_time(
    ocr_start_time: datetime | None,
    ocr_end_time: datetime | None,
    source_duration_seconds: float | None,
    clip_seconds: float,
) -> datetime | None:
    if ocr_start_time is not None:
        return ocr_start_time + timedelta(seconds=clip_seconds)
    if ocr_end_time is not None and source_duration_seconds is not None:
        delta = source_duration_seconds - clip_seconds
        return ocr_end_time - timedelta(seconds=delta)
    return None


def determine_time_alignment_status(
    ocr_start_time: datetime | None,
    ocr_end_time: datetime | None,
    source_duration_seconds: float | None,
) -> str:
    if ocr_start_time is None and ocr_end_time is None:
        return "ocr_missing"
    if ocr_start_time is None or ocr_end_time is None:
        return "ocr_partial"
    if source_duration_seconds is None:
        return "ok"

    observed = (ocr_end_time - ocr_start_time).total_seconds()
    difference = abs(observed - source_duration_seconds)
    if difference <= max(TIME_ALIGNMENT_TOLERANCE_SECONDS, source_duration_seconds * 0.01):
        return "ok"
    return "duration_mismatch"


def build_clip_rows(
    inventory_lookup: dict[str, dict[str, str]],
    segment_plan: list[SegmentPlan],
    project_root: Path,
) -> list[dict[str, str]]:
    clip_rows: list[dict[str, str]] = []

    for segment in segment_plan:
        video_id = str(segment["video_id"])
        inventory_row = inventory_lookup[video_id]
        source_duration_seconds = safe_float(inventory_row.get("duration_seconds"))
        if source_duration_seconds is None or source_duration_seconds <= 0:
            continue

        trusted_video_start = segment["trusted_video_start"]
        trusted_video_end = segment["trusted_video_end"]
        if trusted_video_start is not None and segment["segment_start_time"] is not None:
            segment_start_sec = max(0.0, (segment["segment_start_time"] - trusted_video_start).total_seconds())
            segment_end_sec = min(source_duration_seconds, (segment["segment_end_time"] - trusted_video_start).total_seconds())
        else:
            segment_start_sec = 0.0
            segment_end_sec = source_duration_seconds

        segment_duration_sec = max(0.0, segment_end_sec - segment_start_sec)
        if segment_duration_sec < MIN_OUTPUT_SEGMENT_SECONDS:
            continue

        effective_start_time = parse_iso_datetime(inventory_row.get("effective_start_time", ""))
        effective_end_time = parse_iso_datetime(inventory_row.get("effective_end_time", ""))
        time_alignment_status = determine_time_alignment_status(
            ocr_start_time=effective_start_time,
            ocr_end_time=effective_end_time,
            source_duration_seconds=source_duration_seconds,
        )

        chunk_count = max(1, math.ceil(segment_duration_sec / CLIP_DURATION_SECONDS))
        preview_start_sec = segment_start_sec
        preview_duration_sec = min(PREVIEW_DURATION_SECONDS, segment_duration_sec)
        preview_path = project_root / PREVIEW_DIR / f"{inventory_row['video_id']}{TARGET_CONTAINER_SUFFIX}"

        for chunk_index in range(chunk_count):
            chunk_start_sec = segment_start_sec + chunk_index * CLIP_DURATION_SECONDS
            chunk_end_sec = min(segment_end_sec, segment_start_sec + (chunk_index + 1) * CLIP_DURATION_SECONDS)
            chunk_duration_sec = max(0.0, chunk_end_sec - chunk_start_sec)
            if chunk_duration_sec < MIN_OUTPUT_SEGMENT_SECONDS:
                continue

            clip_id = (
                f"{inventory_row['video_id']}_seg_{int(segment['segment_index']):02d}"
                f"_clip_{chunk_index + 1:04d}"
            )
            standardized_path = project_root / STANDARDIZED_VIDEO_DIR / inventory_row["video_id"] / f"{clip_id}{TARGET_CONTAINER_SUFFIX}"

            clip_start_time = infer_absolute_clip_time(
                ocr_start_time=effective_start_time,
                ocr_end_time=effective_end_time,
                source_duration_seconds=source_duration_seconds,
                clip_seconds=chunk_start_sec,
            )
            clip_end_time = infer_absolute_clip_time(
                ocr_start_time=effective_start_time,
                ocr_end_time=effective_end_time,
                source_duration_seconds=source_duration_seconds,
                clip_seconds=chunk_end_sec,
            )

            clip_rows.append({
                "video_id": inventory_row["video_id"],
                "clip_id": clip_id,
                "clip_index": str(chunk_index + 1),
                "coverage_segment_index": str(segment["segment_index"]),
                "source_path": inventory_row["source_path"],
                "relative_source_path": inventory_row["relative_source_path"],
                "standardized_path": path_to_posix(standardized_path.resolve()),
                "relative_standardized_path": project_relative_or_absolute(
                    standardized_path,
                    project_root,
                ),
                "preview_path": path_to_posix(preview_path.resolve()),
                "segment_start_sec": format_float(segment_start_sec, 3),
                "segment_end_sec": format_float(segment_end_sec, 3),
                "segment_duration_sec": format_float(segment_duration_sec, 3),
                "segment_start_time": format_datetime(segment["segment_start_time"]),
                "segment_end_time": format_datetime(segment["segment_end_time"]),
                "clip_start_sec": format_float(chunk_start_sec, 3),
                "clip_end_sec": format_float(chunk_end_sec, 3),
                "clip_duration_sec": format_float(chunk_duration_sec, 3),
                "clip_start_time": format_datetime(clip_start_time),
                "clip_end_time": format_datetime(clip_end_time),
                "time_alignment_status": time_alignment_status,
                "source_fps": inventory_row.get("fps", ""),
                "target_fps": format_float(TARGET_FPS, 3) if TARGET_FPS else inventory_row.get("fps", ""),
                "source_width": inventory_row.get("width", ""),
                "source_height": inventory_row.get("height", ""),
                "target_width": str(TARGET_WIDTH or inventory_row.get("width", "")),
                "target_height": str(TARGET_HEIGHT or inventory_row.get("height", "")),
                "ocr_start_time": inventory_row.get("ocr_start_time", ""),
                "ocr_end_time": inventory_row.get("ocr_end_time", ""),
                "checked_start_time": inventory_row.get("checked_start_time", ""),
                "checked_end_time": inventory_row.get("checked_end_time", ""),
                "effective_start_time": inventory_row.get("effective_start_time", ""),
                "effective_end_time": inventory_row.get("effective_end_time", ""),
                "ocr_status": inventory_row.get("ocr_status", ""),
                "trusted_interval_start": inventory_row.get("trusted_interval_start", ""),
                "trusted_interval_end": inventory_row.get("trusted_interval_end", ""),
                "dedupe_status": inventory_row.get("dedupe_status", ""),
                "standardization_status": "pending",
                "standardization_error": "",
                "preview_status": "pending" if RUN_PREVIEW_CLIPS else "skipped",
                "preview_error": "",
                "preview_start_sec": format_float(preview_start_sec, 3),
                "preview_duration_sec": format_float(preview_duration_sec, 3),
            })

    clip_rows.sort(
        key=lambda item: (
            item["clip_start_time"] or "9999-99-99 99:99:99",
            item["relative_source_path"],
            item["clip_id"],
        )
    )
    return clip_rows


def materialise_preview_and_clips(clip_rows: list[dict[str, str]]) -> None:
    if not ffmpeg_exists():
        for row in clip_rows:
            row["standardization_status"] = "ffmpeg_missing"
            row["standardization_error"] = "ffmpeg and or ffprobe not found on PATH"
            row["preview_status"] = "ffmpeg_missing" if RUN_PREVIEW_CLIPS else "skipped"
            if RUN_PREVIEW_CLIPS:
                row["preview_error"] = "ffmpeg and or ffprobe not found on PATH"
        return

    preview_done_for_video: set[str] = set()

    for row in clip_rows:
        source_path = Path(row["source_path"])
        preview_path = Path(row["preview_path"])
        standardized_path = Path(row["standardized_path"])
        start_seconds = safe_float(row["clip_start_sec"]) or 0.0
        duration_seconds = safe_float(row["clip_duration_sec"]) or 0.0
        preview_start_sec = safe_float(row.get("preview_start_sec")) or 0.0
        preview_duration_sec = safe_float(row.get("preview_duration_sec")) or 0.0

        if RUN_PREVIEW_CLIPS and row["video_id"] not in preview_done_for_video:
            preview_ok, preview_status = make_preview_clip(
                source_path=source_path,
                preview_path=preview_path,
                start_seconds=preview_start_sec,
                duration_seconds=preview_duration_sec,
            )
            row["preview_status"] = preview_status if preview_ok else f"failed"
            row["preview_error"] = "" if preview_ok else preview_status
            preview_done_for_video.add(row["video_id"])
        elif RUN_PREVIEW_CLIPS:
            row["preview_status"] = "same_as_video_preview"
            row["preview_error"] = ""
        else:
            row["preview_status"] = "skipped"
            row["preview_error"] = ""

        if RUN_STANDARDIZE_AND_SPLIT:
            clip_ok, clip_status = standardize_clip(
                source_path=source_path,
                output_path=standardized_path,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
            )
            row["standardization_status"] = clip_status if clip_ok else "failed"
            row["standardization_error"] = "" if clip_ok else clip_status
        else:
            row["standardization_status"] = "skipped"
            row["standardization_error"] = ""


# ============================================================
# Scene frame sampling
# ============================================================
def project_relative_or_absolute(path: Path, project_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_project_root = project_root.resolve()
    try:
        return path_to_posix(resolved_path.relative_to(resolved_project_root))
    except ValueError:
        return path_to_posix(resolved_path)


def sample_frame_at_ratio(video_path: Path, ratio: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    duration_seconds = frame_count / fps if fps and frame_count else None
    if fps <= 0:
        fps = 30.0
    if frame_count <= 0:
        cap.release()
        return None, None, duration_seconds

    ratio = min(max(ratio, 0.0), 1.0)
    frame_index = min(frame_count - 1, max(0, int(round((frame_count - 1) * ratio))))
    frame = read_frame_robust(video_path, cap, frame_index, fps)
    cap.release()
    if frame is None:
        return None, None, duration_seconds

    frame_time_seconds = frame_index / fps if fps else None
    return frame, frame_time_seconds, duration_seconds


def write_scene_frames(inventory_rows: list[dict[str, str]], project_root: Path) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    scene_root = project_root / SCENE_FRAME_DIR
    ensure_folder(scene_root)

    for inventory_row in inventory_rows:
        if inventory_row.get("selected_for_output") != "yes":
            continue

        video_id = inventory_row["video_id"]
        source_path = Path(inventory_row["source_path"])
        ocr_start_time = parse_iso_datetime(inventory_row.get("ocr_start_time", ""))

        for ratio in SCENE_FRAME_SAMPLE_RATIOS:
            frame, frame_time_seconds, duration_seconds = sample_frame_at_ratio(source_path, ratio)
            if frame is None:
                manifest_rows.append({
                    "video_id": video_id,
                    "source_path": inventory_row["source_path"],
                    "sample_ratio": format_float(ratio, 3),
                    "frame_time_seconds": "",
                    "frame_wallclock_time": "",
                    "frame_path": "",
                    "status": "read_failed",
                })
                continue

            wallclock_time = None
            if ocr_start_time is not None and frame_time_seconds is not None:
                wallclock_time = ocr_start_time + timedelta(seconds=frame_time_seconds)

            output_folder = scene_root / video_id
            ensure_folder(output_folder)
            output_path = output_folder / f"{video_id}_r{int(round(ratio * 100)):02d}.jpg"

            params = [int(cv2.IMWRITE_JPEG_QUALITY), SCENE_FRAME_JPEG_QUALITY]
            ok = cv2.imwrite(str(output_path), frame, params)
            status = "ok" if ok else "write_failed"

            manifest_rows.append({
                "video_id": video_id,
                "source_path": inventory_row["source_path"],
                "sample_ratio": format_float(ratio, 3),
                "frame_time_seconds": format_float(frame_time_seconds, 3) if frame_time_seconds is not None else "",
                "frame_wallclock_time": format_datetime(wallclock_time),
                "frame_path": path_to_posix(output_path.resolve()) if ok else "",
                "status": status,
            })
    return manifest_rows


# ============================================================
# CSV writing
# ============================================================
def write_csv(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ensure_parent_folder(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cleaned = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(cleaned)


TIME_BOUNDS_FIELDNAMES = [
    "video_id",
    "video_name",
    "source_path",
    "relative_source_path",
    "ocr_start_time",
    "ocr_end_time",
    "checked_start_time",
    "checked_end_time",
    "effective_start_time",
    "effective_end_time",
    "ocr_status",
    "timestamp_overlay_visible_auto",
    "trusted_interval_start",
    "trusted_interval_end",
    "trusted_interval_source",
    "dedupe_status",
    "selected_for_output",
    "notes_manual",
]

INVENTORY_FIELDNAMES = [
    "video_id",
    "video_name",
    "source_path",
    "relative_source_path",
    "parent_folder",
    "file_size_bytes",
    "duration_seconds",
    "duration_hms",
    "fps",
    "width",
    "height",
    "resolution",
    "codec_name",
    "container_format",
    "ocr_start_time",
    "ocr_end_time",
    "checked_start_time",
    "checked_end_time",
    "effective_start_time",
    "effective_end_time",
    "ocr_status",
    "timestamp_overlay_visible_auto",
    "trusted_interval_start",
    "trusted_interval_end",
    "trusted_interval_source",
    "trusted_interval_duration_sec",
    "dedupe_status",
    "selected_for_output",
    "dedupe_notes",
    "location_overlay_visible_manual",
    "camera_view_changes_manual",
    "approximate_lighting_manual",
    "notes_manual",
]

PUBLIC_VIDEO_INVENTORY_RELATIVE_PATH = "data/video_inventory.csv"
PUBLIC_OUTPUT_VIDEO_INVENTORY_RELATIVE_PATH = "_output/video_inventory.csv"
PUBLIC_INVENTORY_EXCLUDED_FIELDS = {
    "effective_start_time",
    "effective_end_time",
    "ocr_status",
    "timestamp_overlay_visible_auto",
    "trusted_interval_start",
    "trusted_interval_end",
    "trusted_interval_source",
    "trusted_interval_duration_sec",
    "dedupe_status",
    "selected_for_output",
    "dedupe_notes",
    "location_overlay_visible_manual",
    "camera_view_changes_manual",
    "approximate_lighting_manual",
    "notes_manual",
    "video_id",
    "source_path",
    "relative_source_path",
    "parent_folder",
}
PUBLIC_INVENTORY_FIELDNAMES = [
    field for field in INVENTORY_FIELDNAMES
    if field not in PUBLIC_INVENTORY_EXCLUDED_FIELDS
]
REVIEW_MARKERS = {"need_review", "needs_review", "need review", "needs review"}

CLIP_MANIFEST_FIELDNAMES = [
    "video_id",
    "clip_id",
    "clip_index",
    "coverage_segment_index",
    "source_path",
    "relative_source_path",
    "standardized_path",
    "relative_standardized_path",
    "preview_path",
    "segment_start_sec",
    "segment_end_sec",
    "segment_duration_sec",
    "segment_start_time",
    "segment_end_time",
    "clip_start_sec",
    "clip_end_sec",
    "clip_duration_sec",
    "clip_start_time",
    "clip_end_time",
    "time_alignment_status",
    "source_fps",
    "target_fps",
    "source_width",
    "source_height",
    "target_width",
    "target_height",
    "ocr_start_time",
    "ocr_end_time",
    "checked_start_time",
    "checked_end_time",
    "effective_start_time",
    "effective_end_time",
    "ocr_status",
    "trusted_interval_start",
    "trusted_interval_end",
    "dedupe_status",
    "standardization_status",
    "standardization_error",
    "preview_status",
    "preview_error",
    "preview_start_sec",
    "preview_duration_sec",
]

SCENE_FRAME_MANIFEST_FIELDNAMES = [
    "video_id",
    "source_path",
    "sample_ratio",
    "frame_time_seconds",
    "frame_wallclock_time",
    "frame_path",
    "status",
]


def summarise_statuses(clip_rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in clip_rows:
        value = row.get(key, "") or ""
        counts[value] = counts.get(value, 0) + 1
    return counts


def print_clip_export_summary(clip_rows: list[dict[str, str]], project_root: Path) -> None:
    if not clip_rows:
        print("No clip rows were generated.")
        return

    clip_counts = summarise_statuses(clip_rows, "standardization_status")
    preview_counts = summarise_statuses(clip_rows, "preview_status")

    print()
    print("Clip export summary:")
    print(f"  Created clips: {clip_counts.get('ok', 0)}")
    print(f"  Existing clips reused: {clip_counts.get('existing', 0)}")
    print(f"  Skipped clips: {clip_counts.get('skipped', 0)}")
    print(f"  Failed clips: {clip_counts.get('failed', 0)}")
    print(f"  FFmpeg missing clips: {clip_counts.get('ffmpeg_missing', 0)}")

    if RUN_PREVIEW_CLIPS:
        print("Preview export summary:")
        print(f"  Created previews: {preview_counts.get('ok', 0)}")
        print(f"  Existing previews reused: {preview_counts.get('existing', 0)}")
        print(f"  Shared per video previews: {preview_counts.get('same_as_video_preview', 0)}")
        print(f"  Failed previews: {preview_counts.get('failed', 0)}")
        print(f"  FFmpeg missing previews: {preview_counts.get('ffmpeg_missing', 0)}")

    created_paths = [row.get('relative_standardized_path', '') for row in clip_rows if row.get('standardization_status') in {'ok', 'existing'}]
    if created_paths:
        print("Example clip paths:")
        for path in created_paths[:5]:
            print(f"  {path}")

    failures = [row for row in clip_rows if row.get('standardization_status') == 'failed']
    if failures:
        print("Example clip failures:")
        for row in failures[:5]:
            print(f"  {row.get('clip_id', '')}: {row.get('standardization_error', '')}")


# ============================================================
# Review inventory helpers
# ============================================================
def normalise_review_text(value: str) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def row_needs_review(row: dict[str, str]) -> bool:
    status = normalise_review_text(row.get("ocr_status", ""))
    return status in REVIEW_MARKERS or "need review" in status or "needs review" in status


def row_has_checked_times(row: dict[str, str]) -> bool:
    return bool(str(row.get("checked_start_time", "") or "").strip()) and bool(str(row.get("checked_end_time", "") or "").strip())


def unresolved_review_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row_needs_review(row) and not row_has_checked_times(row)]


def problem_video_names(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        name = str(row.get("video_name", "") or "").strip()
        if not name:
            name = "unknown_video"
        if name not in names:
            names.append(name)
    return names


def raise_for_unresolved_reviews(rows: list[dict[str, str]]) -> None:
    problems = unresolved_review_rows(rows)
    if not problems:
        return
    names = problem_video_names(problems)
    joined = "\n  - ".join(names)
    raise RuntimeError(
        "Stopping because these videos still need review and are missing checked_start_time or checked_end_time:\n"
        f"  - {joined}\n"
        "Fill both checked times in data/video_inventory.csv and rerun."
    )

def public_inventory_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for row in rows:
        output_rows.append({field: str(row.get(field, "") or "") for field in PUBLIC_INVENTORY_FIELDNAMES})
    return output_rows


def write_public_inventory_snapshots(rows: list[dict[str, str]], data_csv: Path, output_csv: Path) -> None:
    public_rows = public_inventory_rows(rows)
    write_csv(data_csv, PUBLIC_INVENTORY_FIELDNAMES, public_rows)
    if output_csv.resolve() != data_csv.resolve():
        write_csv(output_csv, PUBLIC_INVENTORY_FIELDNAMES, public_rows)


def load_existing_inventory_generic(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [
            {str(k): "" if v is None else str(v) for k, v in row.items()}
            for row in csv.DictReader(handle)
            if row
        ]


REQUIRED_INTERNAL_INVENTORY_FIELDS = {
    "video_id",
    "source_path",
    "relative_source_path",
    "video_name",
}


def inventory_rows_have_required_internal_fields(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    first_row = rows[0]
    return all(field in first_row for field in REQUIRED_INTERNAL_INVENTORY_FIELDS)


def load_or_build_full_inventory_rows(
    input_root: Path,
    project_root: Path,
    configured_inventory_csv: Path,
) -> list[dict[str, str]]:
    existing_rows = load_existing_inventory_generic(configured_inventory_csv)
    if inventory_rows_have_required_internal_fields(existing_rows):
        return existing_rows

    if existing_rows:
        print(f"Existing internal inventory is missing required internal columns. Rebuilding: {configured_inventory_csv}")
    else:
        print(f"Building internal inventory: {configured_inventory_csv}")

    rebuilt_rows = build_full_inventory_rows(input_root, project_root, configured_inventory_csv)
    if rebuilt_rows:
        write_csv(configured_inventory_csv, INVENTORY_FIELDNAMES, rebuilt_rows)
        print(f"Saved video inventory CSV: {configured_inventory_csv}")
    return rebuilt_rows


def index_rows_by_video_name(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row.get("video_name", "") or "").strip()
        if name:
            indexed[name] = row
    return indexed


def overlay_review_sheet(full_rows: list[dict[str, str]], review_rows: list[dict[str, str]]) -> None:
    review_by_name = index_rows_by_video_name(review_rows)
    for row in full_rows:
        name = str(row.get("video_name", "") or "").strip()
        review_row = review_by_name.get(name)
        if not review_row:
            continue
        for field in ["checked_start_time", "checked_end_time", "ocr_status"]:
            if field in review_row and str(review_row.get(field, "") or "").strip():
                row[field] = str(review_row[field])


def build_full_inventory_rows(input_root: Path, project_root: Path, configured_inventory_csv: Path) -> list[dict[str, str]]:
    if CLEAN_STALE_TEMP_OUTPUTS_ON_START:
        removed_count = cleanup_stale_temp_outputs(project_root)
        if removed_count:
            print(f"Removed stale temporary outputs: {removed_count}")

    videos = list_video_files(input_root, recursive=RECURSIVE)
    if not videos:
        print(f"No video files found in: {input_root}")
        return []

    print(f"Videos found: {len(videos)}")
    existing_time_bounds_rows = load_existing_csv_rows(project_root / TIME_BOUNDS_CSV_PATH)
    existing_inventory_rows = load_existing_csv_rows(configured_inventory_csv)
    inventory_rows: list[dict[str, str]] = []

    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] Analysing: {video_path.name}")
        try:
            inventory_rows.append(build_inventory_row(video_path, input_root))
        except Exception as exc:
            print(f"[{index}/{len(videos)}] Failed on {video_path.name}: {exc}")

    apply_existing_manual_values(
        inventory_rows=inventory_rows,
        existing_time_bounds_rows=existing_time_bounds_rows,
        existing_inventory_rows=existing_inventory_rows,
    )
    inventory_rows.sort(key=lambda item: item["relative_source_path"].lower())
    return inventory_rows


# ============================================================
# Main
# ============================================================
def main() -> None:
    input_root = Path(INPUT_PATH).expanduser().resolve()
    project_root = Path(PROJECT_ROOT).expanduser().resolve()

    if not input_root.exists():
        print(f"Input path does not exist: {input_root}")
        return

    configured_inventory_csv = project_root / INVENTORY_CSV_PATH
    data_inventory_csv = project_root / PUBLIC_VIDEO_INVENTORY_RELATIVE_PATH
    output_inventory_csv = project_root / PUBLIC_OUTPUT_VIDEO_INVENTORY_RELATIVE_PATH

    review_rows = load_existing_inventory_generic(data_inventory_csv)
    inventory_rows = load_or_build_full_inventory_rows(input_root, project_root, configured_inventory_csv)
    if not inventory_rows:
        return

    if review_rows:
        print(f"Using existing review inventory from: {data_inventory_csv}")
        overlay_review_sheet(inventory_rows, review_rows)

    write_csv(configured_inventory_csv, INVENTORY_FIELDNAMES, inventory_rows)
    write_public_inventory_snapshots(inventory_rows, data_inventory_csv, output_inventory_csv)
    print(f"Saved video inventory CSV: {configured_inventory_csv}")
    print(f"Saved video inventory CSV: {data_inventory_csv}")
    print(f"Saved video inventory CSV: {output_inventory_csv}")

    unresolved = unresolved_review_rows(inventory_rows)
    if unresolved:
        print("Videos still requiring checked times:")
        for name in problem_video_names(unresolved):
            print(f"  - {name}")
        raise_for_unresolved_reviews(inventory_rows)

    segment_plan = build_deduplicated_segment_plan(inventory_rows)

    if RUN_OCR_TIME_BOUNDS:
        time_bounds_rows = []
        for row in inventory_rows:
            time_bounds_rows.append({
                "video_id": row["video_id"],
                "video_name": row["video_name"],
                "source_path": row["source_path"],
                "relative_source_path": row["relative_source_path"],
                "ocr_start_time": row["ocr_start_time"],
                "ocr_end_time": row["ocr_end_time"],
                "checked_start_time": row["checked_start_time"],
                "checked_end_time": row["checked_end_time"],
                "effective_start_time": row["effective_start_time"],
                "effective_end_time": row["effective_end_time"],
                "ocr_status": row["ocr_status"],
                "timestamp_overlay_visible_auto": row["timestamp_overlay_visible_auto"],
                "trusted_interval_start": row["trusted_interval_start"],
                "trusted_interval_end": row["trusted_interval_end"],
                "trusted_interval_source": row["trusted_interval_source"],
                "dedupe_status": row["dedupe_status"],
                "selected_for_output": row["selected_for_output"],
                "notes_manual": row.get("notes_manual", ""),
            })
        time_bounds_csv = project_root / TIME_BOUNDS_CSV_PATH
        write_csv(time_bounds_csv, TIME_BOUNDS_FIELDNAMES, time_bounds_rows)
        print(f"Saved time bounds CSV: {time_bounds_csv}")

    if RUN_VIDEO_INVENTORY:
        inventory_csv = project_root / INVENTORY_CSV_PATH
        write_csv(inventory_csv, INVENTORY_FIELDNAMES, inventory_rows)
        write_public_inventory_snapshots(inventory_rows, data_inventory_csv, output_inventory_csv)
        print(f"Saved video inventory CSV: {inventory_csv}")
        print(f"Saved video inventory CSV: {data_inventory_csv}")
        print(f"Saved video inventory CSV: {output_inventory_csv}")

    clip_rows: list[dict[str, str]] = []
    if RUN_STANDARDIZE_AND_SPLIT or RUN_PREVIEW_CLIPS:
        inventory_lookup = {row["video_id"]: row for row in inventory_rows}
        clip_rows = build_clip_rows(inventory_lookup, segment_plan, project_root)
        materialise_preview_and_clips(clip_rows)

        clip_manifest_csv = project_root / CLIP_MANIFEST_CSV_PATH
        write_csv(clip_manifest_csv, CLIP_MANIFEST_FIELDNAMES, clip_rows)
        print(f"Saved clip manifest CSV: {clip_manifest_csv}")
        if RUN_STANDARDIZE_AND_SPLIT:
            print(f"Standardized clips folder: {project_root / STANDARDIZED_VIDEO_DIR}")
        if RUN_PREVIEW_CLIPS:
            print(f"Preview clips folder: {project_root / PREVIEW_DIR}")
        print_clip_export_summary(clip_rows, project_root)

    if RUN_SCENE_FRAME_SAMPLING:
        scene_frame_rows = write_scene_frames(inventory_rows, project_root)
        scene_frame_manifest_csv = project_root / SCENE_FRAME_MANIFEST_CSV_PATH
        write_csv(scene_frame_manifest_csv, SCENE_FRAME_MANIFEST_FIELDNAMES, scene_frame_rows)
        print(f"Saved scene frame manifest CSV: {scene_frame_manifest_csv}")
        print(f"Scene frames folder: {project_root / SCENE_FRAME_DIR}")

    selected_count = sum(1 for row in inventory_rows if row.get("selected_for_output") == "yes")
    skipped_duplicate_count = sum(1 for row in inventory_rows if row.get("dedupe_status") == "fully_covered_duplicate")
    skipped_no_time_count = sum(1 for row in inventory_rows if row.get("dedupe_status") == "skipped_no_trusted_time_range")

    print()
    print("Done.")
    print(f"Project root: {project_root}")
    print(f"Input root: {input_root}")
    print(f"Videos selected for output coverage: {selected_count}")
    print(f"Videos skipped as full duplicates: {skipped_duplicate_count}")
    print(f"Videos skipped due to missing trusted time range: {skipped_no_time_count}")
    print(f"Output clip rows planned: {len(clip_rows)}")


if __name__ == "__main__":
    main()


class VideoPreparationPipeline(PipelineStage):
    def run(self) -> None:
        global INPUT_PATH, PROJECT_ROOT, RECURSIVE
        global INVENTORY_CSV_PATH, TIME_BOUNDS_CSV_PATH, CLIP_MANIFEST_CSV_PATH, SCENE_FRAME_MANIFEST_CSV_PATH
        global STANDARDIZED_VIDEO_DIR, PREVIEW_DIR, SCENE_FRAME_DIR
        global RUN_OCR_TIME_BOUNDS, RUN_VIDEO_INVENTORY, RUN_STANDARDIZE_AND_SPLIT, RUN_PREVIEW_CLIPS, RUN_SCENE_FRAME_SAMPLING
        global CLIP_DURATION_SECONDS, PREVIEW_DURATION_SECONDS, OVERWRITE_EXISTING_OUTPUTS, KEEP_AUDIO
        global TARGET_CONTAINER_SUFFIX, TARGET_VIDEO_CODEC, TARGET_PRESET, TARGET_CRF, TARGET_FPS, TARGET_WIDTH, TARGET_HEIGHT, PIXEL_FORMAT
        global DEDUPE_OVERLAP_TOLERANCE_SECONDS, MIN_OUTPUT_SEGMENT_SECONDS, SKIP_VIDEOS_WITHOUT_TRUSTED_TIME_RANGE, CLEAN_STALE_TEMP_OUTPUTS_ON_START, TEMP_OUTPUT_SUFFIX
        global SCENE_FRAME_SAMPLE_RATIOS, SCENE_FRAME_JPEG_QUALITY
        global CROP_X, CROP_Y, CROP_W, CROP_H
        global FRAME_OFFSETS, THRESHOLDS, OCR_TIMEOUT_SECONDS, TESSERACT_CONFIGS, TIME_ALIGNMENT_TOLERANCE_SECONDS

        INPUT_PATH = str(self.context.input_path)
        PROJECT_ROOT = str(self.context.project_root)
        RECURSIVE = self.context.recursive

        INVENTORY_CSV_PATH = self.context.inventory_csv_path.as_posix()
        TIME_BOUNDS_CSV_PATH = self.context.time_bounds_csv_path.as_posix()
        CLIP_MANIFEST_CSV_PATH = self.context.clip_manifest_csv_path.as_posix()
        SCENE_FRAME_MANIFEST_CSV_PATH = self.context.scene_frame_manifest_csv_path.as_posix()

        STANDARDIZED_VIDEO_DIR = self.context.standardized_video_dir.as_posix()
        PREVIEW_DIR = self.context.preview_dir.as_posix()
        SCENE_FRAME_DIR = self.context.scene_frame_dir.as_posix()

        RUN_OCR_TIME_BOUNDS = self.context.run_ocr_time_bounds
        RUN_VIDEO_INVENTORY = self.context.run_video_inventory
        RUN_STANDARDIZE_AND_SPLIT = self.context.run_standardize_and_split
        RUN_PREVIEW_CLIPS = self.context.run_preview_clips
        RUN_SCENE_FRAME_SAMPLING = self.context.run_scene_frame_sampling

        CLIP_DURATION_SECONDS = self.context.clip_duration_seconds
        PREVIEW_DURATION_SECONDS = self.context.preview_duration_seconds
        OVERWRITE_EXISTING_OUTPUTS = self.context.overwrite_existing_outputs
        KEEP_AUDIO = self.context.keep_audio

        TARGET_CONTAINER_SUFFIX = self.context.target_container_suffix
        TARGET_VIDEO_CODEC = self.context.target_video_codec
        TARGET_PRESET = self.context.target_preset
        TARGET_CRF = self.context.target_crf
        TARGET_FPS = self.context.target_fps
        TARGET_WIDTH = self.context.target_width
        TARGET_HEIGHT = self.context.target_height
        PIXEL_FORMAT = self.context.pixel_format

        DEDUPE_OVERLAP_TOLERANCE_SECONDS = self.context.dedupe_overlap_tolerance_seconds
        MIN_OUTPUT_SEGMENT_SECONDS = self.context.min_output_segment_seconds
        SKIP_VIDEOS_WITHOUT_TRUSTED_TIME_RANGE = self.context.skip_videos_without_trusted_time_range
        CLEAN_STALE_TEMP_OUTPUTS_ON_START = self.context.clean_stale_temp_outputs_on_start
        TEMP_OUTPUT_SUFFIX = self.context.temp_output_suffix

        SCENE_FRAME_SAMPLE_RATIOS = list(self.context.scene_frame_sample_ratios)
        SCENE_FRAME_JPEG_QUALITY = self.context.scene_frame_jpeg_quality

        CROP_X = self.context.crop_x
        CROP_Y = self.context.crop_y
        CROP_W = self.context.crop_w
        CROP_H = self.context.crop_h

        FRAME_OFFSETS = list(self.context.frame_offsets)
        THRESHOLDS = list(self.context.thresholds)
        OCR_TIMEOUT_SECONDS = self.context.ocr_timeout_seconds
        TESSERACT_CONFIGS = list(self.context.tesseract_configs)
        TIME_ALIGNMENT_TOLERANCE_SECONDS = self.context.time_alignment_tolerance_seconds

        self.logger.info("Starting video preparation for {}.", INPUT_PATH)
        main()
