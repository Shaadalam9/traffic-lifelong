#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None

try:
    from run_analysis import create_context, analysis_root_from_context, cfg, cfg_bool, cfg_path
except Exception as exc:
    raise RuntimeError(
        "Could not import config helpers from run_analysis.py. "
        "Run this script from the project root where run_analysis.py and config exist."
    ) from exc


# ============================================================
# Playback and review settings
# ============================================================
VIDEO_PLAYER_DEFAULT = "opencv"

PRE_EVENT_PADDING_SECONDS_DEFAULT = 2.0
POST_EVENT_PADDING_SECONDS_DEFAULT = 2.0

MAX_DISPLAY_WIDTH_DEFAULT = 1280
MAX_DISPLAY_HEIGHT_DEFAULT = 720
PLAYBACK_SPEED_DEFAULT = 1.0

RESUME_EXISTING_REVIEW_DEFAULT = True
MAX_NEW_REVIEWS_DEFAULT = None

BOX_FRAME_TOLERANCE_DEFAULT = 3


# ============================================================
# Review questions
# ============================================================
REVIEW_FIELDS = [
    "event_visible",
    "single_vehicle_track",
    "class_correct",
    "route_correct",
    "start_end_reasonable",
    "final_label",
    "notes",
]

YES_NO_AMBIGUOUS_FIELDS = [
    "event_visible",
    "single_vehicle_track",
    "class_correct",
    "route_correct",
    "start_end_reasonable",
]

FINAL_LABELS = [
    "valid",
    "route_error",
    "tracking_error",
    "false_positive",
    "ambiguous",
]

QUESTION_TEXT = {
    "event_visible": "Is a vehicle visible in the event window?",
    "single_vehicle_track": "Does the highlighted box follow one same vehicle?",
    "class_correct": "Is the predicted class plausible?",
    "route_correct": "Is the route label correct?",
    "start_end_reasonable": "Does the timing cover the event reasonably?",
}


# ============================================================
# Config helpers
# ============================================================
def cfg_float(key: str, default: float) -> float:
    value = cfg(key, default)
    try:
        return float(value)
    except Exception:
        return default


def cfg_int(key: str, default: int) -> int:
    value = cfg(key, default)
    try:
        return int(value)
    except Exception:
        return default


def cfg_int_or_none(key: str, default: int | None) -> int | None:
    value = cfg(key, default)

    if value in (None, ""):
        return None

    try:
        return int(value)
    except Exception:
        return default


def resolve_against_project(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()

    project_root = cfg_path("project_root", Path("."))
    if project_root is None:
        project_root = Path(".")

    return (project_root.expanduser().resolve() / path).resolve()


def get_review_paths(context) -> tuple[Path, Path]:
    analysis_root = analysis_root_from_context(context)

    configured_input = cfg_path("manual_review_input_csv", None)
    configured_output = cfg_path("manual_review_output_csv", None)

    if configured_input is not None:
        input_csv = resolve_against_project(configured_input)
    else:
        input_csv = (
            analysis_root
            / "enriched_privacy_results"
            / "tables"
            / "manual_review_sample.csv"
        )

    if configured_output is not None:
        output_csv = resolve_against_project(configured_output)
    else:
        output_csv = (
            analysis_root
            / "enriched_privacy_results"
            / "tables"
            / "manual_review_completed.csv"
        )

    return input_csv.resolve(), output_csv.resolve()


def get_video_search_roots(context) -> list[Path]:
    roots: list[Path] = []

    configured_roots = cfg("validation_video_search_roots", None)
    if configured_roots:
        if isinstance(configured_roots, str):
            configured_roots = [configured_roots]

        for item in configured_roots:
            path = Path(str(item)).expanduser()
            roots.append(resolve_against_project(path))

    if context.standardized_video_dir is not None:
        roots.append(context.standardized_video_dir.resolve())

    if context.tracking_output_root is not None:
        roots.append(context.tracking_output_root.resolve())

    if context.input_path is not None:
        roots.append(context.input_path.resolve())

    mitigation_root = cfg_path("mitigation_output_root", None)
    if mitigation_root is not None:
        roots.append(resolve_against_project(mitigation_root))
    else:
        roots.append((context.output_root / "mitigation_videos").resolve())

    unique_roots: list[Path] = []
    seen: set[str] = set()

    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)

    return unique_roots


# ============================================================
# CSV and JSON helpers
# ============================================================
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value or "").strip()
        return float(text) if text else default
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value or "").strip()
        return int(float(text)) if text else default
    except Exception:
        return default


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_json_maybe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def review_key(row: dict[str, str]) -> str:
    return "|".join(
        [
            str(row.get("source_events_file", "")),
            str(row.get("clip_id", "")),
            str(row.get("track_id", "")),
            str(row.get("start_time_sec", "")),
            str(row.get("end_time_sec", "")),
        ]
    )


# ============================================================
# Video and tracks lookup
# ============================================================
def build_video_index(search_roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

    for root in search_roots:
        if not root.exists():
            continue

        if root.is_file() and root.suffix.lower() in video_suffixes:
            index.setdefault(root.name, root)
            index.setdefault(root.stem, root)
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in video_suffixes:
                continue

            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)

    return index


def get_events_path_from_row(row: dict[str, str]) -> Path | None:
    source_events_file = str(row.get("source_events_file", "") or "").strip()

    if not source_events_file:
        return None

    return Path(source_events_file).expanduser()


def get_tracks_csv_for_row(row: dict[str, str], context) -> Path | None:
    events_path = get_events_path_from_row(row)

    candidates: list[Path] = []

    if events_path is not None:
        candidates.append(events_path.with_name("tracks.csv"))

        clip_folder_name = events_path.parent.name
        if clip_folder_name:
            candidates.append(context.tracking_output_root / clip_folder_name / "tracks.csv")

    clip_id = str(row.get("clip_id", "") or "").strip()
    if clip_id:
        candidates.append(context.tracking_output_root / clip_id / "tracks.csv")

    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.exists():
            return candidate.resolve()

    return None


def video_from_tracking_summary(row: dict[str, str], context) -> Path | None:
    tracks_csv = get_tracks_csv_for_row(row, context)
    if tracks_csv is None:
        return None

    summary_path = tracks_csv.with_name("summary.json")
    summary = read_json_maybe(summary_path)

    video_path_text = str(summary.get("video_path", "") or "").strip()
    if video_path_text:
        video_path = Path(video_path_text).expanduser()
        if video_path.exists():
            return video_path.resolve()

    return None


def find_video_for_row(row: dict[str, str], video_index: dict[str, Path], context) -> Path | None:
    direct_video = video_from_tracking_summary(row, context)
    if direct_video is not None:
        return direct_video

    candidates = [
        row.get("video_name", ""),
        row.get("clip_id", ""),
        f"{row.get('clip_id', '')}.mp4" if row.get("clip_id") else "",
    ]

    events_path = get_events_path_from_row(row)
    if events_path is not None:
        candidates.extend(
            [
                events_path.parent.name,
                f"{events_path.parent.name}.mp4",
            ]
        )

        annotated_path = events_path.parent / "annotated.mp4"
        if annotated_path.exists():
            return annotated_path.resolve()

    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate:
            continue

        if candidate in video_index:
            return video_index[candidate]

        stem = Path(candidate).stem
        if stem in video_index:
            return video_index[stem]

    return None


# ============================================================
# Target track boxes
# ============================================================
def load_target_track_boxes(
    row: dict[str, str],
    context,
) -> tuple[dict[int, dict[str, float]], Path | None]:
    tracks_csv = get_tracks_csv_for_row(row, context)
    if tracks_csv is None or not tracks_csv.exists():
        return {}, tracks_csv

    target_track_id = safe_int(row.get("track_id"), default=-1)
    if target_track_id < 0:
        return {}, tracks_csv

    boxes: dict[int, dict[str, float]] = {}

    with tracks_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)

        for track_row in reader:
            track_id = safe_int(track_row.get("track_id"), default=-999999)
            if track_id != target_track_id:
                continue

            frame_index = safe_int(track_row.get("frame_index"), default=-1)
            if frame_index < 0:
                continue

            box = {
                "x1": safe_float(track_row.get("x1")),
                "y1": safe_float(track_row.get("y1")),
                "x2": safe_float(track_row.get("x2")),
                "y2": safe_float(track_row.get("y2")),
                "confidence": safe_float(track_row.get("confidence"), default=0.0),
            }

            boxes[frame_index] = box

    return boxes, tracks_csv


def get_box_for_frame(
    frame_index: int,
    boxes: dict[int, dict[str, float]],
    tolerance: int,
) -> dict[str, float] | None:
    if frame_index in boxes:
        return boxes[frame_index]

    for offset in range(1, tolerance + 1):
        before = frame_index - offset
        after = frame_index + offset

        if before in boxes:
            return boxes[before]

        if after in boxes:
            return boxes[after]

    return None


def draw_target_box(
    frame: Any,
    box: dict[str, float] | None,
    row: dict[str, str],
) -> None:
    if cv2 is None or box is None:
        return

    height, width = frame.shape[:2]

    x1 = max(0, min(width - 1, int(round(box["x1"]))))
    y1 = max(0, min(height - 1, int(round(box["y1"]))))
    x2 = max(0, min(width - 1, int(round(box["x2"]))))
    y2 = max(0, min(height - 1, int(round(box["y2"]))))

    if x2 <= x1 or y2 <= y1:
        return

    box_colour = (0, 255, 255)
    text_colour = (0, 0, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 3)

    label = (
        f"TARGET track={row.get('track_id', '')} "
        f"class={row.get('class_name', '')} "
        f"route={row.get('route_type', '')}"
    )

    text_x = x1
    text_y = max(24, y1 - 8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)

    cv2.rectangle(
        frame,
        (text_x, text_y - text_h - baseline - 6),
        (min(width - 1, text_x + text_w + 8), text_y + baseline),
        box_colour,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (text_x + 4, text_y - 4),
        font,
        scale,
        text_colour,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# Display helpers
# ============================================================
def print_event_summary(
    row: dict[str, str],
    index: int,
    total: int,
    video_path: Path | None,
    tracks_csv: Path | None,
    boxes: dict[int, dict[str, float]],
) -> None:
    print("\n" + "=" * 78)
    print(f"Review item {index}/{total}")
    print("-" * 78)
    print(f"Group:                {row.get('review_group', '')}")
    print(f"Video:                {row.get('video_name', '')}")
    print(f"Clip ID:              {row.get('clip_id', '')}")
    print(f"Target track ID:      {row.get('track_id', '')}")
    print(f"Predicted class:      {row.get('class_name', '')}")
    print(f"Predicted route:      {row.get('route_type', '')}")
    print(f"Wall clock:           {row.get('wallclock_start', '')} to {row.get('wallclock_end', '')}")
    print(f"Event seconds:        {row.get('start_time_sec', '')} to {row.get('end_time_sec', '')}")
    print(f"Duration:             {row.get('duration_sec', '')}")
    print(f"Mean confidence:      {row.get('mean_confidence', '')}")
    print(
        "Size/speed/duration:  "
        f"{row.get('size_bucket', '')}, "
        f"{row.get('speed_bucket', '')}, "
        f"{row.get('duration_bucket', '')}"
    )
    print(f"Signature count:      {row.get('size_motion_colour_signature_count', '')}")
    print(f"Video path:           {video_path if video_path else 'NOT FOUND'}")
    print(f"Tracks CSV:           {tracks_csv if tracks_csv else 'NOT FOUND'}")
    print(f"Target boxes loaded:  {len(boxes)}")
    print("=" * 78)


def overlay_text(frame: Any, lines: list[str]) -> None:
    if cv2 is None:
        return

    x, y = 20, 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    line_height = 28

    for i, line in enumerate(lines):
        yy = y + i * line_height

        cv2.putText(
            frame,
            line,
            (x, yy),
            font,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, yy),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def resize_for_display(frame: Any, max_width: int, max_height: int) -> Any:
    if cv2 is None:
        return frame

    height, width = frame.shape[:2]
    scale = min(
        max_width / max(width, 1),
        max_height / max(height, 1),
        1.0,
    )

    if scale >= 1.0:
        return frame

    return cv2.resize(frame, (int(width * scale), int(height * scale)))


# ============================================================
# Video playback
# ============================================================
def play_with_opencv(
    video_path: Path,
    row: dict[str, str],
    boxes: dict[int, dict[str, float]],
    box_frame_tolerance: int,
    max_display_width: int,
    max_display_height: int,
    playback_speed: float,
    pre_padding: float,
    post_padding: float,
) -> bool:
    if cv2 is None:
        print("OpenCV is not available. Falling back to ffplay. Bounding box overlay will not be shown.")
        return play_with_ffplay(video_path, row, pre_padding, post_padding)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video with OpenCV: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

    event_start = safe_float(row.get("start_time_sec"), 0.0)
    event_end = safe_float(row.get("end_time_sec"), event_start + 5.0)

    start_sec = max(0.0, event_start - pre_padding)
    end_sec = event_end + post_padding

    if duration > 0:
        end_sec = min(duration, end_sec)

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    delay_ms = max(1, int(1000 / max(fps * playback_speed, 1)))

    window_name = "Manual validation: yellow box is target track"
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    paused = False
    current_frame = start_frame

    while current_frame <= end_frame:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or current_frame + 1)
            displayed_frame_index = max(0, current_frame - 1)
            current_sec = displayed_frame_index / fps if fps > 0 else 0.0

            target_box = get_box_for_frame(
                displayed_frame_index,
                boxes,
                box_frame_tolerance,
            )

            draw_target_box(frame, target_box, row)

            if target_box is None:
                box_status = "NO TARGET BOX ON THIS FRAME"
            else:
                box_status = "yellow box = target track"

            lines = [
                f"{row.get('review_group', '')} | track {row.get('track_id', '')}",
                f"class={row.get('class_name', '')} route={row.get('route_type', '')} conf={row.get('mean_confidence', '')[:5]}",
                f"event {event_start:.1f}-{event_end:.1f}s | now {current_sec:.1f}s | frame {displayed_frame_index}",
                box_status,
                "keys: q stop, space pause/resume, r replay",
            ]

            overlay_text(frame, lines)
            frame = resize_for_display(frame, max_display_width, max_display_height)
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(delay_ms if not paused else 100) & 0xFF

        if key == ord("q") or key == 27:
            break

        if key == ord(" "):
            paused = not paused

        if key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            paused = False

    cap.release()
    cv2.destroyAllWindows()
    return True


def play_with_ffplay(
    video_path: Path,
    row: dict[str, str],
    pre_padding: float,
    post_padding: float,
) -> bool:
    if shutil.which("ffplay") is None:
        print("ffplay is not installed or not on PATH.")
        return False

    event_start = safe_float(row.get("start_time_sec"), 0.0)
    event_end = safe_float(row.get("end_time_sec"), event_start + 5.0)

    start_sec = max(0.0, event_start - pre_padding)
    duration_sec = max(
        3.0,
        (event_end - event_start) + pre_padding + post_padding,
    )

    command = [
        "ffplay",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration_sec:.3f}",
        "-autoexit",
        str(video_path),
    ]

    print("Opening ffplay. Note: ffplay mode does not draw the target bounding box.")
    subprocess.run(command, check=False)
    return True


def show_video(
    video_path: Path,
    row: dict[str, str],
    boxes: dict[int, dict[str, float]],
    video_player: str,
    box_frame_tolerance: int,
    max_display_width: int,
    max_display_height: int,
    playback_speed: float,
    pre_padding: float,
    post_padding: float,
) -> bool:
    if video_player.lower() == "ffplay":
        return play_with_ffplay(video_path, row, pre_padding, post_padding)

    return play_with_opencv(
        video_path=video_path,
        row=row,
        boxes=boxes,
        box_frame_tolerance=box_frame_tolerance,
        max_display_width=max_display_width,
        max_display_height=max_display_height,
        playback_speed=playback_speed,
        pre_padding=pre_padding,
        post_padding=post_padding,
    )


# ============================================================
# Questions
# ============================================================
def ask_yes_no_ambiguous(field: str) -> str:
    prompt = f"{QUESTION_TEXT[field]} [y=yes / n=no / a=ambiguous / s=skip / q=quit]: "

    while True:
        value = input(prompt).strip().lower()

        if value in {"y", "yes"}:
            return "yes"

        if value in {"n", "no"}:
            return "no"

        if value in {"a", "ambiguous", "amb"}:
            return "ambiguous"

        if value in {"s", "skip"}:
            return ""

        if value in {"q", "quit"}:
            raise KeyboardInterrupt

        print("Please enter y, n, a, s, or q.")


def ask_final_label() -> str:
    print("Final label options:")

    for index, label in enumerate(FINAL_LABELS, start=1):
        print(f"  {index}. {label}")

    while True:
        value = input("Overall judgement [1-5, or text, q=quit]: ").strip().lower()

        if value in {"q", "quit"}:
            raise KeyboardInterrupt

        if value.isdigit():
            selected = int(value)
            if 1 <= selected <= len(FINAL_LABELS):
                return FINAL_LABELS[selected - 1]

        if value in FINAL_LABELS:
            return value

        print("Please enter a number from 1 to 5, or one of the label names.")


def ask_review_answers() -> dict[str, str]:
    answers: dict[str, str] = {}

    for field in YES_NO_AMBIGUOUS_FIELDS:
        answers[field] = ask_yes_no_ambiguous(field)

    answers["final_label"] = ask_final_label()
    answers["notes"] = input("Notes [optional]: ").strip()

    return answers


# ============================================================
# Main
# ============================================================
def main() -> None:
    context = create_context()

    input_csv, output_csv = get_review_paths(context)
    search_roots = get_video_search_roots(context)

    video_player = str(cfg("validation_video_player", VIDEO_PLAYER_DEFAULT))

    pre_padding = cfg_float(
        "validation_pre_event_padding_seconds",
        PRE_EVENT_PADDING_SECONDS_DEFAULT,
    )
    post_padding = cfg_float(
        "validation_post_event_padding_seconds",
        POST_EVENT_PADDING_SECONDS_DEFAULT,
    )
    max_display_width = cfg_int(
        "validation_max_display_width",
        MAX_DISPLAY_WIDTH_DEFAULT,
    )
    max_display_height = cfg_int(
        "validation_max_display_height",
        MAX_DISPLAY_HEIGHT_DEFAULT,
    )
    playback_speed = cfg_float(
        "validation_playback_speed",
        PLAYBACK_SPEED_DEFAULT,
    )
    box_frame_tolerance = cfg_int(
        "validation_box_frame_tolerance",
        BOX_FRAME_TOLERANCE_DEFAULT,
    )
    resume_existing = cfg_bool(
        "validation_resume_existing_review",
        RESUME_EXISTING_REVIEW_DEFAULT,
    )
    max_new_reviews = cfg_int_or_none(
        "validation_max_new_reviews",
        MAX_NEW_REVIEWS_DEFAULT,
    )

    print("Interactive event validator")
    print("Version: target-track-bounding-box-v1")
    print(f"Output root:   {context.output_root}")
    print(f"Tracking root: {context.tracking_output_root}")
    print(f"Review input:  {input_csv}")
    print(f"Review output: {output_csv}")
    print(f"Video player:  {video_player}")
    print(f"Box tolerance: {box_frame_tolerance} frame(s)")
    print("Video search roots:")

    for root in search_roots:
        print(f"  {root}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Review input CSV not found: {input_csv}")

    input_rows = read_csv_rows(input_csv)

    if not input_rows:
        print(f"No rows found in: {input_csv}")
        return

    existing_rows = read_csv_rows(output_csv)
    existing_by_key = (
        {review_key(row): row for row in existing_rows}
        if resume_existing
        else {}
    )

    print(f"Input rows: {len(input_rows)}")
    print(f"Already reviewed rows: {len(existing_by_key)}")
    print("Indexing video files...")

    video_index = build_video_index(search_roots)

    print(f"Indexed videos: {len(set(video_index.values()))}")

    fieldnames = list(input_rows[0].keys())

    for field in REVIEW_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    extra_fields = [
        "review_video_path",
        "review_tracks_csv",
        "review_target_boxes_loaded",
    ]

    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    output_rows = existing_rows[:] if resume_existing else []
    new_reviews = 0

    try:
        for row_index, row in enumerate(input_rows, start=1):
            key = review_key(row)

            if key in existing_by_key:
                continue

            if max_new_reviews is not None and new_reviews >= max_new_reviews:
                break

            video_path = find_video_for_row(row, video_index, context)
            boxes, tracks_csv = load_target_track_boxes(row, context)

            print_event_summary(
                row=row,
                index=row_index,
                total=len(input_rows),
                video_path=video_path,
                tracks_csv=tracks_csv,
                boxes=boxes,
            )

            if video_path is None:
                print("Video not found. You can still mark this row as ambiguous or skip it.")
            elif not boxes:
                print("Target track boxes not found. The video can play, but the target vehicle will not be highlighted.")
                print("For 'single vehicle track', use 'a' unless you can confidently judge.")
            else:
                print("The yellow box shows the exact target track for this review item.")

            if video_path is not None:
                while True:
                    show_video(
                        video_path=video_path,
                        row=row,
                        boxes=boxes,
                        video_player=video_player,
                        box_frame_tolerance=box_frame_tolerance,
                        max_display_width=max_display_width,
                        max_display_height=max_display_height,
                        playback_speed=playback_speed,
                        pre_padding=pre_padding,
                        post_padding=post_padding,
                    )

                    action = input(
                        "Replay video? [r=replay / c=continue to questions / q=quit]: "
                    ).strip().lower()

                    if action in {"r", "replay"}:
                        continue

                    if action in {"q", "quit"}:
                        raise KeyboardInterrupt

                    break

            answers = ask_review_answers()

            completed_row = dict(row)
            completed_row.update(answers)
            completed_row["review_video_path"] = str(video_path or "")
            completed_row["review_tracks_csv"] = str(tracks_csv or "")
            completed_row["review_target_boxes_loaded"] = str(len(boxes))

            output_rows.append(completed_row)
            existing_by_key[key] = completed_row
            new_reviews += 1

            write_csv_rows(output_csv, fieldnames, output_rows)
            print(f"Saved review {new_reviews} new row(s) to: {output_csv}")

    except KeyboardInterrupt:
        print("\nReview stopped by user. Saving progress...")
        write_csv_rows(output_csv, fieldnames, output_rows)
        print(f"Saved: {output_csv}")
        return

    write_csv_rows(output_csv, fieldnames, output_rows)

    print("\nDone.")
    print(f"New reviews completed: {new_reviews}")
    print(f"Total rows in output: {len(output_rows)}")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()