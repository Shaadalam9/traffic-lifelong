#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from tqdm import tqdm
from utils.base import PipelineStage


# ============================================================
# Edit only these values
# ============================================================

INPUT_PATH = "/Users/alam/Repos/traffic-lifelong/metadata/previews/live_from_fresno_california_supercar_spotting_traffic_camera_police_scanner_radio_2025_11_23_21_42.mp4"          # file or folder of videos
OUTPUT_ROOT = "tracking_outputs"              # where outputs will be written
RECURSIVE = True

MODEL_WEIGHTS = "yolo11s.pt"                 # replace with yolov8s.pt / yolo11n.pt if preferred
DEVICE = "cpu"                                    # 0 for first GPU, "cpu" for CPU
IMG_SIZE = 1280
CONFIDENCE_THRESHOLD = 0.70
IOU_THRESHOLD = 0.45

# Tracking setup
TRACKER_CONFIG = "bytetrack.yaml"             # bytetrack.yaml or botsort.yaml
TARGET_CLASSES = [2, 3, 5, 7]                 # car, motorcycle, bus, truck in COCO
PERSIST_TRACKS = True

# Video output options
WRITE_ANNOTATED_VIDEO = True
ANNOTATED_VIDEO_CODEC = "mp4v"                # fallback-friendly OpenCV codec
WRITE_FRAME_PREVIEWS = False
FRAME_PREVIEW_EVERY_N = 150

# Resume / overwrite behaviour
SKIP_IF_OUTPUT_EXISTS = True
OVERWRITE_EXISTING = False

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpeg", ".mpg",
    ".wmv", ".asf", ".ts", ".mts", ".m2ts", ".3gp", ".3g2", ".f4v",
    ".flv", ".dv", ".ogv", ".vob", ".mxf", ".dav", ".h264", ".h265",
    ".hevc",
}

# ============================================================
# Implementation
# ============================================================


@dataclass
class VideoInfo:
    path: Path
    stem_safe: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float


def slugify_filename(name: str) -> str:
    cleaned = []
    prev_was_sep = False
    for ch in name.lower():
        if ch.isalnum():
            cleaned.append(ch)
            prev_was_sep = False
        else:
            if not prev_was_sep:
                cleaned.append("_")
                prev_was_sep = True
    result = "".join(cleaned).strip("_")
    return result or "video"


def list_video_files(input_path: Path, recursive: bool) -> list[Path]:
    if not input_path.exists():
        return []
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in VIDEO_EXTENSIONS else []

    pattern = "**/*" if recursive else "*"
    files: list[Path] = []
    for path in input_path.glob(pattern):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        files.append(path)
    return sorted(files)


def inspect_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0:
        fps = 30.0

    duration_seconds = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(
        path=path,
        stem_safe=slugify_filename(path.stem),
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_if_exists(path: Path) -> None:
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def should_skip(video_dir: Path) -> bool:
    summary_path = video_dir / "summary.json"
    detections_path = video_dir / "tracks.csv"
    if not SKIP_IF_OUTPUT_EXISTS:
        return False
    return summary_path.exists() and detections_path.exists()


def box_xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        max(0.0, x2 - x1),
        max(0.0, y2 - y1),
    )


def open_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*ANNOTATED_VIDEO_CODEC)
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def draw_box(
    frame: Any,
    xyxy: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(0, y1 - text_h - baseline - 4)
    cv2.rectangle(frame, (x1, y_text), (x1 + text_w + 8, y_text + text_h + baseline + 8), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, y_text + text_h + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def color_for_track(track_id: int) -> tuple[int, int, int]:
    # deterministic color per track id
    base = int(abs(track_id) * 2654435761 % (256 * 256 * 256))
    b = base & 255
    g = (base >> 8) & 255
    r = (base >> 16) & 255
    return int(b), int(g), int(r)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def update_tracking_progress(
    progress_bar: Any,
    frame_index: int,
    total_rows_written: int,
    unique_track_ids: set[int],
) -> None:
    if frame_index % 50 != 0:
        return
    if not hasattr(progress_bar, "set_postfix"):
        return
    valid_track_count = len([track_id for track_id in unique_track_ids if track_id >= 0])
    progress_bar.set_postfix(rows=total_rows_written, tracks=valid_track_count)


def process_video(model: YOLO, video_info: VideoInfo, output_root: Path) -> dict[str, Any]:
    video_dir = output_root / video_info.stem_safe

    if OVERWRITE_EXISTING:
        remove_if_exists(video_dir)
    ensure_dir(video_dir)

    if should_skip(video_dir):
        return {
            "video_name": video_info.path.name,
            "status": "skipped_existing_outputs",
            "output_dir": str(video_dir),
        }

    tracks_csv_path = video_dir / "tracks.csv"
    summary_json_path = video_dir / "summary.json"
    annotated_video_path = video_dir / "annotated.mp4"
    preview_dir = video_dir / "frame_previews"

    if WRITE_FRAME_PREVIEWS:
        ensure_dir(preview_dir)

    writer = None
    if WRITE_ANNOTATED_VIDEO:
        writer = open_video_writer(
            annotated_video_path,
            video_info.fps,
            video_info.width,
            video_info.height,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open annotated video writer: {annotated_video_path}")

    total_frames_seen = 0
    total_rows_written = 0
    unique_track_ids: set[int] = set()
    class_counts: dict[str, int] = {}

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "video_name",
            "frame_index",
            "timestamp_sec",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "width",
            "height",
        ]
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()

        results_iter = model.track(
            source=str(video_info.path),
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            device=DEVICE,
            classes=TARGET_CLASSES if TARGET_CLASSES else None,
            tracker=TRACKER_CONFIG,
            persist=PERSIST_TRACKS,
            stream=True,
            save=False,
            verbose=False,
        )

        progress_total = video_info.frame_count if video_info.frame_count > 0 else None
        progress_bar = tqdm(
            results_iter,
            total=progress_total,
            desc=f"Tracking {video_info.path.name}",
            unit="frame",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        )

        for frame_index, result in enumerate(progress_bar):
            total_frames_seen += 1
            frame = result.orig_img
            boxes = result.boxes

            if boxes is None or boxes.xyxy is None or len(boxes) == 0:
                if writer is not None and frame is not None:
                    writer.write(frame)
                update_tracking_progress(progress_bar, frame_index, total_rows_written, unique_track_ids)
                continue

            xyxy_list = boxes.xyxy.cpu().numpy().tolist()
            conf_list = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)
            cls_list = boxes.cls.cpu().numpy().tolist() if boxes.cls is not None else [-1] * len(xyxy_list)
            if boxes.id is not None:
                id_list = boxes.id.cpu().numpy().tolist()
            else:
                id_list = [-1] * len(xyxy_list)

            names = result.names if hasattr(result, "names") and result.names is not None else {}

            for det_index, (xyxy, conf, cls_id, track_id) in enumerate(zip(xyxy_list, conf_list, cls_list, id_list)):
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                center_x, center_y, width, height = box_xyxy_to_xywh(x1, y1, x2, y2)
                cls_int = int(cls_id) if cls_id is not None else -1
                track_int = int(track_id) if track_id is not None else -1
                class_name = str(names.get(cls_int, f"class_{cls_int}"))

                unique_track_ids.add(track_int)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                writer_csv.writerow(
                    {
                        "video_name": video_info.path.name,
                        "frame_index": frame_index,
                        "timestamp_sec": round(frame_index / video_info.fps, 6),
                        "track_id": track_int,
                        "class_id": cls_int,
                        "class_name": class_name,
                        "confidence": round(float(conf), 6),
                        "x1": round(x1, 3),
                        "y1": round(y1, 3),
                        "x2": round(x2, 3),
                        "y2": round(y2, 3),
                        "center_x": round(center_x, 3),
                        "center_y": round(center_y, 3),
                        "width": round(width, 3),
                        "height": round(height, 3),
                    }
                )
                total_rows_written += 1

                if writer is not None and frame is not None:
                    label = f"{class_name} id={track_int} {float(conf):.2f}"
                    draw_box(
                        frame,
                        (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                        label,
                        color_for_track(track_int),
                    )

            if writer is not None and frame is not None:
                cv2.putText(
                    frame,
                    f"frame={frame_index}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)

            if WRITE_FRAME_PREVIEWS and frame is not None and frame_index % FRAME_PREVIEW_EVERY_N == 0:
                preview_path = preview_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(preview_path), frame)

            update_tracking_progress(progress_bar, frame_index, total_rows_written, unique_track_ids)

    if writer is not None:
        writer.release()

    summary = {
        "video_name": video_info.path.name,
        "video_path": str(video_info.path),
        "fps": video_info.fps,
        "width": video_info.width,
        "height": video_info.height,
        "frame_count_reported": video_info.frame_count,
        "duration_seconds_reported": video_info.duration_seconds,
        "frames_processed": total_frames_seen,
        "rows_written": total_rows_written,
        "unique_track_ids": sorted(unique_track_ids),
        "unique_track_count": len([tid for tid in unique_track_ids if tid >= 0]),
        "class_counts": class_counts,
        "tracks_csv": str(tracks_csv_path),
        "annotated_video": str(annotated_video_path) if WRITE_ANNOTATED_VIDEO else "",
        "status": "ok",
    }
    save_json(summary_json_path, summary)
    return summary


def main() -> None:
    input_path = Path(INPUT_PATH).expanduser()
    output_root = Path(OUTPUT_ROOT).expanduser()
    ensure_dir(output_root)

    video_paths = list_video_files(input_path, RECURSIVE)
    if not video_paths:
        print(f"No video files found in: {input_path}")
        return

    print(f"Videos found: {len(video_paths)}")
    print(f"Loading model: {MODEL_WEIGHTS}")
    model = YOLO(MODEL_WEIGHTS)

    run_summary_rows: list[dict[str, Any]] = []
    for index, video_path in enumerate(video_paths, start=1):
        print(f"[{index}/{len(video_paths)}] Processing: {video_path.name}")
        try:
            video_info = inspect_video(video_path)
            summary = process_video(model, video_info, output_root)
            run_summary_rows.append(summary)
            print(f"  status={summary.get('status')} rows={summary.get('rows_written', 0)}")
        except Exception as exc:
            failed = {
                "video_name": video_path.name,
                "video_path": str(video_path),
                "status": f"failed: {exc}",
            }
            run_summary_rows.append(failed)
            print(f"  failed: {exc}")

    run_summary_path = output_root / "run_summary.json"
    save_json(run_summary_path, {"videos": run_summary_rows})

    print()
    print(f"Done. Output root: {output_root}")
    print(f"Run summary: {run_summary_path}")


if __name__ == "__main__":
    main()


class YoloTrackingPipeline(PipelineStage):
    def run(self) -> None:
        global INPUT_PATH, OUTPUT_ROOT, RECURSIVE
        global MODEL_WEIGHTS, DEVICE, IMG_SIZE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
        global TRACKER_CONFIG, TARGET_CLASSES, PERSIST_TRACKS
        global WRITE_ANNOTATED_VIDEO, ANNOTATED_VIDEO_CODEC, WRITE_FRAME_PREVIEWS, FRAME_PREVIEW_EVERY_N
        global SKIP_IF_OUTPUT_EXISTS, OVERWRITE_EXISTING

        INPUT_PATH = str(self.context.tracking_input_path)
        OUTPUT_ROOT = str(self.context.tracking_output_root)
        RECURSIVE = self.context.recursive

        MODEL_WEIGHTS = self.context.model_weights
        DEVICE = self.context.device
        IMG_SIZE = self.context.img_size
        CONFIDENCE_THRESHOLD = self.context.confidence_threshold
        IOU_THRESHOLD = self.context.iou_threshold

        TRACKER_CONFIG = self.context.tracker_config
        TARGET_CLASSES = list(self.context.target_classes)
        PERSIST_TRACKS = self.context.persist_tracks

        WRITE_ANNOTATED_VIDEO = self.context.write_annotated_video
        ANNOTATED_VIDEO_CODEC = self.context.annotated_video_codec
        WRITE_FRAME_PREVIEWS = self.context.write_frame_previews
        FRAME_PREVIEW_EVERY_N = self.context.frame_preview_every_n

        SKIP_IF_OUTPUT_EXISTS = self.context.skip_if_output_exists
        OVERWRITE_EXISTING = self.context.overwrite_existing_tracking

        self.logger.info("Starting tracking for {}.", INPUT_PATH)
        main()
