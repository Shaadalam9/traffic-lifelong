#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

EVENTS_ROOT = "tracking_outputs"
CLIP_MANIFEST_CSV = "metadata/clip_manifest.csv"
VIDEO_INVENTORY_CSV = "metadata/video_inventory.csv"
VIDEO_TIME_BOUNDS_CSV = "metadata/video_time_bounds.csv"

MASTER_EVENTS_CSV = "event_tables/master_events.csv"
MASTER_EVENTS_JSON = "event_tables/master_events_summary.json"

RECURSIVE = True
EVENTS_FILENAME = "events.csv"
SKIP_EMPTY_EVENTS = True


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_datetime_maybe(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def to_iso_second(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt is not None else ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else default
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        return int(float(text)) if text else default
    except Exception:
        return default


def find_event_files(root: Path, recursive: bool, filename: str) -> list[Path]:
    if not root.exists():
        return []
    if recursive:
        return sorted(p for p in root.rglob(filename) if p.is_file())
    return sorted(p for p in root.glob(filename) if p.is_file())


def choose_first_nonempty(*values: str) -> str:
    for value in values:
        if str(value).strip():
            return str(value).strip()
    return ""


def load_clip_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        keys = set()
        clip_id = choose_first_nonempty(row.get("clip_id", ""))
        output_path = choose_first_nonempty(row.get("output_path", ""))
        source_path = choose_first_nonempty(row.get("source_path", ""))
        video_name = choose_first_nonempty(row.get("video_name", ""), row.get("source_video_name", ""))
        for value in [clip_id, output_path, source_path, video_name]:
            if value:
                keys.add(value)
                keys.add(Path(value).name)
                keys.add(Path(value).stem)
        for key in keys:
            by_key[key] = row
    return by_key


def load_video_inventory(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        keys = set()
        video_id = choose_first_nonempty(row.get("video_id", ""))
        video_name = choose_first_nonempty(row.get("video_name", ""))
        relative_path = choose_first_nonempty(row.get("relative_path", ""), row.get("video_relative_path", ""))
        for value in [video_id, video_name, relative_path]:
            if value:
                keys.add(value)
                keys.add(Path(value).name)
                keys.add(Path(value).stem)
        for key in keys:
            by_key[key] = row
    return by_key


def load_time_bounds(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        keys = set()
        video_name = choose_first_nonempty(row.get("video_name", ""))
        relative_path = choose_first_nonempty(row.get("relative_path", ""), row.get("video_relative_path", ""))
        for value in [video_name, relative_path]:
            if value:
                keys.add(value)
                keys.add(Path(value).name)
                keys.add(Path(value).stem)
        for key in keys:
            by_key[key] = row
    return by_key


def derive_wallclock_times(
    event_row: dict[str, str],
    clip_meta: dict[str, str] | None,
    inventory_meta: dict[str, str] | None,
    time_bounds_meta: dict[str, str] | None,
) -> tuple[str, str, str]:
    start_sec = safe_float(event_row.get("start_time_sec", ""))
    end_sec = safe_float(event_row.get("end_time_sec", ""))

    if clip_meta:
        clip_start_text = choose_first_nonempty(
            clip_meta.get("clip_start_time", ""),
            clip_meta.get("effective_clip_start_time", ""),
            clip_meta.get("trusted_clip_start_time", ""),
        )
        clip_start_dt = parse_datetime_maybe(clip_start_text)
        if clip_start_dt is not None:
            return (
                to_iso_second(clip_start_dt + timedelta(seconds=start_sec)),
                to_iso_second(clip_start_dt + timedelta(seconds=end_sec)),
                "clip_manifest",
            )

    if inventory_meta:
        inv_start_text = choose_first_nonempty(
            inventory_meta.get("effective_start_time", ""),
            inventory_meta.get("trusted_interval_start", ""),
            inventory_meta.get("checked_start_time", ""),
            inventory_meta.get("ocr_start_time", ""),
        )
        inv_start_dt = parse_datetime_maybe(inv_start_text)
        if inv_start_dt is not None:
            return (
                to_iso_second(inv_start_dt + timedelta(seconds=start_sec)),
                to_iso_second(inv_start_dt + timedelta(seconds=end_sec)),
                "video_inventory",
            )

    if time_bounds_meta:
        tb_start_text = choose_first_nonempty(
            time_bounds_meta.get("effective_start_time", ""),
            time_bounds_meta.get("checked_start_time", ""),
            time_bounds_meta.get("ocr_start_time", ""),
        )
        tb_start_dt = parse_datetime_maybe(tb_start_text)
        if tb_start_dt is not None:
            return (
                to_iso_second(tb_start_dt + timedelta(seconds=start_sec)),
                to_iso_second(tb_start_dt + timedelta(seconds=end_sec)),
                "video_time_bounds",
            )

    return "", "", ""


def detect_keys(event_file: Path, first_row: dict[str, str]) -> set[str]:
    keys = {
        str(event_file),
        event_file.name,
        event_file.stem,
        event_file.parent.name,
        str(event_file.parent),
    }
    video_name = choose_first_nonempty(first_row.get("video_name", ""))
    clip_id = choose_first_nonempty(first_row.get("clip_id", ""))
    video_id = choose_first_nonempty(first_row.get("video_id", ""))
    for value in [video_name, clip_id, video_id]:
        if value:
            keys.add(value)
            keys.add(Path(value).name)
            keys.add(Path(value).stem)
    return keys


def first_match(mapping: dict[str, dict[str, str]], keys: set[str]) -> dict[str, str] | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def merge_events() -> None:
    events_root = Path(EVENTS_ROOT).expanduser()
    master_csv_path = Path(MASTER_EVENTS_CSV).expanduser()
    master_json_path = Path(MASTER_EVENTS_JSON).expanduser()

    clip_manifest_map = load_clip_manifest(Path(CLIP_MANIFEST_CSV).expanduser())
    inventory_map = load_video_inventory(Path(VIDEO_INVENTORY_CSV).expanduser())
    time_bounds_map = load_time_bounds(Path(VIDEO_TIME_BOUNDS_CSV).expanduser())

    event_files = find_event_files(events_root, RECURSIVE, EVENTS_FILENAME)
    if not event_files:
        raise FileNotFoundError(f"No {EVENTS_FILENAME} files found under: {events_root}")

    merged_rows: list[dict[str, Any]] = []
    total_source_rows = 0
    files_used = 0

    for event_file in event_files:
        rows = read_csv_rows(event_file)
        if SKIP_EMPTY_EVENTS and not rows:
            continue

        files_used += 1
        total_source_rows += len(rows)

        first_row = rows[0] if rows else {}
        keys = detect_keys(event_file, first_row)

        clip_meta = first_match(clip_manifest_map, keys)
        inventory_meta = first_match(inventory_map, keys)
        time_bounds_meta = first_match(time_bounds_map, keys)

        derived_video_name = choose_first_nonempty(
            first_row.get("video_name", ""),
            clip_meta.get("video_name", "") if clip_meta else "",
            inventory_meta.get("video_name", "") if inventory_meta else "",
            time_bounds_meta.get("video_name", "") if time_bounds_meta else "",
        )
        derived_video_id = choose_first_nonempty(
            first_row.get("video_id", ""),
            clip_meta.get("video_id", "") if clip_meta else "",
            inventory_meta.get("video_id", "") if inventory_meta else "",
        )
        derived_clip_id = choose_first_nonempty(
            first_row.get("clip_id", ""),
            clip_meta.get("clip_id", "") if clip_meta else "",
            event_file.parent.name,
        )
        derived_day_id = choose_first_nonempty(
            first_row.get("day_id", ""),
            clip_meta.get("day_id", "") if clip_meta else "",
            inventory_meta.get("day_id", "") if inventory_meta else "",
        )

        for row in rows:
            wallclock_start, wallclock_end, wallclock_source = derive_wallclock_times(
                row, clip_meta, inventory_meta, time_bounds_meta
            )
            merged_rows.append(
                {
                    "source_events_file": str(event_file),
                    "video_name": choose_first_nonempty(row.get("video_name", ""), derived_video_name),
                    "video_id": choose_first_nonempty(row.get("video_id", ""), derived_video_id),
                    "clip_id": choose_first_nonempty(row.get("clip_id", ""), derived_clip_id),
                    "day_id": choose_first_nonempty(row.get("day_id", ""), derived_day_id),
                    "track_id": row.get("track_id", ""),
                    "class_name": row.get("class_name", ""),
                    "start_frame": row.get("start_frame", ""),
                    "end_frame": row.get("end_frame", ""),
                    "start_time_sec": row.get("start_time_sec", ""),
                    "end_time_sec": row.get("end_time_sec", ""),
                    "duration_sec": row.get("duration_sec", ""),
                    "num_points": row.get("num_points", ""),
                    "mean_confidence": row.get("mean_confidence", ""),
                    "entry_zone": row.get("entry_zone", ""),
                    "exit_zone": row.get("exit_zone", ""),
                    "route_type": row.get("route_type", ""),
                    "first_crossing_time_sec": row.get("first_crossing_time_sec", ""),
                    "last_crossing_time_sec": row.get("last_crossing_time_sec", ""),
                    "wallclock_start": wallclock_start,
                    "wallclock_end": wallclock_end,
                    "wallclock_source": wallclock_source,
                }
            )

    merged_rows.sort(
        key=lambda r: (
            str(r.get("video_name", "")),
            safe_float(r.get("start_time_sec", "")),
            safe_int(r.get("track_id", "")),
        )
    )

    ensure_parent_dir(master_csv_path)
    ensure_parent_dir(master_json_path)

    fieldnames = [
        "source_events_file",
        "video_name",
        "video_id",
        "clip_id",
        "day_id",
        "track_id",
        "class_name",
        "start_frame",
        "end_frame",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "num_points",
        "mean_confidence",
        "entry_zone",
        "exit_zone",
        "route_type",
        "first_crossing_time_sec",
        "last_crossing_time_sec",
        "wallclock_start",
        "wallclock_end",
        "wallclock_source",
    ]

    with master_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    route_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    for row in merged_rows:
        route = str(row.get("route_type", "")).strip()
        cls = str(row.get("class_name", "")).strip()
        if route:
            route_counts[route] = route_counts.get(route, 0) + 1
        if cls:
            class_counts[cls] = class_counts.get(cls, 0) + 1

    summary = {
        "events_root": str(events_root),
        "events_files_found": len(event_files),
        "events_files_used": files_used,
        "source_event_rows": total_source_rows,
        "merged_event_rows": len(merged_rows),
        "master_events_csv": str(master_csv_path),
        "route_counts": route_counts,
        "class_counts": class_counts,
    }

    with master_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Event files found: {len(event_files)}")
    print(f"Event files used: {files_used}")
    print(f"Merged event rows: {len(merged_rows)}")
    print(f"Master CSV: {master_csv_path}")
    print(f"Summary JSON: {master_json_path}")
    print(f"Route counts: {route_counts}")


if __name__ == "__main__":
    merge_events()


from utils.base import PipelineContext, PipelineStage


class EventTableMerger(PipelineStage):
    def run(self) -> None:
        global EVENTS_ROOT, CLIP_MANIFEST_CSV, VIDEO_INVENTORY_CSV, VIDEO_TIME_BOUNDS_CSV
        global MASTER_EVENTS_CSV, MASTER_EVENTS_JSON
        global RECURSIVE, EVENTS_FILENAME, SKIP_EMPTY_EVENTS

        EVENTS_ROOT = str(self.context.merge_events_root)
        CLIP_MANIFEST_CSV = str(self.context.clip_manifest_csv_path)
        VIDEO_INVENTORY_CSV = str(self.context.inventory_csv_path)
        VIDEO_TIME_BOUNDS_CSV = str(self.context.time_bounds_csv_path)

        MASTER_EVENTS_CSV = str(self.context.merge_master_csv)
        MASTER_EVENTS_JSON = str(self.context.merge_master_json)

        RECURSIVE = self.context.recursive
        EVENTS_FILENAME = self.context.merge_events_filename
        SKIP_EMPTY_EVENTS = self.context.skip_empty_events

        self.logger.info("Starting event merge from {}.", EVENTS_ROOT)
        merge_events()
