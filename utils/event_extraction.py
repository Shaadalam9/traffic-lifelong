#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

TRACKS_CSV_PATH = "/Users/alam/Repos/traffic-lifelong/tracking_outputs/live_from_fresno_california_supercar_spotting_traffic_camera_police_scanner_radio_2025_11_23_21_42/tracks.csv"
SCENE_REGIONS_JSON_PATH = "readme/scene_regions.json"
EVENTS_CSV_PATH = "events.csv"
EVENTS_DEBUG_JSON_PATH = "events_debug.json"

REQUIRED_ENTRY_BOUNDARY = "boundary_bottom"

EXIT_TO_ROUTE = {
    "boundary_far_left": "left",
    "boundary_far_center": "straight",
    "boundary_far_right": "right",
}

MIN_TRACK_POINTS = 5
MIN_TRACK_DURATION_SEC = 0.75
MIN_CROSSING_FRAME_GAP = 3
DROP_TRACKS_WITHOUT_REQUIRED_ENTRY = True


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def seg_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    def orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c) -> bool:
        return (
            min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
        )

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True
    if abs(o1) < 1e-9 and on_segment(p1, p2, q1):
        return True
    if abs(o2) < 1e-9 and on_segment(p1, p2, q2):
        return True
    if abs(o3) < 1e-9 and on_segment(q1, q2, p1):
        return True
    if abs(o4) < 1e-9 and on_segment(q1, q2, p2):
        return True
    return False


def polyline_segments(points: list[list[float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if len(points) < 2:
        return []
    out = []
    for i in range(len(points) - 1):
        p1 = (float(points[i][0]), float(points[i][1]))
        p2 = (float(points[i + 1][0]), float(points[i + 1][1]))
        out.append((p1, p2))
    return out


def track_segment_crosses_boundary(
    a: tuple[float, float],
    b: tuple[float, float],
    boundary_segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> bool:
    for q1, q2 in boundary_segments:
        if seg_intersection(a, b, q1, q2):
            return True
    return False


def load_tracks_csv(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                track_id = int(float(row["track_id"]))
            except Exception:
                continue
            if track_id < 0:
                continue
            grouped[track_id].append(
                {
                    "video_name": row.get("video_name", ""),
                    "frame_index": int(float(row["frame_index"])),
                    "timestamp_sec": float(row["timestamp_sec"]),
                    "track_id": track_id,
                    "class_id": int(float(row["class_id"])),
                    "class_name": row.get("class_name", ""),
                    "confidence": float(row["confidence"]),
                    "center_x": float(row["center_x"]),
                    "center_y": float(row["center_y"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                }
            )
    for track_id in grouped:
        grouped[track_id].sort(key=lambda r: (r["frame_index"], r["timestamp_sec"]))
    return dict(grouped)


def extract_boundaries(scene: dict[str, Any]) -> dict[str, list[tuple[tuple[float, float], tuple[float, float]]]]:
    out: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = {}
    boundaries = scene.get("boundaries", {})
    for name, payload in boundaries.items():
        centerline = payload.get("centerline", [])
        segments = polyline_segments(centerline)
        if segments:
            out[name] = segments
    return out


def find_boundary_crossings(
    track_rows: list[dict[str, Any]],
    boundaries: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]],
) -> list[dict[str, Any]]:
    crossings: list[dict[str, Any]] = []
    last_crossing_frame_by_boundary: dict[str, int] = {}
    for i in range(len(track_rows) - 1):
        r1 = track_rows[i]
        r2 = track_rows[i + 1]
        p1 = (r1["center_x"], r1["center_y"])
        p2 = (r2["center_x"], r2["center_y"])
        for boundary_name, boundary_segments in boundaries.items():
            if track_segment_crosses_boundary(p1, p2, boundary_segments):
                prev_frame = last_crossing_frame_by_boundary.get(boundary_name, -10**9)
                if r2["frame_index"] - prev_frame < MIN_CROSSING_FRAME_GAP:
                    continue
                last_crossing_frame_by_boundary[boundary_name] = r2["frame_index"]
                crossings.append(
                    {
                        "boundary": boundary_name,
                        "frame_index": r2["frame_index"],
                        "timestamp_sec": r2["timestamp_sec"],
                        "segment_index": i,
                    }
                )
    crossings.sort(key=lambda item: (item["frame_index"], item["timestamp_sec"]))
    return crossings


def summarise_track(track_rows: list[dict[str, Any]]) -> dict[str, Any]:
    start = track_rows[0]
    end = track_rows[-1]
    classes = [r["class_name"] for r in track_rows if r.get("class_name")]
    class_name = max(set(classes), key=classes.count) if classes else ""
    mean_conf = sum(r["confidence"] for r in track_rows) / len(track_rows)
    return {
        "video_name": start["video_name"],
        "track_id": start["track_id"],
        "class_name": class_name,
        "start_frame": start["frame_index"],
        "end_frame": end["frame_index"],
        "start_time_sec": start["timestamp_sec"],
        "end_time_sec": end["timestamp_sec"],
        "duration_sec": max(0.0, end["timestamp_sec"] - start["timestamp_sec"]),
        "num_points": len(track_rows),
        "mean_confidence": mean_conf,
    }


def classify_route(entry_boundary: str, exit_boundary: str) -> str:
    if entry_boundary != REQUIRED_ENTRY_BOUNDARY:
        return ""
    return EXIT_TO_ROUTE.get(exit_boundary, "")

def point_to_segment_distance_sq(
    point: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> float:
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end

    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy

    if denom <= 1e-9:
        return (px - x1) ** 2 + (py - y1) ** 2

    t = ((px - x1) * dx + (py - y1) * dy) / denom
    t = max(0.0, min(1.0, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return (px - closest_x) ** 2 + (py - closest_y) ** 2


def mean_distance_to_boundary_sq(
    points: list[tuple[float, float]],
    boundary_segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> float:
    if not points or not boundary_segments:
        return float("inf")

    distances = []
    for point in points:
        best = min(
            point_to_segment_distance_sq(point, seg_start, seg_end)
            for seg_start, seg_end in boundary_segments
        )
        distances.append(best)

    return sum(distances) / len(distances)


def choose_entry_and_exit(
    crossings: list[dict[str, Any]],
    track_rows: list[dict[str, Any]] | None = None,
    boundaries: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    if not crossings:
        return "", "", []

    entry_crossing = None
    entry_index = None

    for index, crossing in enumerate(crossings):
        if crossing["boundary"] == REQUIRED_ENTRY_BOUNDARY:
            entry_crossing = crossing
            entry_index = index
            break

    if entry_crossing is None or entry_index is None:
        first = crossings[0]
        return first["boundary"], "", [first]

    exit_crossings = [
        crossing
        for crossing in crossings[entry_index + 1:]
        if crossing["boundary"] in EXIT_TO_ROUTE
    ]

    # Important:
    # Do not invent an exit for tracks that never crossed any exit boundary.
    # Otherwise the event count becomes inflated.
    if not exit_crossings:
        return entry_crossing["boundary"], "", [entry_crossing]

    if track_rows is not None and boundaries is not None:
        tail_rows = track_rows[-min(8, len(track_rows)):]
        tail_points = [
            (float(row["center_x"]), float(row["center_y"]))
            for row in tail_rows
        ]

        candidate_distances = {}
        for boundary_name in EXIT_TO_ROUTE:
            if boundary_name in boundaries:
                candidate_distances[boundary_name] = mean_distance_to_boundary_sq(
                    tail_points,
                    boundaries[boundary_name],
                )

        if candidate_distances:
            exit_boundary = min(candidate_distances, key=candidate_distances.get)

            chosen_exit = None
            for crossing in exit_crossings:
                if crossing["boundary"] == exit_boundary:
                    chosen_exit = crossing
                    break

            if chosen_exit is None:
                final_row = track_rows[-1]
                chosen_exit = {
                    "boundary": exit_boundary,
                    "frame_index": final_row["frame_index"],
                    "timestamp_sec": final_row["timestamp_sec"],
                    "segment_index": len(track_rows) - 1,
                    "selection_method": "nearest_final_position",
                }

            return (
                entry_crossing["boundary"],
                exit_boundary,
                [entry_crossing, chosen_exit],
            )

    # Fallback: use the last crossed exit.
    exit_crossing = exit_crossings[-1]

    return (
        entry_crossing["boundary"],
        exit_crossing["boundary"],
        [entry_crossing, exit_crossing],
    )


def build_events(
    tracks: dict[int, list[dict[str, Any]]],
    boundaries: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    debug: dict[str, Any] = {
        "tracks_considered": 0,
        "tracks_rejected_short": 0,
        "tracks_rejected_no_crossings": 0,
        "tracks_rejected_wrong_entry": 0,
        "tracks_rejected_invalid_exit": 0,
        "tracks_accepted": 0,
        "per_track": [],
    }

    for track_id, track_rows in tracks.items():
        debug["tracks_considered"] += 1
        summary = summarise_track(track_rows)
        if summary["num_points"] < MIN_TRACK_POINTS or summary["duration_sec"] < MIN_TRACK_DURATION_SEC:
            debug["tracks_rejected_short"] += 1
            debug["per_track"].append({**summary, "status": "rejected_short"})
            continue

        crossings = find_boundary_crossings(track_rows, boundaries)
        if not crossings:
            debug["tracks_rejected_no_crossings"] += 1
            debug["per_track"].append({**summary, "status": "rejected_no_crossings"})
            continue

        entry_boundary, exit_boundary, chosen_crossings = choose_entry_and_exit(
            crossings,
            track_rows,
            boundaries,
        )
        if DROP_TRACKS_WITHOUT_REQUIRED_ENTRY and entry_boundary != REQUIRED_ENTRY_BOUNDARY:
            debug["tracks_rejected_wrong_entry"] += 1
            debug["per_track"].append(
                {
                    **summary,
                    "status": "rejected_wrong_entry",
                    "entry_boundary": entry_boundary,
                    "exit_boundary": exit_boundary,
                    "all_crossings": crossings,
                    "chosen_crossings": chosen_crossings,
                }
            )
            continue

        route_type = classify_route(entry_boundary, exit_boundary)
        if not route_type:
            debug["tracks_rejected_invalid_exit"] += 1
            debug["per_track"].append(
                {
                    **summary,
                    "status": "rejected_invalid_exit",
                    "entry_boundary": entry_boundary,
                    "exit_boundary": exit_boundary,
                    "all_crossings": crossings,
                    "chosen_crossings": chosen_crossings,
                }
            )
            continue

        event = {
            **summary,
            "entry_zone": entry_boundary,
            "exit_zone": exit_boundary,
            "route_type": route_type,
            "first_crossing_time_sec": chosen_crossings[0]["timestamp_sec"],
            "last_crossing_time_sec": chosen_crossings[-1]["timestamp_sec"],
            "num_all_crossings": len(crossings),
            "num_chosen_crossings": len(chosen_crossings),
        }
        events.append(event)
        debug["tracks_accepted"] += 1
        debug["per_track"].append(
            {
                **event,
                "status": "accepted",
                "all_crossings": crossings,
                "chosen_crossings": chosen_crossings,
            }
        )

    events.sort(key=lambda item: (item["start_frame"], item["track_id"]))
    return events, debug


def write_events_csv(path: Path, events: list[dict[str, Any]]) -> None:
    fieldnames = [
        "video_name",
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
        "num_all_crossings",
        "num_chosen_crossings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)


def write_debug_json(path: Path, debug: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(debug, handle, indent=2)


def main() -> None:
    tracks_path = Path(TRACKS_CSV_PATH).expanduser()
    scene_path = Path(SCENE_REGIONS_JSON_PATH).expanduser()
    events_path = Path(EVENTS_CSV_PATH).expanduser()
    debug_path = Path(EVENTS_DEBUG_JSON_PATH).expanduser()

    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_path}")
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene regions JSON not found: {scene_path}")

    scene = load_json(scene_path)
    tracks = load_tracks_csv(tracks_path)
    boundaries = extract_boundaries(scene)

    required = {REQUIRED_ENTRY_BOUNDARY, *EXIT_TO_ROUTE.keys()}
    missing = [name for name in required if name not in boundaries]
    if missing:
        raise RuntimeError(f"Missing boundary centerlines in scene config: {missing}")

    events, debug = build_events(tracks, boundaries)
    write_events_csv(events_path, events)
    write_debug_json(debug_path, debug)

    print(f"Tracks loaded: {len(tracks)}")
    print(f"Boundaries loaded: {sorted(boundaries.keys())}")
    print(f"Accepted events: {len(events)}")
    print(f"Events CSV: {events_path}")
    print(f"Debug JSON: {debug_path}")

    if events:
        print()
        print("First few events:")
        for event in events[:10]:
            print(
                f"track_id={event['track_id']} "
                f"entry={event['entry_zone']} "
                f"exit={event['exit_zone']} "
                f"route={event['route_type']} "
                f"t=({event['start_time_sec']:.2f}-{event['end_time_sec']:.2f})"
            )
    else:
        print()
        print("No valid events extracted yet. Check EVENTS_DEBUG_JSON_PATH for rejection reasons.")


if __name__ == "__main__":
    main()


from utils.base import PipelineContext, PipelineStage

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable


def _list_track_csvs(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root] if root.name == "tracks.csv" else []
    pattern = "**/tracks.csv" if recursive else "*/tracks.csv"
    return sorted(p for p in root.glob(pattern) if p.is_file())


class EventExtractionPipeline(PipelineStage):
    def run(self) -> None:
        global TRACKS_CSV_PATH, SCENE_REGIONS_JSON_PATH, EVENTS_CSV_PATH, EVENTS_DEBUG_JSON_PATH
        global REQUIRED_ENTRY_BOUNDARY, EXIT_TO_ROUTE, MIN_TRACK_POINTS, MIN_TRACK_DURATION_SEC, MIN_CROSSING_FRAME_GAP, DROP_TRACKS_WITHOUT_REQUIRED_ENTRY

        SCENE_REGIONS_JSON_PATH = str(self.context.scene_regions_json)
        REQUIRED_ENTRY_BOUNDARY = self.context.required_entry_boundary
        EXIT_TO_ROUTE = dict(self.context.exit_to_route)
        MIN_TRACK_POINTS = self.context.min_track_points
        MIN_TRACK_DURATION_SEC = self.context.min_track_duration_sec
        MIN_CROSSING_FRAME_GAP = self.context.min_crossing_frame_gap
        DROP_TRACKS_WITHOUT_REQUIRED_ENTRY = self.context.drop_tracks_without_required_entry

        if self.context.event_tracks_csv is not None:
            track_files = [self.context.event_tracks_csv]
        else:
            track_files = _list_track_csvs(self.context.tracking_output_root, self.context.recursive)

        if not track_files:
            raise FileNotFoundError(f"No tracks.csv files found under: {self.context.tracking_output_root}")

        self.logger.info("Starting event extraction for {} track files.", len(track_files))
        for tracks_path in tqdm(track_files, desc="Event extraction", unit="file"):
            TRACKS_CSV_PATH = str(tracks_path)
            EVENTS_CSV_PATH = str(tracks_path.with_name(self.context.event_output_csv.name))
            EVENTS_DEBUG_JSON_PATH = str(tracks_path.with_name(self.context.event_debug_json.name))
            main()
