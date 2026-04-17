from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.base import PipelineStage
from utils.geometry_utils import bbox_center, point_to_polyline_distance
from utils.io_utils import read_json, write_json


class EventExtractionPipeline(PipelineStage):
    def run(self) -> None:
        scene = read_json(self.config.scene_regions_json)
        regions = scene.get("regions", {})
        track_files = sorted(self.config.tracking_output_root.rglob("tracks.csv"))
        if not track_files:
            raise FileNotFoundError("No tracks.csv files found under tracking outputs")

        for tracks_csv in track_files:
            events_csv = tracks_csv.parent / "events.csv"
            debug_json = tracks_csv.parent / "events_debug.json"

            if events_csv.exists() and self.config.skip_if_output_exists and not self.config.overwrite_existing_tracking:
                self.logger.info(f"Skipping existing events for {tracks_csv.parent.name}")
                continue

            df = pd.read_csv(tracks_csv)
            if df.empty:
                pd.DataFrame([]).to_csv(events_csv, index=False)
                write_json(debug_json, {"events": [], "reason": "empty_tracks"})
                continue

            rows = []
            debug = {"events": []}

            for track_id, group in df.groupby("track_id"):
                group = group.sort_values("frame_idx")
                if len(group) < self.config.min_track_points:
                    continue

                start_sec = float(group["timestamp_sec"].min())
                end_sec = float(group["timestamp_sec"].max())
                duration_sec = end_sec - start_sec
                if duration_sec < self.config.min_track_duration_sec:
                    continue

                first = group.iloc[0]
                last = group.iloc[-1]
                start_pt = bbox_center(first["x1"], first["y1"], first["x2"], first["y2"])
                end_pt = bbox_center(last["x1"], last["y1"], last["x2"], last["y2"])

                entry_boundary, entry_distance = self._nearest_boundary(start_pt, regions)
                exit_boundary, exit_distance = self._nearest_boundary(end_pt, regions)

                if self.config.drop_tracks_without_required_entry and entry_boundary != self.config.required_entry_boundary:
                    continue

                route_type = self.config.exit_to_route.get(exit_boundary, "unknown")
                mean_conf = float(group["confidence"].mean())
                mean_width = float(group["width"].mean())
                mean_height = float(group["height"].mean())

                row = {
                    "event_id": f"{first['clip_id']}_{int(track_id)}",
                    "video_id": first["video_id"],
                    "clip_id": first["clip_id"],
                    "track_id": int(track_id),
                    "start_frame": int(group["frame_idx"].min()),
                    "end_frame": int(group["frame_idx"].max()),
                    "start_time_sec": round(start_sec, 3),
                    "end_time_sec": round(end_sec, 3),
                    "duration_sec": round(duration_sec, 3),
                    "vehicle_class_id": int(first["class_id"]),
                    "mean_detection_confidence": round(mean_conf, 4),
                    "entry_zone": entry_boundary,
                    "exit_zone": exit_boundary,
                    "route_type": route_type,
                    "mean_bbox_width": round(mean_width, 3),
                    "mean_bbox_height": round(mean_height, 3),
                    "bbox_aspect_ratio": round(mean_width / mean_height, 4) if mean_height else 0.0,
                    "track_points": int(len(group)),
                    "source_tracks_csv": str(tracks_csv),
                }
                rows.append(row)
                debug["events"].append({
                    "event_id": row["event_id"],
                    "entry_boundary": entry_boundary,
                    "entry_distance": entry_distance,
                    "exit_boundary": exit_boundary,
                    "exit_distance": exit_distance,
                })

            pd.DataFrame(rows).to_csv(events_csv, index=False)
            write_json(debug_json, debug)

    def _nearest_boundary(self, point: tuple[float, float], regions: dict) -> tuple[str, float]:
        best_name = "unknown"
        best_dist = float("inf")
        for name, region in regions.items():
            if "boundary" not in name:
                continue
            if region.get("type") not in {"polyline", "polygon", "box"}:
                continue
            points = region.get("points", [])
            if region.get("type") == "box" and len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                box_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
                dist = point_to_polyline_distance(point, box_points)
            else:
                dist = point_to_polyline_distance(point, points)
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name, float(best_dist)
