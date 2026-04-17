from __future__ import annotations

from pathlib import Path

from utils.base import LegacyScriptRunner
from utils.config import PipelineConfig


class EventExtractionPipeline(LegacyScriptRunner):
    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def _run_for_tracks_file(self, tracks_csv: Path) -> None:
        event_output_csv = self.config.event_output_csv_for_tracks(tracks_csv)
        event_debug_json = self.config.event_debug_json_for_tracks(tracks_csv)
        event_output_csv.parent.mkdir(parents=True, exist_ok=True)

        self.run_legacy_script(
            "extract_events_from_tracks.py",
            {
                "TRACKS_CSV_PATH": tracks_csv,
                "SCENE_REGIONS_JSON_PATH": self.config.scene_regions_json,
                "EVENTS_CSV_PATH": event_output_csv,
                "EVENTS_DEBUG_JSON_PATH": event_debug_json,
                "REQUIRED_ENTRY_BOUNDARY": self.config.required_entry_boundary,
                "EXIT_TO_ROUTE": self.config.exit_to_route,
                "MIN_TRACK_POINTS": self.config.min_track_points,
                "MIN_TRACK_DURATION_SEC": self.config.min_track_duration_sec,
                "MIN_CROSSING_FRAME_GAP": self.config.min_crossing_frame_gap,
                "DROP_TRACKS_WITHOUT_REQUIRED_ENTRY": self.config.drop_tracks_without_required_entry,
            },
        )

    def run(self) -> None:
        self.config.ensure_output_directories()
        track_csv_files = self.config.discover_track_csv_files()
        if not track_csv_files:
            raise FileNotFoundError(
                f"No tracks.csv files found under {self.config.tracking_output_root}"
            )

        for tracks_csv in track_csv_files:
            self._run_for_tracks_file(tracks_csv)
