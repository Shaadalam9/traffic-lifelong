from __future__ import annotations

from utils.base import LegacyScriptRunner
from utils.config import PipelineConfig


class EventTableMerger(LegacyScriptRunner):
    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def run(self) -> None:
        self.config.ensure_output_directories()
        self.run_legacy_script(
            "merge_event_tables.py",
            {
                "EVENTS_ROOT": self.config.merge_events_root,
                "CLIP_MANIFEST_CSV": self.config.clip_manifest_csv_path,
                "VIDEO_INVENTORY_CSV": self.config.inventory_csv_path,
                "VIDEO_TIME_BOUNDS_CSV": self.config.time_bounds_csv_path,
                "MASTER_EVENTS_CSV": self.config.merge_master_csv,
                "MASTER_EVENTS_JSON": self.config.merge_master_json,
                "RECURSIVE": self.config.recursive,
                "EVENTS_FILENAME": self.config.merge_events_filename,
                "SKIP_EMPTY_EVENTS": self.config.skip_empty_events,
            },
        )
