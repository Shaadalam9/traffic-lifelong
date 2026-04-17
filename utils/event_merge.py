from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.base import PipelineStage
from utils.io_utils import write_json


class EventTableMerger(PipelineStage):
    def run(self) -> None:
        event_files = sorted(self.config.merge_events_root.rglob(self.config.merge_events_filename))
        frames = []
        per_file_counts = []

        for path in event_files:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty and self.config.skip_empty_events:
                continue
            frames.append(df)
            per_file_counts.append({"path": str(path), "rows": int(len(df))})

        if frames:
            master = pd.concat(frames, ignore_index=True)
        else:
            master = pd.DataFrame([])

        self.config.merge_master_csv.parent.mkdir(parents=True, exist_ok=True)
        master.to_csv(self.config.merge_master_csv, index=False)

        summary = {
            "event_files_found": len(event_files),
            "event_files_used": len(frames),
            "master_rows": int(len(master)),
            "per_file_counts": per_file_counts,
        }
        write_json(self.config.merge_master_json, summary)
