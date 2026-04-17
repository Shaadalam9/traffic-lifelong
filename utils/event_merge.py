from __future__ import annotations

import pandas as pd
from tqdm.auto import tqdm

from utils.base import PipelineStage
from utils.io_utils import write_json


class EventTableMerger(PipelineStage):
    def run(self) -> None:
        event_files = sorted(self.config.merge_events_root.rglob(self.config.merge_events_filename))
        frames = []
        per_file_counts = []

        self.logger.info(f'Found {len(event_files)} event file(s) for merge')
        file_bar = tqdm(event_files, desc='Merging event tables', unit='file')
        for path in file_bar:
            file_bar.set_postfix_str(path.parent.name)
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                self.logger.warning(f'Failed to read {path}: {exc}')
                continue
            if df.empty and self.config.skip_empty_events:
                continue
            frames.append(df)
            per_file_counts.append({'path': str(path), 'rows': int(len(df))})

        if frames:
            master = pd.concat(frames, ignore_index=True)
        else:
            master = pd.DataFrame([])

        self.config.merge_master_csv.parent.mkdir(parents=True, exist_ok=True)
        master.to_csv(self.config.merge_master_csv, index=False)

        summary = {
            'event_files_found': len(event_files),
            'event_files_used': len(frames),
            'master_rows': int(len(master)),
            'per_file_counts': per_file_counts,
        }
        write_json(self.config.merge_master_json, summary)
        self.logger.info(
            f'Merge finished: {len(frames)} file(s) used, {len(master)} total row(s) written to {self.config.merge_master_csv}'
        )
