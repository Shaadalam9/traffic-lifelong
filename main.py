#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from utils.config import PipelineConfig
from utils.event_extraction import EventExtractionPipeline
from utils.event_merge import EventTableMerger
from utils.scene_config import SceneConfigBuilder
from utils.tracking import YoloTrackingPipeline
from utils.video_preparation import VideoPreparationPipeline


CONFIG = PipelineConfig(
    project_root=Path("."),
    input_path=Path("raw_videos"),
    annotations_xml=Path("annotations.xml"),
    output_root=Path("_output"),
    run_video_preparation=True,
    run_scene_config_export=True,
    run_tracking=True,
    run_event_extraction=True,
    run_event_merge=True,
    model_weights="yolo11s.pt",
    device="cpu",
    confidence_threshold=0.70,
    tracker_config="bytetrack.yaml",
)


def main() -> None:
    CONFIG.ensure_output_directories()

    if CONFIG.run_video_preparation:
        VideoPreparationPipeline(CONFIG).run()

    if CONFIG.run_scene_config_export:
        SceneConfigBuilder(CONFIG).run()

    if CONFIG.run_tracking:
        YoloTrackingPipeline(CONFIG).run()

    if CONFIG.run_event_extraction:
        EventExtractionPipeline(CONFIG).run()

    if CONFIG.run_event_merge:
        EventTableMerger(CONFIG).run()


if __name__ == "__main__":
    main()
