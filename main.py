#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import common
from custom_logger import CustomLogger
from logmod import logs
from tqdm.auto import tqdm

from utils.base import PipelineContext
from utils.event_extraction import EventExtractionPipeline
from utils.event_merge import EventTableMerger
from utils.scene_config import SceneConfigBuilder
from utils.tracking import YoloTrackingPipeline
from utils.video_preparation import VideoPreparationPipeline

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

CONFIG = PipelineContext(
    project_root=Path('.'),
    input_path=Path(common.get_configs("input_path")),
    annotations_xml=Path(common.get_configs("annotations_xml")),
    output_root=Path(common.get_configs("output_root")),
    run_video_preparation=bool(common.get_configs("run_video_preparation")),
    run_scene_config_export=bool(common.get_configs("run_scene_config_export")),
    run_tracking=bool(common.get_configs("run_tracking")),
    run_event_extraction=bool(common.get_configs("run_event_extraction")),
    run_event_merge=bool(common.get_configs("run_event_merge")),
    model_weights=str(common.get_configs("model_weights")),
    device=common.get_configs("device"),
    confidence_threshold=float(common.get_configs("confidence_threshold")),
    tracker_config=str(common.get_configs("tracker_config")),
)


def main() -> None:
    logger.info('Starting traffic lifelong pipeline')
    CONFIG.ensure_directories()

    stages: list[tuple[str, object]] = []
    if CONFIG.run_video_preparation:
        stages.append(('video preparation', VideoPreparationPipeline(CONFIG)))
    if CONFIG.run_scene_config_export:
        stages.append(('scene config export', SceneConfigBuilder(CONFIG)))
    if CONFIG.run_tracking:
        stages.append(('tracking', YoloTrackingPipeline(CONFIG)))
    if CONFIG.run_event_extraction:
        stages.append(('event extraction', EventExtractionPipeline(CONFIG)))
    if CONFIG.run_event_merge:
        stages.append(('event merge', EventTableMerger(CONFIG)))

    if not stages:
        logger.warning('No stages are enabled in config')
        return

    stage_bar = tqdm(stages, desc='Pipeline stages', unit='stage')
    for stage_name, stage_runner in stage_bar:
        stage_bar.set_postfix_str(stage_name)
        logger.info(f'Running {stage_name}')
        stage_runner.run()
        logger.info(f'Finished {stage_name}')

    logger.info('Pipeline finished')


if __name__ == '__main__':
    main()
