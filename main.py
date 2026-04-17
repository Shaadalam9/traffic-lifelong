#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import common
from custom_logger import CustomLogger
from logmod import logs

from utils.base import PipelineContext
from utils.event_extraction import EventExtractionPipeline
from utils.event_merge import EventTableMerger
from utils.scene_config import SceneConfigBuilder
from utils.tracking import YoloTrackingPipeline
from utils.video_preparation import VideoPreparationPipeline

logs(show_level=common.get_configs('logger_level'), show_color=True)
logger = CustomLogger(__name__)

CONFIG = PipelineContext(
    project_root=Path('.'),
    input_path=Path(common.get_configs('input_path')),
    annotations_xml=Path(common.get_configs('annotations_xml')),
    output_root=Path(common.get_configs('output_root')),
    run_video_preparation=bool(common.get_configs('run_video_preparation')),
    run_scene_config_export=bool(common.get_configs('run_scene_config_export')),
    run_tracking=bool(common.get_configs('run_tracking')),
    run_event_extraction=bool(common.get_configs('run_event_extraction')),
    run_event_merge=bool(common.get_configs('run_event_merge')),
    model_weights=str(common.get_configs('model_weights')),
    device=common.get_configs('device'),
    confidence_threshold=float(common.get_configs('confidence_threshold')),
    tracker_config=str(common.get_configs('tracker_config')),
)


def main() -> None:
    logger.info('Starting traffic lifelong pipeline')
    CONFIG.ensure_directories()

    if CONFIG.run_video_preparation:
        logger.info('Running video preparation')
        VideoPreparationPipeline(CONFIG).run()

    if CONFIG.run_scene_config_export:
        logger.info('Running scene config export')
        SceneConfigBuilder(CONFIG).run()

    if CONFIG.run_tracking:
        logger.info('Running tracking')
        YoloTrackingPipeline(CONFIG).run()

    if CONFIG.run_event_extraction:
        logger.info('Running event extraction')
        EventExtractionPipeline(CONFIG).run()

    if CONFIG.run_event_merge:
        logger.info('Running event merge')
        EventTableMerger(CONFIG).run()

    logger.info('Pipeline finished')


if __name__ == '__main__':
    main()
