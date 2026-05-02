#!/usr/bin/env python3
from __future__ import annotations

from custom_logger import CustomLogger

from main import CONFIG
from utils.scene_config import SceneConfigBuilder
from utils.tracking import YoloTrackingPipeline
from utils.video_preparation import VideoPreparationPipeline


logger = CustomLogger(__name__)


def set_extraction_stage_flags() -> None:
    """
    Configure the pipeline to run only the data extraction half.

    This half starts from the configured input videos and stops after YOLO tracking.
    Expected main outputs include metadata, standardised clips, scene regions,
    tracking_outputs/*/tracks.csv, and tracking_outputs/*/summary.json.

    Annotated YOLO video output is controlled by the config key:
        write_annotated_video
    """
    CONFIG.run_video_preparation = True
    CONFIG.run_scene_config_export = True
    CONFIG.run_tracking = True
    CONFIG.run_event_extraction = False
    CONFIG.run_event_merge = False


def log_annotated_video_setting() -> None:
    if CONFIG.write_annotated_video:
        logger.info(
            "Annotated YOLO videos are enabled. Codec: {}.",
            CONFIG.annotated_video_codec,
        )
    else:
        logger.info("Annotated YOLO videos are disabled by config.")


def main() -> None:
    set_extraction_stage_flags()

    logger.info("Starting extraction pipeline")
    log_annotated_video_setting()

    logger.info("Running video preparation")
    VideoPreparationPipeline(CONFIG).run()

    logger.info("Running scene config export")
    SceneConfigBuilder(CONFIG).run()

    logger.info("Running YOLO tracking")
    YoloTrackingPipeline(CONFIG).run()

    logger.info("Extraction pipeline finished")


if __name__ == "__main__":
    main()
