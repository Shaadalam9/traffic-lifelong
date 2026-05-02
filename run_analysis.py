#!/usr/bin/env python3
from __future__ import annotations

from custom_logger import CustomLogger

from main import CONFIG
from utils.event_extraction import EventExtractionPipeline
from utils.event_merge import EventTableMerger


logger = CustomLogger(__name__)


def set_analysis_stage_flags() -> None:
    """
    Configure the pipeline to run only the post YOLO analysis half.

    This half assumes YOLO tracking has already produced tracking_outputs/*/tracks.csv.
    It converts tracks into events and then merges those event tables into the
    configured master event table.

    Annotated YOLO videos are not produced here because tracking does not run in
    the analysis half.
    """
    CONFIG.run_video_preparation = False
    CONFIG.run_scene_config_export = False
    CONFIG.run_tracking = False
    CONFIG.run_event_extraction = True
    CONFIG.run_event_merge = True

    # This setting matters only when tracking runs, but keeping it false here
    # makes the analysis runner explicitly post YOLO and output table only.
    CONFIG.write_annotated_video = False


def main() -> None:
    set_analysis_stage_flags()

    logger.info("Starting analysis pipeline")
    logger.info("Annotated YOLO videos are disabled in the analysis runner.")

    logger.info("Running event extraction")
    EventExtractionPipeline(CONFIG).run()

    logger.info("Running event merge")
    EventTableMerger(CONFIG).run()

    logger.info("Analysis pipeline finished")


if __name__ == "__main__":
    main()
