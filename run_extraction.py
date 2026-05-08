#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import common
from custom_logger import CustomLogger
from logmod import logs

from utils.base import PipelineContext
from utils.scene_config import SceneConfigBuilder
from utils.tracking import YoloTrackingPipeline
from utils.video_preparation import VideoPreparationPipeline


def cfg(key: str, default=None):
    try:
        value = common.get_configs(key)
    except Exception:
        return default
    return default if value is None else value


def cfg_path(key: str, default=None) -> Path | None:
    value = cfg(key, default)
    if value in (None, ""):
        return None
    return Path(value)


def build_config() -> PipelineContext:
    project_root = cfg_path("project_root", ".")
    if project_root is None:
        raise ValueError("project_root is missing from config")

    output_root = cfg_path("output_root", "_output")
    if output_root is None:
        raise ValueError("output_root is missing from config")

    return PipelineContext(
        project_root=project_root,
        output_root=output_root,
        input_path=cfg_path("input_path", "."),
        annotations_xml=cfg_path("annotations_xml", "annotations.xml"),

        run_video_preparation=True,
        run_scene_config_export=True,
        run_tracking=True,
        run_event_extraction=False,
        run_event_merge=False,

        scene_regions_json=cfg_path("scene_regions_json", output_root / "scene" / "scene_regions.json"),
        scene_image_name=cfg("scene_image_name", None),
        scene_round_digits=cfg("scene_round_digits", 2),

        tracking_input_path=cfg_path("tracking_input_path", None),
        tracking_output_root=cfg_path("tracking_output_root", output_root / "tracking_outputs"),
        model_weights=cfg("model_weights", "yolo11s.pt"),
        device=cfg("device", "cpu"),
        img_size=cfg("img_size", 1280),
        confidence_threshold=cfg("confidence_threshold", 0.70),
        iou_threshold=cfg("iou_threshold", 0.45),
        tracker_config=cfg("tracker_config", "bytetrack.yaml"),
        target_classes=cfg("target_classes", [2, 3, 5, 7]),
        persist_tracks=cfg("persist_tracks", True),

        write_annotated_video=cfg("write_annotated_video", True),
        annotated_video_codec=cfg("annotated_video_codec", "mp4v"),
        write_frame_previews=cfg("write_frame_previews", False),
        frame_preview_every_n=cfg("frame_preview_every_n", 150),

        skip_if_output_exists=cfg("skip_if_output_exists", True),
        overwrite_existing_tracking=cfg("overwrite_existing_tracking", False),

        event_tracks_csv=cfg_path("event_tracks_csv", None),
        event_output_csv=cfg_path("event_output_csv", "events.csv"),
        event_debug_json=cfg_path("event_debug_json", "events_debug.json"),
        required_entry_boundary=cfg("required_entry_boundary", "boundary_bottom"),
        exit_to_route=cfg(
            "exit_to_route",
            {
                "boundary_far_left": "left",
                "boundary_far_center": "straight",
                "boundary_far_right": "right",
            },
        ),
        min_track_points=cfg("min_track_points", 5),
        min_track_duration_sec=cfg("min_track_duration_sec", 0.75),
        min_crossing_frame_gap=cfg("min_crossing_frame_gap", 3),
        drop_tracks_without_required_entry=cfg("drop_tracks_without_required_entry", True),

        merge_events_root=cfg_path("merge_events_root", None),
        merge_master_csv=cfg_path("merge_master_csv", output_root / "event_tables" / "master_events.csv"),
        merge_master_json=cfg_path("merge_master_json", output_root / "event_tables" / "master_events_summary.json"),
        merge_events_filename=cfg("merge_events_filename", "events.csv"),
        skip_empty_events=cfg("skip_empty_events", True),

        inventory_csv_path=cfg_path("inventory_csv_path", output_root / "metadata" / "video_inventory.csv"),
        time_bounds_csv_path=cfg_path("time_bounds_csv_path", output_root / "metadata" / "video_time_bounds.csv"),
        clip_manifest_csv_path=cfg_path("clip_manifest_csv_path", output_root / "metadata" / "clip_manifest.csv"),
        scene_frame_manifest_csv_path=cfg_path("scene_frame_manifest_csv_path", output_root / "metadata" / "scene_frame_manifest.csv"),
        standardized_video_dir=cfg_path("standardized_video_dir", output_root / "standardized_videos"),
        preview_dir=cfg_path("preview_dir", output_root / "metadata" / "previews"),
        scene_frame_dir=cfg_path("scene_frame_dir", output_root / "frames_for_scene_setup"),

        recursive=cfg("recursive", True),

        run_ocr_time_bounds=cfg("run_ocr_time_bounds", True),
        run_video_inventory=cfg("run_video_inventory", True),
        run_standardize_and_split=cfg("run_standardize_and_split", True),
        run_preview_clips=cfg("run_preview_clips", True),
        run_scene_frame_sampling=cfg("run_scene_frame_sampling", True),

        clip_duration_seconds=cfg("clip_duration_seconds", 1800),
        preview_duration_seconds=cfg("preview_duration_seconds", 20),
        overwrite_existing_outputs=cfg("overwrite_existing_outputs", False),
        keep_audio=cfg("keep_audio", False),

        target_container_suffix=cfg("target_container_suffix", ".mp4"),
        target_video_codec=cfg("target_video_codec", "libx264"),
        target_preset=cfg("target_preset", "medium"),
        target_crf=cfg("target_crf", 18),
        target_fps=cfg("target_fps", 10.0),
        target_width=cfg("target_width", None),
        target_height=cfg("target_height", None),
        pixel_format=cfg("pixel_format", "yuv420p"),

        dedupe_overlap_tolerance_seconds=cfg("dedupe_overlap_tolerance_seconds", 1.0),
        min_output_segment_seconds=cfg("min_output_segment_seconds", 1.0),
        skip_videos_without_trusted_time_range=cfg("skip_videos_without_trusted_time_range", True),
        clean_stale_temp_outputs_on_start=cfg("clean_stale_temp_outputs_on_start", True),
        temp_output_suffix=cfg("temp_output_suffix", ".tmp"),

        scene_frame_sample_ratios=cfg("scene_frame_sample_ratios", [0.05, 0.25, 0.50, 0.75, 0.95]),
        scene_frame_jpeg_quality=cfg("scene_frame_jpeg_quality", 95),

        crop_x=cfg("crop_x", 0.015),
        crop_y=cfg("crop_y", 0.020),
        crop_w=cfg("crop_w", 0.310),
        crop_h=cfg("crop_h", 0.080),

        frame_offsets=cfg("frame_offsets", [0, 2, 5, 10]),
        thresholds=cfg("thresholds", [140, 170, 200, 225]),
        ocr_timeout_seconds=cfg("ocr_timeout_seconds", 1.5),
        tesseract_configs=cfg(
            "tesseract_configs",
            [
                "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:/- APMapm",
                "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:/- APMapm",
            ],
        ),
        time_alignment_tolerance_seconds=cfg("time_alignment_tolerance_seconds", 5.0),
    )


def main() -> None:
    logs(show_level=cfg("logger_level", "info"), show_color=True)
    logger = CustomLogger(__name__)
    config = build_config()

    logger.info("Starting extraction pipeline")
    if config.write_annotated_video:
        logger.info("Annotated YOLO videos are enabled. Codec: {}.", config.annotated_video_codec)
    else:
        logger.info("Annotated YOLO videos are disabled by config.")

    logger.info("Running video preparation")
    VideoPreparationPipeline(config).run()

    logger.info("Running scene config export")
    SceneConfigBuilder(config).run()

    logger.info("Running YOLO tracking")
    YoloTrackingPipeline(config).run()

    logger.info("Extraction pipeline finished")


if __name__ == "__main__":
    main()
