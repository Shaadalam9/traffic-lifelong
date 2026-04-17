from __future__ import annotations

from utils.base import LegacyScriptRunner
from utils.config import PipelineConfig


class VideoPreparationPipeline(LegacyScriptRunner):
    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def run(self) -> None:
        self.config.ensure_output_directories()
        self.run_legacy_script(
            "analysis.py",
            {
                "INPUT_PATH": self.config.input_path,
                "PROJECT_ROOT": self.config.project_root,
                "RECURSIVE": self.config.recursive,
                "INVENTORY_CSV_PATH": self.config.inventory_csv_path,
                "TIME_BOUNDS_CSV_PATH": self.config.time_bounds_csv_path,
                "CLIP_MANIFEST_CSV_PATH": self.config.clip_manifest_csv_path,
                "SCENE_FRAME_MANIFEST_CSV_PATH": self.config.scene_frame_manifest_csv_path,
                "STANDARDIZED_VIDEO_DIR": self.config.standardized_video_dir,
                "PREVIEW_DIR": self.config.preview_dir,
                "SCENE_FRAME_DIR": self.config.scene_frame_dir,
                "RUN_OCR_TIME_BOUNDS": self.config.run_ocr_time_bounds,
                "RUN_VIDEO_INVENTORY": self.config.run_video_inventory,
                "RUN_STANDARDIZE_AND_SPLIT": self.config.run_standardize_and_split,
                "RUN_PREVIEW_CLIPS": self.config.run_preview_clips,
                "RUN_SCENE_FRAME_SAMPLING": self.config.run_scene_frame_sampling,
                "CLIP_DURATION_SECONDS": self.config.clip_duration_seconds,
                "PREVIEW_DURATION_SECONDS": self.config.preview_duration_seconds,
                "OVERWRITE_EXISTING_OUTPUTS": self.config.overwrite_existing_outputs,
                "KEEP_AUDIO": self.config.keep_audio,
                "TARGET_CONTAINER_SUFFIX": self.config.target_container_suffix,
                "TARGET_VIDEO_CODEC": self.config.target_video_codec,
                "TARGET_PRESET": self.config.target_preset,
                "TARGET_CRF": self.config.target_crf,
                "TARGET_FPS": self.config.target_fps,
                "TARGET_WIDTH": self.config.target_width,
                "TARGET_HEIGHT": self.config.target_height,
                "PIXEL_FORMAT": self.config.pixel_format,
                "DEDUPE_OVERLAP_TOLERANCE_SECONDS": self.config.dedupe_overlap_tolerance_seconds,
                "MIN_OUTPUT_SEGMENT_SECONDS": self.config.min_output_segment_seconds,
                "SKIP_VIDEOS_WITHOUT_TRUSTED_TIME_RANGE": self.config.skip_videos_without_trusted_time_range,
                "CLEAN_STALE_TEMP_OUTPUTS_ON_START": self.config.clean_stale_temp_outputs_on_start,
                "TEMP_OUTPUT_SUFFIX": self.config.temp_output_suffix,
                "SCENE_FRAME_SAMPLE_RATIOS": self.config.scene_frame_sample_ratios,
                "SCENE_FRAME_JPEG_QUALITY": self.config.scene_frame_jpeg_quality,
                "CROP_X": self.config.crop_x,
                "CROP_Y": self.config.crop_y,
                "CROP_W": self.config.crop_w,
                "CROP_H": self.config.crop_h,
                "FRAME_OFFSETS": self.config.frame_offsets,
                "THRESHOLDS": self.config.thresholds,
                "OCR_TIMEOUT_SECONDS": self.config.ocr_timeout_seconds,
                "TESSERACT_CONFIGS": self.config.tesseract_configs,
                "TIME_ALIGNMENT_TOLERANCE_SECONDS": self.config.time_alignment_tolerance_seconds,
            },
        )
