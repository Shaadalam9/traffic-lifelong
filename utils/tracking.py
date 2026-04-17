from __future__ import annotations

from utils.base import LegacyScriptRunner
from utils.config import PipelineConfig


class YoloTrackingPipeline(LegacyScriptRunner):
    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def run(self) -> None:
        self.config.ensure_output_directories()
        self.run_legacy_script(
            "run_yolo_tracking.py",
            {
                "INPUT_PATH": self.config.tracking_input_path,
                "OUTPUT_ROOT": self.config.tracking_output_root,
                "RECURSIVE": self.config.recursive,
                "MODEL_WEIGHTS": self.config.model_weights,
                "DEVICE": self.config.device,
                "IMG_SIZE": self.config.img_size,
                "CONFIDENCE_THRESHOLD": self.config.confidence_threshold,
                "IOU_THRESHOLD": self.config.iou_threshold,
                "TRACKER_CONFIG": self.config.tracker_config,
                "TARGET_CLASSES": self.config.target_classes,
                "PERSIST_TRACKS": self.config.persist_tracks,
                "WRITE_ANNOTATED_VIDEO": self.config.write_annotated_video,
                "ANNOTATED_VIDEO_CODEC": self.config.annotated_video_codec,
                "WRITE_FRAME_PREVIEWS": self.config.write_frame_previews,
                "FRAME_PREVIEW_EVERY_N": self.config.frame_preview_every_n,
                "SKIP_IF_OUTPUT_EXISTS": self.config.skip_if_output_exists,
                "OVERWRITE_EXISTING": self.config.overwrite_existing_tracking,
            },
        )
