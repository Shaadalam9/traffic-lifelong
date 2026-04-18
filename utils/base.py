from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from custom_logger import CustomLogger


@dataclass
class PipelineContext:
    project_root: Path = Path(".")
    input_path: Path = Path(".")
    annotations_xml: Path = Path("annotations.xml")

    run_video_preparation: bool = False
    run_scene_config_export: bool = False
    run_tracking: bool = False
    run_event_extraction: bool = False
    run_event_merge: bool = False

    scene_regions_json: Path = Path("_output/scene/scene_regions.json")
    scene_image_name: str | None = None
    scene_round_digits: int = 2

    tracking_input_path: Path | None = None
    tracking_output_root: Path = Path("_output/tracking_outputs")
    model_weights: str = "yolo11s.pt"
    device: str | int = "cpu"
    img_size: int = 1280
    confidence_threshold: float = 0.70
    iou_threshold: float = 0.45
    tracker_config: str = "bytetrack.yaml"
    target_classes: list[int] | None = None
    persist_tracks: bool = True

    write_annotated_video: bool = True
    annotated_video_codec: str = "mp4v"
    write_frame_previews: bool = False
    frame_preview_every_n: int = 150

    skip_if_output_exists: bool = True
    overwrite_existing_tracking: bool = False

    event_tracks_csv: Path | None = None
    event_output_csv: Path = Path("events.csv")
    event_debug_json: Path = Path("events_debug.json")
    required_entry_boundary: str = "boundary_bottom"
    exit_to_route: dict[str, str] | None = None
    min_track_points: int = 5
    min_track_duration_sec: float = 0.75
    min_crossing_frame_gap: int = 3
    drop_tracks_without_required_entry: bool = True

    merge_events_root: Path | None = None
    merge_master_csv: Path = Path("_output/event_tables/master_events.csv")
    merge_master_json: Path = Path("_output/event_tables/master_events_summary.json")
    merge_events_filename: str = "events.csv"
    skip_empty_events: bool = True

    inventory_csv_path: Path = Path("data/video_inventory.csv")
    time_bounds_csv_path: Path = Path("_output/metadata/video_time_bounds.csv")
    clip_manifest_csv_path: Path = Path("_output/metadata/clip_manifest.csv")
    scene_frame_manifest_csv_path: Path = Path("_output/metadata/scene_frame_manifest.csv")
    standardized_video_dir: Path = Path("_output/standardized_videos")
    preview_dir: Path = Path("_output/metadata/previews")
    scene_frame_dir: Path = Path("_output/frames_for_scene_setup")

    recursive: bool = True

    run_ocr_time_bounds: bool = True
    run_video_inventory: bool = True
    run_standardize_and_split: bool = True
    run_preview_clips: bool = True
    run_scene_frame_sampling: bool = True

    clip_duration_seconds: int = 1800
    preview_duration_seconds: int = 20
    overwrite_existing_outputs: bool = False
    keep_audio: bool = False

    target_container_suffix: str = ".mp4"
    target_video_codec: str = "libx264"
    target_preset: str = "medium"
    target_crf: int = 18
    target_fps: float = 10.0
    target_width: int | None = None
    target_height: int | None = None
    pixel_format: str = "yuv420p"

    dedupe_overlap_tolerance_seconds: float = 1.0
    min_output_segment_seconds: float = 1.0
    skip_videos_without_trusted_time_range: bool = True
    clean_stale_temp_outputs_on_start: bool = True
    temp_output_suffix: str = ".tmp"

    scene_frame_sample_ratios: list[float] | None = None
    scene_frame_jpeg_quality: int = 95

    crop_x: float = 0.015
    crop_y: float = 0.020
    crop_w: float = 0.310
    crop_h: float = 0.080

    frame_offsets: list[int] | None = None
    thresholds: list[int] | None = None
    ocr_timeout_seconds: float = 1.5
    tesseract_configs: list[str] | None = None
    time_alignment_tolerance_seconds: float = 5.0

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self.input_path = Path(self.input_path)
        self.annotations_xml = Path(self.annotations_xml)
        self.scene_regions_json = Path(self.scene_regions_json)
        self.tracking_output_root = Path(self.tracking_output_root)
        self.inventory_csv_path = Path(self.inventory_csv_path)
        self.time_bounds_csv_path = Path(self.time_bounds_csv_path)
        self.clip_manifest_csv_path = Path(self.clip_manifest_csv_path)
        self.scene_frame_manifest_csv_path = Path(self.scene_frame_manifest_csv_path)
        self.standardized_video_dir = Path(self.standardized_video_dir)
        self.preview_dir = Path(self.preview_dir)
        self.scene_frame_dir = Path(self.scene_frame_dir)
        self.event_output_csv = Path(self.event_output_csv)
        self.event_debug_json = Path(self.event_debug_json)
        self.merge_master_csv = Path(self.merge_master_csv)
        self.merge_master_json = Path(self.merge_master_json)

        if self.tracking_input_path is not None:
            self.tracking_input_path = Path(self.tracking_input_path)
        else:
            self.tracking_input_path = self.standardized_video_dir

        if self.event_tracks_csv is not None:
            self.event_tracks_csv = Path(self.event_tracks_csv)

        if self.merge_events_root is not None:
            self.merge_events_root = Path(self.merge_events_root)
        else:
            self.merge_events_root = self.tracking_output_root

        if self.target_classes is None:
            self.target_classes = [2, 3, 5, 7]

        if self.exit_to_route is None:
            self.exit_to_route = {
                "boundary_far_left": "left",
                "boundary_far_center": "straight",
                "boundary_far_right": "right",
            }

        if self.scene_frame_sample_ratios is None:
            self.scene_frame_sample_ratios = [0.05, 0.25, 0.50, 0.75, 0.95]

        if self.frame_offsets is None:
            self.frame_offsets = [0, 2, 5, 10]

        if self.thresholds is None:
            self.thresholds = [140, 170, 200, 225]

        if self.tesseract_configs is None:
            self.tesseract_configs = [
                "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:/- APMapm",
                "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:/- APMapm",
            ]


class PipelineStage:
    def __init__(self, context: PipelineContext) -> None:
        self.context = context
        self.logger = CustomLogger(self.__class__.__name__)
