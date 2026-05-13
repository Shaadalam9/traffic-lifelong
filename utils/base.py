from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from custom_logger import CustomLogger


@dataclass
class PipelineContext:
    project_root: Path
    output_root: Path
    input_path: Path
    annotations_xml: Path

    run_video_preparation: bool
    run_scene_config_export: bool
    run_tracking: bool
    run_event_extraction: bool
    run_event_merge: bool

    scene_regions_json: Path
    scene_image_name: str | None
    scene_round_digits: int

    tracking_input_path: Path | None
    tracking_output_root: Path
    model_weights: str
    device: str | int
    img_size: int
    confidence_threshold: float
    iou_threshold: float
    tracker_config: str
    target_classes: list[int]
    persist_tracks: bool

    write_annotated_video: bool
    annotated_video_codec: str
    write_frame_previews: bool
    frame_preview_every_n: int

    skip_if_output_exists: bool
    overwrite_existing_tracking: bool

    event_tracks_csv: Path | None
    event_output_csv: Path
    event_debug_json: Path
    required_entry_boundary: str
    exit_to_route: dict[str, str]
    min_track_points: int
    min_track_duration_sec: float
    min_crossing_frame_gap: int
    drop_tracks_without_required_entry: bool

    merge_events_root: Path | None
    merge_master_csv: Path
    merge_master_json: Path
    merge_events_filename: str
    skip_empty_events: bool

    inventory_csv_path: Path
    time_bounds_csv_path: Path
    clip_manifest_csv_path: Path
    scene_frame_manifest_csv_path: Path
    standardized_video_dir: Path
    preview_dir: Path
    scene_frame_dir: Path

    recursive: bool

    run_ocr_time_bounds: bool
    run_video_inventory: bool
    run_standardize_and_split: bool
    run_preview_clips: bool
    run_scene_frame_sampling: bool

    clip_duration_seconds: int
    preview_duration_seconds: int
    overwrite_existing_outputs: bool
    keep_audio: bool

    target_container_suffix: str
    target_video_codec: str
    target_preset: str
    target_crf: int
    target_fps: float
    target_width: int | None
    target_height: int | None
    pixel_format: str

    dedupe_overlap_tolerance_seconds: float
    min_output_segment_seconds: float
    skip_videos_without_trusted_time_range: bool
    clean_stale_temp_outputs_on_start: bool
    temp_output_suffix: str

    scene_frame_sample_ratios: list[float]
    scene_frame_jpeg_quality: int

    # Used by mitigation video generation to mask the visible timestamp/overlay area.
    crop_x: float
    crop_y: float
    crop_w: float
    crop_h: float

    time_alignment_tolerance_seconds: float

    def _resolve_path(self, value: Path | str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.output_root = self._resolve_path(self.output_root)
        self.input_path = self._resolve_path(self.input_path)
        self.annotations_xml = self._resolve_path(self.annotations_xml)

        self.scene_regions_json = self._resolve_path(self.scene_regions_json)
        self.tracking_output_root = self._resolve_path(self.tracking_output_root)

        self.inventory_csv_path = self._resolve_path(self.inventory_csv_path)
        self.time_bounds_csv_path = self._resolve_path(self.time_bounds_csv_path)
        self.clip_manifest_csv_path = self._resolve_path(self.clip_manifest_csv_path)
        self.scene_frame_manifest_csv_path = self._resolve_path(self.scene_frame_manifest_csv_path)
        self.standardized_video_dir = self._resolve_path(self.standardized_video_dir)
        self.preview_dir = self._resolve_path(self.preview_dir)
        self.scene_frame_dir = self._resolve_path(self.scene_frame_dir)

        self.event_output_csv = Path(self.event_output_csv)
        self.event_debug_json = Path(self.event_debug_json)

        self.merge_master_csv = self._resolve_path(self.merge_master_csv)
        self.merge_master_json = self._resolve_path(self.merge_master_json)

        if self.tracking_input_path is not None:
            self.tracking_input_path = self._resolve_path(self.tracking_input_path)
        else:
            self.tracking_input_path = self.standardized_video_dir

        if self.event_tracks_csv is not None:
            self.event_tracks_csv = self._resolve_path(self.event_tracks_csv)

        if self.merge_events_root is not None:
            self.merge_events_root = self._resolve_path(self.merge_events_root)
        else:
            self.merge_events_root = self.tracking_output_root


class PipelineStage:
    def __init__(self, context: PipelineContext) -> None:
        self.context = context
        self.logger = CustomLogger(self.__class__.__name__)
