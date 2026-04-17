from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    project_root: Path = Path(".")
    input_path: Path = Path("raw_videos")
    annotations_xml: Path = Path("annotations.xml")
    output_root: Path = Path("_output")
    recursive: bool = True

    run_video_preparation: bool = True
    run_scene_config_export: bool = True
    run_tracking: bool = True
    run_event_extraction: bool = True
    run_event_merge: bool = True

    legacy_dir_name: str = "legacy"
    scene_image_name: str | None = None
    scene_round_digits: int = 2

    model_weights: str = "yolo11s.pt"
    device: str | int = "cpu"
    img_size: int = 1280
    confidence_threshold: float = 0.70
    iou_threshold: float = 0.45
    tracker_config: str = "bytetrack.yaml"
    target_classes: list[int] = field(default_factory=lambda: [2, 3, 5, 7])
    persist_tracks: bool = True
    write_annotated_video: bool = True
    annotated_video_codec: str = "mp4v"
    write_frame_previews: bool = False
    frame_preview_every_n: int = 150
    skip_if_output_exists: bool = True
    overwrite_existing_tracking: bool = False

    required_entry_boundary: str = "boundary_bottom"
    exit_to_route: dict[str, str] = field(default_factory=lambda: {
        "boundary_far_left": "left",
        "boundary_far_center": "straight",
        "boundary_far_right": "right",
    })
    min_track_points: int = 5
    min_track_duration_sec: float = 0.75
    min_crossing_frame_gap: int = 3
    drop_tracks_without_required_entry: bool = True

    merge_events_filename: str = "events.csv"
    skip_empty_events: bool = True

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

    scene_frame_sample_ratios: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])
    scene_frame_jpeg_quality: int = 95

    crop_x: float = 0.015
    crop_y: float = 0.020
    crop_w: float = 0.310
    crop_h: float = 0.080
    frame_offsets: list[int] = field(default_factory=lambda: [0, 2, 5, 10])
    thresholds: list[int] = field(default_factory=lambda: [140, 170, 200, 225])
    ocr_timeout_seconds: float = 1.5
    tesseract_configs: list[str] = field(default_factory=lambda: [
        "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:/- APMapm",
        "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:/- APMapm",
    ])
    time_alignment_tolerance_seconds: float = 5.0

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self.input_path = Path(self.input_path)
        self.annotations_xml = Path(self.annotations_xml)
        self.output_root = Path(self.output_root)

    @property
    def legacy_root(self) -> Path:
        return self.project_root / self.legacy_dir_name

    @property
    def metadata_dir(self) -> Path:
        return self.output_root / "metadata"

    @property
    def scene_dir(self) -> Path:
        return self.output_root / "scene"

    @property
    def standardized_video_dir(self) -> Path:
        return self.output_root / "standardized_videos"

    @property
    def preview_dir(self) -> Path:
        return self.metadata_dir / "previews"

    @property
    def scene_frame_dir(self) -> Path:
        return self.output_root / "frames_for_scene_setup"

    @property
    def tracking_output_root(self) -> Path:
        return self.output_root / "tracking_outputs"

    @property
    def tracking_input_path(self) -> Path:
        return self.standardized_video_dir

    @property
    def scene_regions_json(self) -> Path:
        return self.scene_dir / "scene_regions.json"

    @property
    def inventory_csv_path(self) -> Path:
        return self.metadata_dir / "video_inventory.csv"

    @property
    def time_bounds_csv_path(self) -> Path:
        return self.metadata_dir / "video_time_bounds.csv"

    @property
    def clip_manifest_csv_path(self) -> Path:
        return self.metadata_dir / "clip_manifest.csv"

    @property
    def scene_frame_manifest_csv_path(self) -> Path:
        return self.metadata_dir / "scene_frame_manifest.csv"

    @property
    def merge_events_root(self) -> Path:
        return self.tracking_output_root

    @property
    def event_tables_dir(self) -> Path:
        return self.output_root / "event_tables"

    @property
    def merge_master_csv(self) -> Path:
        return self.event_tables_dir / "master_events.csv"

    @property
    def merge_master_json(self) -> Path:
        return self.event_tables_dir / "master_events_summary.json"

    def ensure_output_directories(self) -> None:
        directories = [
            self.output_root,
            self.metadata_dir,
            self.scene_dir,
            self.standardized_video_dir,
            self.preview_dir,
            self.scene_frame_dir,
            self.tracking_output_root,
            self.event_tables_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def discover_track_csv_files(self) -> list[Path]:
        if not self.tracking_output_root.exists():
            return []
        return sorted(self.tracking_output_root.rglob("tracks.csv"))

    def event_output_csv_for_tracks(self, tracks_csv: Path) -> Path:
        return tracks_csv.with_name("events.csv")

    def event_debug_json_for_tracks(self, tracks_csv: Path) -> Path:
        return tracks_csv.with_name("events_debug.json")
