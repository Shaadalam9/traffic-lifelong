from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from custom_logger import CustomLogger


@dataclass
class PipelineContext:
    project_root: Path
    input_path: Path
    annotations_xml: Path
    output_root: Path = Path('_output')

    run_video_preparation: bool = True
    run_scene_config_export: bool = True
    run_tracking: bool = True
    run_event_extraction: bool = True
    run_event_merge: bool = True

    recursive: bool = True

    model_weights: str = 'yolo11s.pt'
    device: str | int = 'cpu'
    img_size: int = 1280
    confidence_threshold: float = 0.70
    iou_threshold: float = 0.45
    tracker_config: str = 'bytetrack.yaml'
    target_classes: list[int] = field(default_factory=lambda: [2, 3, 5, 7])
    persist_tracks: bool = True
    write_annotated_video: bool = True
    annotated_video_codec: str = 'mp4v'
    write_frame_previews: bool = False
    frame_preview_every_n: int = 150
    skip_if_output_exists: bool = True
    overwrite_existing_tracking: bool = False

    scene_image_name: str | None = None
    scene_round_digits: int = 2

    required_entry_boundary: str = 'boundary_bottom'
    exit_to_route: dict[str, str] = field(default_factory=lambda: {
        'boundary_far_left': 'left',
        'boundary_far_center': 'straight',
        'boundary_far_right': 'right',
    })
    min_track_points: int = 5
    min_track_duration_sec: float = 0.75
    min_crossing_frame_gap: int = 3
    drop_tracks_without_required_entry: bool = True

    clip_duration_seconds: int = 1800
    preview_duration_seconds: int = 20
    overwrite_existing_outputs: bool = False
    keep_audio: bool = False

    target_container_suffix: str = '.mp4'
    target_video_codec: str = 'libx264'
    target_preset: str = 'medium'
    target_crf: int = 18
    target_fps: float = 10.0
    target_width: int | None = None
    target_height: int | None = None
    pixel_format: str = 'yuv420p'

    scene_frame_sample_ratios: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])
    scene_frame_jpeg_quality: int = 95

    @property
    def metadata_dir(self) -> Path:
        return self.output_root / 'metadata'

    @property
    def inventory_csv_path(self) -> Path:
        return self.metadata_dir / 'video_inventory.csv'

    @property
    def time_bounds_csv_path(self) -> Path:
        return self.metadata_dir / 'video_time_bounds.csv'

    @property
    def clip_manifest_csv_path(self) -> Path:
        return self.metadata_dir / 'clip_manifest.csv'

    @property
    def scene_frame_manifest_csv_path(self) -> Path:
        return self.metadata_dir / 'scene_frame_manifest.csv'

    @property
    def standardized_video_dir(self) -> Path:
        return self.output_root / 'standardized_videos'

    @property
    def preview_dir(self) -> Path:
        return self.metadata_dir / 'previews'

    @property
    def scene_frame_dir(self) -> Path:
        return self.output_root / 'frames_for_scene_setup'

    @property
    def scene_dir(self) -> Path:
        return self.output_root / 'scene'

    @property
    def scene_regions_json(self) -> Path:
        return self.scene_dir / 'scene_regions.json'

    @property
    def tracking_output_root(self) -> Path:
        return self.output_root / 'tracking_outputs'

    @property
    def merge_events_root(self) -> Path:
        return self.tracking_output_root

    @property
    def merge_master_csv(self) -> Path:
        return self.output_root / 'event_tables' / 'master_events.csv'

    @property
    def merge_master_json(self) -> Path:
        return self.output_root / 'event_tables' / 'master_events_summary.json'

    @property
    def merge_events_filename(self) -> str:
        return 'events.csv'

    @property
    def skip_empty_events(self) -> bool:
        return True

    def ensure_directories(self) -> None:
        for path in [
            self.output_root,
            self.metadata_dir,
            self.standardized_video_dir,
            self.preview_dir,
            self.scene_frame_dir,
            self.scene_dir,
            self.tracking_output_root,
            self.output_root / 'event_tables',
        ]:
            path.mkdir(parents=True, exist_ok=True)


class PipelineStage:
    def __init__(self, config: PipelineContext) -> None:
        self.config = config
        self.logger = CustomLogger(self.__class__.__name__)

    def ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
