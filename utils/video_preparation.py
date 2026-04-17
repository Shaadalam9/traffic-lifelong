from __future__ import annotations

from tqdm.auto import tqdm

from utils.base import PipelineStage
from utils.io_utils import write_csv
from utils.path_utils import list_video_files, safe_stem
from utils.video_utils import copy_video, get_video_metadata, sample_frame, write_preview


class VideoPreparationPipeline(PipelineStage):
    def run(self) -> None:
        videos = list_video_files(self.config.input_path, recursive=self.config.recursive)
        if not videos:
            raise FileNotFoundError(f"No video files found under {self.config.input_path}")

        self.logger.info(f"Found {len(videos)} video file(s) for preparation")

        inventory_rows = []
        time_bounds_rows = []
        clip_rows = []
        scene_rows = []

        video_bar = tqdm(videos, desc='Preparing videos', unit='video')
        for video_path in video_bar:
            video_bar.set_postfix_str(video_path.name)
            meta = get_video_metadata(video_path)
            video_id = safe_stem(video_path)
            standardized_path = self.config.standardized_video_dir / f"{video_id}{video_path.suffix.lower()}"
            preview_path = self.config.preview_dir / f"{video_id}_preview.mp4"

            if not standardized_path.exists() or self.config.overwrite_existing_outputs:
                copy_video(video_path, standardized_path)

            if not preview_path.exists() or self.config.overwrite_existing_outputs:
                try:
                    write_preview(standardized_path, preview_path, seconds=self.config.preview_duration_seconds)
                except Exception as exc:
                    self.logger.warning(f"Preview generation failed for {video_path}: {exc}")

            inventory_rows.append({
                'video_id': video_id,
                'file_name': video_path.name,
                'source_path': str(video_path),
                'standardized_path': str(standardized_path),
                'duration_sec': round(meta['duration_sec'], 3),
                'fps': round(meta['fps'], 3),
                'width': meta['width'],
                'height': meta['height'],
                'timestamp_overlay_visible': '',
                'location_overlay_visible': '',
            })

            time_bounds_rows.append({
                'video_id': video_id,
                'source_path': str(video_path),
                'trusted_time_start': '',
                'trusted_time_end': '',
                'status': 'placeholder_no_ocr',
            })

            clip_rows.append({
                'video_id': video_id,
                'clip_id': video_id,
                'clip_path': str(standardized_path),
                'source_path': str(video_path),
            })

            ratio_bar = tqdm(
                self.config.scene_frame_sample_ratios,
                desc=f'Scene frames for {video_path.name}',
                unit='frame',
                leave=False,
            )
            for ratio in ratio_bar:
                ratio_bar.set_postfix_str(str(ratio))
                frame_name = f"{video_id}_{str(ratio).replace('.', '_')}.jpg"
                frame_path = self.config.scene_frame_dir / frame_name
                ok = sample_frame(standardized_path, ratio, frame_path, self.config.scene_frame_jpeg_quality)
                if ok:
                    scene_rows.append({
                        'video_id': video_id,
                        'ratio': ratio,
                        'frame_path': str(frame_path),
                    })

        write_csv(self.config.inventory_csv_path, inventory_rows)
        write_csv(self.config.time_bounds_csv_path, time_bounds_rows)
        write_csv(self.config.clip_manifest_csv_path, clip_rows)
        write_csv(self.config.scene_frame_manifest_csv_path, scene_rows)

        self.logger.info(
            f"Video preparation finished: {len(inventory_rows)} inventory rows, "
            f"{len(clip_rows)} clip rows, {len(scene_rows)} scene frames"
        )
