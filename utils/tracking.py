from __future__ import annotations

import cv2
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

from utils.base import PipelineStage
from utils.io_utils import read_csv


class YoloTrackingPipeline(PipelineStage):
    def run(self) -> None:
        clips = read_csv(self.config.clip_manifest_csv_path)
        if not clips:
            raise FileNotFoundError(f'No clips found in {self.config.clip_manifest_csv_path}')

        self.logger.info(f'Loading YOLO model: {self.config.model_weights}')
        model = YOLO(self.config.model_weights)
        self.logger.info(f'Found {len(clips)} clip(s) for tracking on device={self.config.device}')

        clip_bar = tqdm(clips, desc='Tracking clips', unit='clip')
        for clip in clip_bar:
            clip_path = clip['clip_path']
            clip_id = clip['clip_id']
            clip_bar.set_postfix_str(clip_id)
            output_dir = self.config.tracking_output_root / clip_id
            tracks_csv = output_dir / 'tracks.csv'
            annotated_video = output_dir / 'annotated.mp4'

            if tracks_csv.exists() and self.config.skip_if_output_exists and not self.config.overwrite_existing_tracking:
                self.logger.info(f'Skipping existing tracking output for {clip_id}')
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(clip_path))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

            writer = None
            if self.config.write_annotated_video:
                fourcc = cv2.VideoWriter_fourcc(*self.config.annotated_video_codec)
                writer = cv2.VideoWriter(str(annotated_video), fourcc, fps if fps > 0 else 10.0, (width, height))

            rows: list[dict] = []
            results = model.track(
                source=str(clip_path),
                stream=True,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.img_size,
                tracker=self.config.tracker_config,
                persist=self.config.persist_tracks,
                classes=self.config.target_classes,
                device=self.config.device,
                verbose=False,
            )

            frame_bar = tqdm(total=total_frames if total_frames > 0 else None, desc=f'Frames {clip_id}', unit='frame', leave=False)
            for frame_idx, result in enumerate(results):
                frame_bar.update(1)
                boxes = result.boxes
                orig = result.orig_img
                if boxes is None:
                    if writer is not None and orig is not None:
                        writer.write(orig)
                    continue

                ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                classes = boxes.cls.cpu().tolist() if boxes.cls is not None else [None] * len(boxes)
                confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(boxes)
                xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []

                for track_id, cls_id, conf, box in zip(ids, classes, confs, xyxy):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    rows.append({
                        'video_id': clip['video_id'],
                        'clip_id': clip_id,
                        'source_path': clip['source_path'],
                        'clip_path': str(clip_path),
                        'frame_idx': frame_idx,
                        'timestamp_sec': frame_idx / fps if fps > 0 else 0.0,
                        'track_id': int(track_id) if track_id is not None else -1,
                        'class_id': int(cls_id) if cls_id is not None else -1,
                        'confidence': float(conf) if conf is not None else 0.0,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'width': x2 - x1,
                        'height': y2 - y1,
                    })

                if writer is not None:
                    frame = result.plot()
                    writer.write(frame)

            frame_bar.close()
            if writer is not None:
                writer.release()

            pd.DataFrame(rows).to_csv(tracks_csv, index=False)
            self.logger.info(f'Tracking finished for {clip_id}: {len(rows)} detection row(s) written to {tracks_csv}')
