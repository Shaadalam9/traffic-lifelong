from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from utils.base import PipelineStage
from utils.io_utils import read_csv
from utils.path_utils import safe_stem

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


class YoloTrackingPipeline(PipelineStage):
    def run(self) -> None:
        if YOLO is None:
            raise ImportError("Ultralytics is not installed. Please install ultralytics first.")

        clips = read_csv(self.config.clip_manifest_csv_path)
        if not clips:
            raise RuntimeError("Clip manifest is empty. Run video preparation first.")

        model = YOLO(self.config.model_weights)

        for clip in clips:
            clip_path = Path(clip["clip_path"])
            clip_id = clip["clip_id"]
            output_dir = self.config.tracking_output_root / clip_id
            tracks_csv = output_dir / "tracks.csv"
            annotated_video = output_dir / "annotated.mp4"

            if tracks_csv.exists() and self.config.skip_if_output_exists and not self.config.overwrite_existing_tracking:
                self.logger.info(f"Skipping existing tracking output for {clip_id}")
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(clip_path))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

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

            for frame_idx, result in enumerate(results):
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
                        "video_id": clip["video_id"],
                        "clip_id": clip_id,
                        "source_path": clip["source_path"],
                        "clip_path": str(clip_path),
                        "frame_idx": frame_idx,
                        "timestamp_sec": frame_idx / fps if fps > 0 else 0.0,
                        "track_id": int(track_id) if track_id is not None else -1,
                        "class_id": int(cls_id) if cls_id is not None else -1,
                        "confidence": float(conf) if conf is not None else 0.0,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": x2 - x1,
                        "height": y2 - y1,
                    })

                if writer is not None:
                    frame = result.plot()
                    writer.write(frame)

            cap.release()
            if writer is not None:
                writer.release()

            pd.DataFrame(rows).to_csv(tracks_csv, index=False)
