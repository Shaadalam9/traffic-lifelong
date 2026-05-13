#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
import common
import pandas as pd
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import cv2
from tqdm import tqdm
from logmod import logs
from custom_logger import CustomLogger
from utils.base import PipelineContext
from utils import event_extraction as event_extraction_module
from utils import event_merge as event_merge_module


# ============================================================
# Edit only these switches when needed
# ============================================================
CONFIG_PATH = Path("config")

RUN_EVENT_EXTRACTION = True
RUN_EVENT_MERGE = True
RUN_BASIC_STATISTICAL_RESULTS = True
RUN_PRIVACY_FEATURE_ENRICHMENT = False
RUN_ENRICHED_PRIVACY_RESULTS = False
RUN_MANUAL_REVIEW_SAMPLE = False

# Set this True to regenerate HTML, PNG, and EPS figures from existing CSV/JSON outputs
# without rerunning event extraction, merging, enrichment, or the full analysis pipeline.
RUN_REGENERATE_EXISTING_PLOTS = False

# Leave this False for normal analysis runs. Set True only when you want
# mitigation video variants generated from representative clips.
RUN_MITIGATION_VIDEO_GENERATION = False

# Colour extraction reads video frames and is slower than size or motion features.
RUN_COLOUR_EXTRACTION = False

# Mitigation settings
MAX_MITIGATION_SOURCE_CLIPS = 20
MITIGATION_VARIANTS = [
    "overlay_masked",
    "low_resolution",
    "low_frame_rate",
    "low_resolution_low_frame_rate",
    "heavy_downsample",
]

# Analysis settings
TOP_N = 25
LOW_CONFUSABILITY_THRESHOLD = 5
RARE_RECURRENCE_MAX_EVENTS = 20
MANUAL_REVIEW_SAMPLE_PER_GROUP = 25
RANDOM_SEED = 42

# Feature settings
COLOUR_CROP_SHRINK_FRACTION = 0.20
MIN_COLOUR_CROP_PIXELS = 100
MIN_MEAN_VALUE_FOR_COLOUR = 35
DURATION_BUCKETS_SECONDS = [5, 10, 20, 40]
SPEED_BUCKET_QUANTILES = [0.25, 0.50, 0.75]
SIZE_BUCKET_QUANTILES = [0.25, 0.50, 0.75]

logger = CustomLogger(__name__)


# ============================================================
# Config helpers
# ============================================================
def load_json_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


_JSON_CONFIG = load_json_config()


def cfg(key: str, default: Any = None) -> Any:
    if common is not None:
        try:
            value = common.get_configs(key)
            return default if value is None else value
        except Exception:
            pass
    return _JSON_CONFIG.get(key, default)


def cfg_bool(key: str, default: bool) -> bool:
    value = cfg(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def cfg_path(key: str, default: Any = None) -> Path | None:
    value = cfg(key, default)
    if value in (None, ""):
        return None
    return Path(value).expanduser()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_parent(path)
    frame.to_csv(path, index=False)


def path_to_posix(path: Path) -> str:
    return path.as_posix()


def create_context() -> PipelineContext:
    project_root = cfg_path("project_root", ".")
    if project_root is None:
        raise ValueError("project_root is missing from config")

    output_root = cfg_path("output_root", "_output")
    if output_root is None:
        raise ValueError("output_root is missing from config")

    context = PipelineContext(
        project_root=project_root,
        output_root=output_root,
        input_path=cfg_path("input_path", "."),
        annotations_xml=cfg_path("annotations_xml", "annotations.xml"),

        run_video_preparation=False,
        run_scene_config_export=False,
        run_tracking=False,
        run_event_extraction=True,
        run_event_merge=True,

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

        write_annotated_video=False,
        annotated_video_codec=cfg("annotated_video_codec", "mp4v"),
        write_frame_previews=cfg("write_frame_previews", False),
        frame_preview_every_n=cfg("frame_preview_every_n", 150),

        skip_if_output_exists=cfg("skip_if_output_exists", True),
        overwrite_existing_tracking=cfg("overwrite_existing_tracking", False),

        event_tracks_csv=cfg_path("event_tracks_csv", None),
        event_output_csv=cfg_path("event_output_csv", "events.csv"),
        event_debug_json=cfg_path("event_debug_json", "events_debug.json"),
        required_entry_boundary=cfg("required_entry_boundary", "boundary_bottom"),
        exit_to_route=cfg("exit_to_route", {
            "boundary_far_left": "left",
            "boundary_far_center": "straight",
            "boundary_far_right": "right",
        }),
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
        scene_frame_manifest_csv_path=cfg_path("scene_frame_manifest_csv_path", output_root / "metadata" / "scene_frame_manifest.csv"),  # noqa: E501
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
        tesseract_configs=cfg("tesseract_configs", [
            "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:/- APMapm",
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:/- APMapm",
        ]),
        time_alignment_tolerance_seconds=cfg("time_alignment_tolerance_seconds", 5.0),
    )
    return context


def analysis_root_from_context(context: PipelineContext) -> Path:
    configured = cfg_path("analysis_output_root", None)
    if configured is not None:
        return configured.resolve()
    return context.output_root / "analysis_results"


# ============================================================
# Event extraction and merge
# ============================================================
def list_track_csvs(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root] if root.name == "tracks.csv" else []
    if not root.exists():
        return []
    pattern = "**/tracks.csv" if recursive else "*/tracks.csv"
    return sorted(p for p in root.glob(pattern) if p.is_file())


def read_json_maybe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def run_event_extraction(context: PipelineContext) -> dict[str, Any]:
    event_extraction_module.SCENE_REGIONS_JSON_PATH = str(context.scene_regions_json)
    event_extraction_module.REQUIRED_ENTRY_BOUNDARY = context.required_entry_boundary
    event_extraction_module.EXIT_TO_ROUTE = dict(context.exit_to_route)
    event_extraction_module.MIN_TRACK_POINTS = context.min_track_points
    event_extraction_module.MIN_TRACK_DURATION_SEC = context.min_track_duration_sec
    event_extraction_module.MIN_CROSSING_FRAME_GAP = context.min_crossing_frame_gap
    event_extraction_module.DROP_TRACKS_WITHOUT_REQUIRED_ENTRY = context.drop_tracks_without_required_entry

    if context.event_tracks_csv is not None:
        track_files = [context.event_tracks_csv]
    else:
        track_files = list_track_csvs(context.tracking_output_root, context.recursive)

    if not track_files:
        raise FileNotFoundError(f"No tracks.csv files found under: {context.tracking_output_root}")

    logger.info("Found {} tracks.csv files.", len(track_files))

    summary = {
        "track_files_processed": 0,
        "files_with_zero_tracks": 0,
        "files_with_zero_accepted_events": 0,
        "total_tracks_considered": 0,
        "total_events_extracted": 0,
        "tracks_rejected_short": 0,
        "tracks_rejected_no_crossings": 0,
        "tracks_rejected_wrong_entry": 0,
        "tracks_rejected_invalid_exit": 0,
        "files": [],
    }

    for tracks_path in tqdm(track_files, desc="Event extraction", unit="file"):
        events_csv = tracks_path.with_name(context.event_output_csv.name)
        debug_json = tracks_path.with_name(context.event_debug_json.name)

        event_extraction_module.TRACKS_CSV_PATH = str(tracks_path)
        event_extraction_module.EVENTS_CSV_PATH = str(events_csv)
        event_extraction_module.EVENTS_DEBUG_JSON_PATH = str(debug_json)

        # event_extraction.main prints one block per file. Keep tqdm readable by hiding it.
        with contextlib.redirect_stdout(io.StringIO()):
            event_extraction_module.main()

        debug = read_json_maybe(debug_json)
        tracks_considered = int(debug.get("tracks_considered", 0) or 0)
        events_extracted = int(debug.get("tracks_accepted", 0) or 0)
        rejected_short = int(debug.get("tracks_rejected_short", 0) or 0)
        rejected_no_crossings = int(debug.get("tracks_rejected_no_crossings", 0) or 0)
        rejected_wrong_entry = int(debug.get("tracks_rejected_wrong_entry", 0) or 0)
        rejected_invalid_exit = int(debug.get("tracks_rejected_invalid_exit", 0) or 0)

        summary["track_files_processed"] += 1
        summary["files_with_zero_tracks"] += 1 if tracks_considered == 0 else 0
        summary["files_with_zero_accepted_events"] += 1 if events_extracted == 0 else 0
        summary["total_tracks_considered"] += tracks_considered
        summary["total_events_extracted"] += events_extracted
        summary["tracks_rejected_short"] += rejected_short
        summary["tracks_rejected_no_crossings"] += rejected_no_crossings
        summary["tracks_rejected_wrong_entry"] += rejected_wrong_entry
        summary["tracks_rejected_invalid_exit"] += rejected_invalid_exit
        summary["files"].append({
            "tracks_csv": str(tracks_path),
            "events_csv": str(events_csv),
            "events_debug_json": str(debug_json),
            "tracks_considered": tracks_considered,
            "events_extracted": events_extracted,
            "tracks_rejected_short": rejected_short,
            "tracks_rejected_no_crossings": rejected_no_crossings,
            "tracks_rejected_wrong_entry": rejected_wrong_entry,
            "tracks_rejected_invalid_exit": rejected_invalid_exit,
        })

    summary_path = analysis_root_from_context(context) / "event_extraction_run_summary.json"
    write_json(summary_path, summary)

    print("Event extraction summary:")
    print(f"  Track files processed: {summary['track_files_processed']}")
    print(f"  Files with zero tracks: {summary['files_with_zero_tracks']}")
    print(f"  Files with zero accepted events: {summary['files_with_zero_accepted_events']}")
    print(f"  Total tracks considered: {summary['total_tracks_considered']}")
    print(f"  Total events extracted: {summary['total_events_extracted']}")
    print(f"  Saved summary: {summary_path}")
    return summary


def run_event_merge(context: PipelineContext) -> None:
    event_merge_module.EVENTS_ROOT = str(context.merge_events_root)
    event_merge_module.CLIP_MANIFEST_CSV = str(context.clip_manifest_csv_path)
    event_merge_module.VIDEO_INVENTORY_CSV = str(context.inventory_csv_path)
    event_merge_module.VIDEO_TIME_BOUNDS_CSV = str(context.time_bounds_csv_path)
    event_merge_module.MASTER_EVENTS_CSV = str(context.merge_master_csv)
    event_merge_module.MASTER_EVENTS_JSON = str(context.merge_master_json)
    event_merge_module.RECURSIVE = context.recursive
    event_merge_module.EVENTS_FILENAME = context.merge_events_filename
    event_merge_module.SKIP_EMPTY_EVENTS = context.skip_empty_events
    event_merge_module.merge_events()


# ============================================================
# Basic statistical analysis
# ============================================================
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    clean = safe_numeric(series).dropna()
    if clean.empty:
        return {
            "count": 0, "mean": None, "median": None, "std": None, "min": None,
            "p10": None, "p25": None, "p75": None, "p90": None, "p95": None, "max": None,
        }
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "p10": float(clean.quantile(0.10)),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def shannon_entropy(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * probs.map(lambda p: math.log2(float(p)))).sum())


def cramer_v(table: pd.DataFrame) -> float:
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total <= 0:
        return 0.0
    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    mask = expected > 0
    chi2 = float((((observed - expected) ** 2)[mask] / expected[mask]).sum())
    r, k = observed.shape
    denom = total * (min(k - 1, r - 1))
    if denom <= 0:
        return 0.0
    return float(math.sqrt(chi2 / denom))


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["wallclock_start_dt"] = pd.to_datetime(df.get("wallclock_start", pd.Series(dtype=str)), errors="coerce")
    df["date"] = df["wallclock_start_dt"].dt.date.astype("string")
    df.loc[df["wallclock_start_dt"].isna(), "date"] = ""
    df["hour"] = df["wallclock_start_dt"].dt.hour.astype("Int64")
    hour = df["wallclock_start_dt"].dt.hour
    minute = df["wallclock_start_dt"].dt.minute
    half_hour = (hour * 2 + (minute // 30)).astype("Int64")
    df["half_hour_label"] = half_hour.map(format_half_hour_label)
    df.loc[df["wallclock_start_dt"].isna(), "half_hour_label"] = "unknown_time"
    return df


def format_half_hour_label(value: Any) -> str:
    try:
        if pd.isna(value):
            return "unknown_time"
        index = int(value)
    except Exception:
        return "unknown_time"
    return f"{index // 2:02d}:{30 if index % 2 else 0:02d}"


def value_counts_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].fillna("unknown").astype(str).value_counts().reset_index()
    counts.columns = [column, "count"]
    total = int(counts["count"].sum())
    counts["percent"] = counts["count"] / total * 100 if total else 0.0
    return counts


def grouped_numeric_summary(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, group in df.groupby(group_col, dropna=False):
        summary = numeric_summary(group[value_col])
        summary[group_col] = value
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    columns = [group_col, "count", "mean", "median", "std", "min", "p10", "p25", "p75", "p90", "p95", "max"]
    return pd.DataFrame(rows)[columns].sort_values("count", ascending=False)


def final_figures_dir() -> Path:
    configured = cfg_path("final_figures_dir", None)
    if configured is not None:
        return configured.resolve()
    try:
        root_dir = getattr(common, "root_dir", None)
        if root_dir:
            return Path(root_dir).expanduser().resolve() / "figures"
    except Exception:
        pass
    project_root = cfg_path("project_root", ".")
    if project_root is None:
        project_root = Path(".")
    return project_root.expanduser().resolve() / "figures"


def save_plotly_figure(
    fig: go.Figure,
    filename: str,
    output_dir: Path | None = None,
    width: int = 1600,
    height: int = 900,
    scale: int = 1,
    save_final: bool = False,
    save_png: bool = True,
    save_eps: bool = True,
) -> None:
    """
    Save a Plotly figure as HTML, PNG, and EPS in the provided output_dir.

    HTML is always saved. PNG and EPS are optional. By default, files are
    saved only in their original analysis plots folder. Set save_final=True
    only when you also want a copy in <project_root>/figures.

    EPS export can fail with Kaleido on some Linux setups because Plotly
    converts PDF to EPS internally. When that happens, this function tries
    safer fallbacks:
      1. write PDF and convert it with pdftops, if pdftops is installed;
      2. create a raster EPS from the PNG using Pillow.

    The fallback EPS is still suitable for LaTeX workflows that require an
    .eps file, although the Pillow fallback is raster based rather than a
    fully vector EPS.
    """
    if output_dir is None:
        try:
            common_output_dir = getattr(common, "output_dir", None)
            output_dir = Path(common_output_dir).expanduser() if common_output_dir else Path("figures")
        except Exception:
            output_dir = Path("figures")

    output_dir = Path(output_dir).expanduser().resolve()
    output_final = final_figures_dir() if save_final else None

    os.makedirs(output_dir, exist_ok=True)
    if output_final is not None:
        os.makedirs(output_final, exist_ok=True)

    html_path = output_dir / f"{filename}.html"
    logger.info(f"Saving html file for {filename}.")
    py.offline.plot(fig, filename=str(html_path), auto_open=False)

    if save_final and output_final is not None:
        py.offline.plot(fig, filename=str(output_final / f"{filename}.html"), auto_open=False)

    png_path = output_dir / f"{filename}.png"
    eps_path = output_dir / f"{filename}.eps"

    if save_png:
        try:
            logger.info(f"Saving png file for {filename}.")
            fig.write_image(str(png_path), width=width, height=height, scale=scale)
            if save_final and output_final is not None:
                shutil.copy(str(png_path), str(output_final / f"{filename}.png"))
        except Exception as exc:
            logger.error(f"Could not save PNG for {filename}: {exc}")

    if not save_eps:
        return

    try:
        logger.info(f"Saving eps file for {filename}.")
        fig.write_image(str(eps_path), width=width, height=height)
        if save_final and output_final is not None:
            shutil.copy(str(eps_path), str(output_final / f"{filename}.eps"))
        return
    except Exception as exc:
        logger.warning(f"Direct EPS export failed for {filename}: {exc}")

    pdf_path = output_dir / f"{filename}.pdf"
    try:
        pdftops_path = shutil.which("pdftops")
        if pdftops_path is None:
            raise FileNotFoundError("pdftops was not found")

        logger.info(f"Trying PDF to EPS fallback for {filename} using pdftops.")
        fig.write_image(str(pdf_path), width=width, height=height)
        subprocess.run(
            [pdftops_path, "-eps", str(pdf_path), str(eps_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if save_final and output_final is not None:
            shutil.copy(str(eps_path), str(output_final / f"{filename}.eps"))
        return
    except Exception as exc:
        logger.warning(f"PDF to EPS fallback failed for {filename}: {exc}")

    try:
        if not png_path.exists():
            logger.info(f"Creating PNG for raster EPS fallback for {filename}.")
            fig.write_image(str(png_path), width=width, height=height, scale=scale)

        logger.info(f"Trying raster PNG to EPS fallback for {filename} using Pillow.")
        from PIL import Image

        with Image.open(png_path) as image:
            image.convert("RGB").save(eps_path, format="EPS")

        if save_final and output_final is not None:
            shutil.copy(str(eps_path), str(output_final / f"{filename}.eps"))
    except Exception as exc:
        logger.error(f"All EPS export methods failed for {filename}: {exc}")


def figure_font_size() -> int:
    """Read the default figure font size from config."""
    value = cfg("font_size", cfg("figure_font_size", 22))
    try:
        return int(float(value))
    except Exception:
        return 22


def human_readable_text(value: Any) -> str:
    """Convert code style labels such as size_motion_colour into readable labels."""
    text = str(value).strip()
    if not text:
        return text

    explicit_labels = {
        "baseline": "Baseline",
        "size_motion": "Size and motion",
        "size_motion_colour": "Size, motion and colour",
        "unique_signatures": "Unique signatures",
        "singleton_signatures": "Singleton signatures",
        "low_confusability_signatures": "Low confusability signatures",
        "rare_recurrence_candidate_signatures": "Rare recurrence candidates",
        "signature_count": "Signature count",
        "mean_confidence": "Mean confidence",
        "duration_sec": "Duration seconds",
        "route_type": "Route",
        "class_name": "Vehicle class",
        "half_hour_label": "Half hour",
    }
    if text in explicit_labels:
        return explicit_labels[text]

    text = text.replace("_", " ").replace("  ", " ")
    # Preserve common all-caps abbreviations if they appear as full tokens.
    words = []
    for word in text.split(" "):
        if word.upper() in {"ID", "OCR", "FPS", "IOU"}:
            words.append(word.upper())
        else:
            words.append(word.capitalize())
    return " ".join(words)


def clean_axis_label(column: str) -> str:
    return human_readable_text(column)


def make_labels_human_readable(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    readable = frame.copy()
    for column in columns:
        if column in readable.columns and not pd.api.types.is_numeric_dtype(readable[column]):
            readable[column] = readable[column].map(human_readable_text)
    return readable


def apply_common_plot_layout(
    fig: go.Figure,
    x_tickangle: int = 0,
    bottom_margin: int = 70,
    yaxis_title: str | None = None,
) -> None:
    font_size = figure_font_size()
    fig.update_layout(
        template="plotly_white",
        font=dict(size=font_size),
        title_text=None,
        legend=dict(
            title=dict(font=dict(size=font_size)),
            font=dict(size=font_size),
        ),
        margin=dict(l=70, r=30, t=30, b=bottom_margin),
        xaxis=dict(
            title=dict(font=dict(size=font_size)),
            tickfont=dict(size=font_size),
            tickangle=x_tickangle,
        ),
        yaxis=dict(
            title=dict(font=dict(size=font_size)),
            tickfont=dict(size=font_size),
        ),
    )
    if yaxis_title is not None:
        fig.update_yaxes(title_text=yaxis_title)


def plot_bar(frame: pd.DataFrame, x_col: str, y_col: str, path: Path, title: str, rotate: int = 0) -> None:
    if frame.empty:
        return
    ensure_parent(path)
    plot_frame = make_labels_human_readable(frame, [x_col])
    fig = px.bar(
        plot_frame,
        x=x_col,
        y=y_col,
        labels={x_col: clean_axis_label(x_col), y_col: clean_axis_label(y_col)},
    )
    apply_common_plot_layout(
        fig,
        x_tickangle=-rotate if rotate else 0,
        bottom_margin=120 if rotate else 70,
    )
    save_plotly_figure(fig, path.stem, output_dir=path.parent)


def plot_line(frame: pd.DataFrame, x_col: str, y_col: str, path: Path, title: str) -> None:
    if frame.empty:
        return
    ensure_parent(path)
    plot_frame = make_labels_human_readable(frame, [x_col])
    fig = px.line(
        plot_frame,
        x=x_col,
        y=y_col,
        markers=True,
        labels={x_col: clean_axis_label(x_col), y_col: clean_axis_label(y_col)},
    )
    apply_common_plot_layout(fig)
    save_plotly_figure(fig, path.stem, output_dir=path.parent)


def plot_box_by_group(df: pd.DataFrame, group_col: str, value_col: str, path: Path, title: str) -> None:
    clean = df[[group_col, value_col]].copy()
    clean[value_col] = safe_numeric(clean[value_col])
    clean = clean.dropna(subset=[value_col])
    if clean.empty:
        return
    ensure_parent(path)
    clean = make_labels_human_readable(clean, [group_col])
    fig = px.box(
        clean,
        x=group_col,
        y=value_col,
        points=False,
        labels={group_col: clean_axis_label(group_col), value_col: clean_axis_label(value_col)},
    )
    apply_common_plot_layout(fig)
    save_plotly_figure(fig, path.stem, output_dir=path.parent)


def plot_histogram(frame: pd.DataFrame, value_col: str, path: Path, title: str, nbins: int = 50) -> None:
    if frame.empty or value_col not in frame.columns:
        return
    clean = frame.copy()
    clean[value_col] = safe_numeric(clean[value_col])
    clean = clean.dropna(subset=[value_col])
    if clean.empty:
        return
    ensure_parent(path)
    fig = px.histogram(
        clean,
        x=value_col,
        nbins=nbins,
        labels={value_col: clean_axis_label(value_col)},
    )
    apply_common_plot_layout(fig, yaxis_title="Number of signatures")
    save_plotly_figure(fig, path.stem, output_dir=path.parent)


def plot_signature_level_comparison(summary_table: pd.DataFrame, plots_dir: Path) -> None:
    if summary_table.empty:
        return
    plot_data = summary_table.copy()
    if "signature_level" not in plot_data.columns:
        return

    metrics = [
        ("unique_signatures", "Unique signatures"),
        ("singleton_signatures", "Singleton signatures"),
        ("low_confusability_signatures", "Low confusability signatures"),
        ("rare_recurrence_candidate_signatures", "Rare recurrence candidates"),
    ]
    available = [(col, label) for col, label in metrics if col in plot_data.columns]
    if not available:
        return

    long_rows: list[dict[str, Any]] = []
    for _, row in plot_data.iterrows():
        level = human_readable_text(row["signature_level"])
        for col, label in available:
            long_rows.append({
                "signature_level": level,
                "metric": label,
                "value": row.get(col, 0),
            })
    long_df = pd.DataFrame(long_rows)
    fig = px.bar(
        long_df,
        x="signature_level",
        y="value",
        color="metric",
        barmode="group",
        labels={"signature_level": "Signature level", "value": "Count", "metric": "Metric"},
    )
    apply_common_plot_layout(fig, bottom_margin=90)
    save_plotly_figure(fig, "signature_level_comparison", output_dir=plots_dir)


def signature_counts(df: pd.DataFrame, signature_name: str, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = "unknown"
    counts = (
        df.groupby(columns, dropna=False)
        .size()
        .reset_index(name="signature_count")
        .sort_values("signature_count", ascending=False)
    )
    counts.insert(0, "signature_level", signature_name)
    counts["signature"] = counts[columns].astype(str).agg(" | ".join, axis=1)
    return counts


def signature_summary(counts: pd.DataFrame, total_events: int, columns: list[str]) -> dict[str, Any]:
    if counts.empty:
        return {
            "signature_definition": columns,
            "unique_signatures": 0,
            "singleton_signatures": 0,
            "events_in_singleton_signatures": 0,
            "percent_events_in_singleton_signatures": 0.0,
            "low_confusability_signatures": 0,
            "events_in_low_confusability_signatures": 0,
            "percent_events_in_low_confusability_signatures": 0.0,
            "median_events_per_signature": None,
            "mean_events_per_signature": None,
            "max_events_per_signature": None,
            "signature_entropy_bits": 0.0,
        }
    singletons = counts[counts["signature_count"] == 1]
    low = counts[counts["signature_count"] <= LOW_CONFUSABILITY_THRESHOLD]
    events_singleton = int(singletons["signature_count"].sum())
    events_low = int(low["signature_count"].sum())
    return {
        "signature_definition": columns,
        "unique_signatures": int(len(counts)),
        "singleton_signatures": int(len(singletons)),
        "events_in_singleton_signatures": events_singleton,
        "percent_events_in_singleton_signatures": events_singleton / total_events * 100 if total_events else 0.0,
        "low_confusability_signatures": int(len(low)),
        "events_in_low_confusability_signatures": events_low,
        "percent_events_in_low_confusability_signatures": events_low / total_events * 100 if total_events else 0.0,
        "median_events_per_signature": float(counts["signature_count"].median()),
        "mean_events_per_signature": float(counts["signature_count"].mean()),
        "max_events_per_signature": int(counts["signature_count"].max()),
        "signature_entropy_bits": shannon_entropy(counts["signature_count"]),
    }


def run_basic_statistical_results(context: PipelineContext) -> dict[str, Any]:
    master_csv = context.merge_master_csv
    if not master_csv.exists():
        raise FileNotFoundError(f"Master events CSV not found: {master_csv}")

    output_dir = analysis_root_from_context(context)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    df = pd.read_csv(master_csv)
    df = add_time_columns(df)
    for column in ["duration_sec", "mean_confidence", "start_time_sec", "end_time_sec"]:
        if column in df.columns:
            df[column] = safe_numeric(df[column])

    total = len(df)
    route_counts = value_counts_table(df, "route_type") if "route_type" in df.columns else pd.DataFrame()
    class_counts = value_counts_table(df, "class_name") if "class_name" in df.columns else pd.DataFrame()
    events_by_hour = df.dropna(subset=["hour"]).groupby("hour").size().reset_index(name="count")
    events_by_date = df[df["date"] != ""].groupby("date").size().reset_index(name="count")
    route_by_hour = pd.crosstab(df["hour"],
                                df["route_type"]).reset_index() if "route_type" in df.columns else pd.DataFrame()
    route_by_class = pd.crosstab(df["route_type"], df["class_name"]
                                 ) if {"route_type", "class_name"}.issubset(df.columns) else pd.DataFrame()

    duration_stats_by_route = grouped_numeric_summary(df, "route_type", "duration_sec"
                                                      ) if "duration_sec" in df.columns else pd.DataFrame()
    confidence_stats_by_route = grouped_numeric_summary(df, "route_type", "mean_confidence"
                                                        ) if "mean_confidence" in df.columns else pd.DataFrame()
    confidence_stats_by_class = grouped_numeric_summary(df, "class_name", "mean_confidence"
                                                        ) if "mean_confidence" in df.columns else pd.DataFrame()

    baseline_cols = ["class_name", "route_type", "half_hour_label"]
    baseline_counts = signature_counts(df, "baseline", baseline_cols)
    baseline_summary = signature_summary(baseline_counts, total, baseline_cols)

    recurrence = (
        df.groupby(baseline_cols, dropna=False)
        .agg(signature_count=("route_type", "size"), distinct_days=("date", "nunique"))
        .reset_index()
    )
    recurrence = recurrence[recurrence["distinct_days"] >= 2].sort_values(["distinct_days", "signature_count"],
                                                                          ascending=[False, False])
    recurrence["signature"] = recurrence[baseline_cols].astype(str).agg(" | ".join, axis=1)

    write_csv(tables_dir / "route_counts.csv", route_counts)
    write_csv(tables_dir / "class_counts.csv", class_counts)
    write_csv(tables_dir / "events_by_hour.csv", events_by_hour)
    write_csv(tables_dir / "events_by_date.csv", events_by_date)
    write_csv(tables_dir / "route_by_hour.csv", route_by_hour)
    write_csv(tables_dir / "route_by_class.csv", route_by_class.reset_index() if not route_by_class.empty else route_by_class)  # noqa: E501
    write_csv(tables_dir / "duration_statistics_by_route.csv", duration_stats_by_route)
    write_csv(tables_dir / "confidence_statistics_by_route.csv", confidence_stats_by_route)
    write_csv(tables_dir / "confidence_statistics_by_class.csv", confidence_stats_by_class)
    write_csv(tables_dir / "confusability_signatures.csv", baseline_counts)
    write_csv(tables_dir / "top_confusable_signatures.csv", baseline_counts.head(TOP_N))
    write_csv(tables_dir / "recurrence_candidate_signatures.csv", recurrence)

    if "duration_sec" in df.columns:
        write_csv(tables_dir / "top_long_duration_events.csv",
                  df.sort_values("duration_sec", ascending=False).head(100))
    if "mean_confidence" in df.columns:
        write_csv(tables_dir / "low_confidence_events.csv",
                  df.sort_values("mean_confidence", ascending=True).head(100))

    plot_bar(route_counts, "route_type", "count", plots_dir / "route_counts.png", "Route counts")
    plot_bar(class_counts.head(TOP_N), "class_name", "count", plots_dir / "class_counts_top.png", "Class counts",
             rotate=45)
    plot_line(events_by_hour, "hour", "count", plots_dir / "events_by_hour.png", "Events by hour")
    plot_bar(events_by_date, "date", "count", plots_dir / "events_by_date.png", "Events by date", rotate=45)
    plot_box_by_group(df, "route_type", "duration_sec", plots_dir / "duration_by_route_boxplot.png",
                      "Duration by route")
    plot_bar(baseline_counts.head(TOP_N), "signature",
             "signature_count", plots_dir / "top_confusable_signatures.png", "Top baseline signatures", rotate=80)

    summary = {
        "master_events_csv": str(master_csv),
        "analysis_output_dir": str(output_dir),
        "total_events": int(total),
        "events_with_wallclock_time": int(df["wallclock_start_dt"].notna().sum()),
        "percent_with_wallclock_time": float(df["wallclock_start_dt"].notna().mean() * 100) if total else 0.0,
        "unique_videos": int(df["video_id"].nunique()) if "video_id" in df.columns else None,
        "unique_clips": int(df["clip_id"].nunique()) if "clip_id" in df.columns else None,
        "unique_routes": int(df["route_type"].nunique()) if "route_type" in df.columns else None,
        "unique_classes": int(df["class_name"].nunique()) if "class_name" in df.columns else None,
        "first_wallclock_time": str(df["wallclock_start_dt"].min()) if df["wallclock_start_dt"].notna().any() else "",
        "last_wallclock_time": str(df["wallclock_start_dt"].max()) if df["wallclock_start_dt"].notna().any() else "",
        "route_entropy_bits": shannon_entropy(route_counts["count"]) if not route_counts.empty else 0.0,
        "class_entropy_bits": shannon_entropy(class_counts["count"]) if not class_counts.empty else 0.0,
        "route_class_cramers_v": cramer_v(route_by_class) if not route_by_class.empty else 0.0,
        "duration_seconds": numeric_summary(df["duration_sec"]) if "duration_sec" in df.columns else {},
        "mean_confidence": numeric_summary(df["mean_confidence"]) if "mean_confidence" in df.columns else {},
        "confusability": baseline_summary,
        "recurrence_candidate_signature_count": int(len(recurrence)),
    }
    write_json(output_dir / "summary_statistics.json", summary)

    print("Basic statistical results:")
    print(f"  Total events: {total}")
    print(f"  Output folder: {output_dir}")
    return summary


# ============================================================
# Privacy feature enrichment
# ============================================================
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        return float(text) if text else default
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        text = str(value).strip()
        return int(float(text)) if text else default
    except Exception:
        return default


def prepare_tracks(tracks_csv: Path) -> pd.DataFrame:
    tracks = pd.read_csv(tracks_csv)
    for column in [
        "track_id", "frame_index", "timestamp_sec", "confidence",
        "x1", "y1", "x2", "y2", "center_x", "center_y", "width", "height",
    ]:
        if column in tracks.columns:
            tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
    required = ["track_id", "frame_index", "center_x", "center_y"]
    tracks = tracks.dropna(subset=[column for column in required if column in tracks.columns])
    tracks["track_id"] = tracks["track_id"].astype(int)
    tracks["frame_index"] = tracks["frame_index"].astype(int)
    return tracks.sort_values(["track_id", "frame_index"])


def path_length_px(track_rows: pd.DataFrame) -> float:
    if len(track_rows) < 2:
        return 0.0
    dx = track_rows["center_x"].diff()
    dy = track_rows["center_y"].diff()
    return float(((dx * dx + dy * dy) ** 0.5).fillna(0).sum())


def choose_sample_row(track_rows: pd.DataFrame) -> pd.Series:
    if track_rows.empty:
        return pd.Series(dtype=object)
    return track_rows.iloc[len(track_rows) // 2]


def aggregate_tracks_for_clip(tracks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for track_id, track_rows in tracks.groupby("track_id", sort=False):
        track_rows = track_rows.sort_values("frame_index")
        sample = choose_sample_row(track_rows)
        duration = safe_float(track_rows["timestamp_sec"].iloc[-1]) - safe_float(
            track_rows["timestamp_sec"].iloc[0]) if "timestamp_sec" in track_rows else 0.0
        duration = max(0.0, duration)
        length_px = path_length_px(track_rows)
        mean_width = float(track_rows["width"].mean()) if "width" in track_rows else math.nan
        mean_height = float(track_rows["height"].mean()) if "height" in track_rows else math.nan
        mean_area = mean_width * mean_height if not math.isnan(mean_width) and not math.isnan(mean_height) else math.nan  # noqa: E501
        aspect = mean_width / mean_height if mean_height and not math.isnan(mean_height) else math.nan
        rows.append({
            "track_id": int(track_id),
            "mean_bbox_width": mean_width,
            "mean_bbox_height": mean_height,
            "mean_bbox_area": mean_area,
            "bbox_aspect_ratio": aspect,
            "track_length_px": length_px,
            "track_observed_duration_sec": duration,
            "mean_speed_proxy_px_per_sec": length_px / duration if duration > 0 else math.nan,
            "sample_frame_index": int(sample.get("frame_index", -1)) if not sample.empty else -1,
            "sample_x1": float(sample.get("x1", math.nan)) if not sample.empty else math.nan,
            "sample_y1": float(sample.get("y1", math.nan)) if not sample.empty else math.nan,
            "sample_x2": float(sample.get("x2", math.nan)) if not sample.empty else math.nan,
            "sample_y2": float(sample.get("y2", math.nan)) if not sample.empty else math.nan,
        })
    return pd.DataFrame(rows)


def shrink_box(x1: float, y1: float, x2: float, y2: float, fraction: float) -> tuple[int, int, int, int]:
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    dx = width * fraction * 0.5
    dy = height * fraction * 0.5
    return int(round(x1 + dx)), int(round(y1 + dy)), int(round(x2 - dx)), int(round(y2 - dy))


def coarse_colour_from_crop(crop: Any) -> str:
    if crop is None or crop.size == 0 or cv2 is None:
        return "unknown_colour"
    if crop.shape[0] * crop.shape[1] < MIN_COLOUR_CROP_PIXELS:
        return "unknown_colour"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = float(hsv[:, :, 0].mean())
    mean_s = float(hsv[:, :, 1].mean())
    mean_v = float(hsv[:, :, 2].mean())
    if mean_v < MIN_MEAN_VALUE_FOR_COLOUR:
        return "dark"
    if mean_s < 30:
        if mean_v > 200:
            return "white"
        if mean_v < 80:
            return "black"
        return "grey"
    hue_deg = mean_h * 2.0
    if hue_deg < 20 or hue_deg >= 340:
        return "red"
    if hue_deg < 45:
        return "orange_yellow"
    if hue_deg < 80:
        return "yellow_green"
    if hue_deg < 170:
        return "green_cyan"
    if hue_deg < 260:
        return "blue"
    if hue_deg < 320:
        return "purple"
    return "red"


def extract_colour_for_row(video_path: Path, row: pd.Series) -> str:
    if cv2 is None or not video_path.exists():
        return "unknown_colour"
    frame_index = safe_int(row.get("sample_frame_index", -1), -1)
    if frame_index < 0:
        return "unknown_colour"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "unknown_colour"
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return "unknown_colour"
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = shrink_box(
        safe_float(row.get("sample_x1")), safe_float(row.get("sample_y1")),
        safe_float(row.get("sample_x2")), safe_float(row.get("sample_y2")),
        COLOUR_CROP_SHRINK_FRACTION,
    )
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return "unknown_colour"
    return coarse_colour_from_crop(frame[y1:y2, x1:x2])


def add_quantile_bucket(df: pd.DataFrame, source_col: str, bucket_col: str,
                        quantiles: list[float], labels: list[str]) -> None:
    values = pd.to_numeric(df[source_col], errors="coerce")
    clean = values.dropna()
    if clean.empty:
        df[bucket_col] = "unknown"
        return
    cutoffs = [float(clean.quantile(q)) for q in quantiles]

    def bucket(value: Any) -> str:
        try:
            number = float(value)
        except Exception:
            return "unknown"
        if math.isnan(number):
            return "unknown"
        for cutoff, label in zip(cutoffs, labels):
            if number <= cutoff:
                return label
        return labels[-1]

    df[bucket_col] = values.map(bucket)


def duration_bucket(value: Any) -> str:
    seconds = safe_float(value, math.nan)
    if math.isnan(seconds):
        return "unknown_duration"
    for cutoff in DURATION_BUCKETS_SECONDS:
        if seconds <= cutoff:
            return f"duration_le_{cutoff}s"
    return f"duration_gt_{DURATION_BUCKETS_SECONDS[-1]}s"


def read_tracking_summary_for_events_file(events_file: Path) -> dict[str, Any]:
    return read_json_maybe(events_file.with_name("summary.json"))


def enrich_privacy_features(context: PipelineContext) -> Path:
    master_csv = context.merge_master_csv
    if not master_csv.exists():
        raise FileNotFoundError(f"Master events CSV not found: {master_csv}")

    output_dir = analysis_root_from_context(context) / "privacy_features"
    ensure_dir(output_dir)
    enriched_csv = output_dir / "enriched_events.csv"

    events = pd.read_csv(master_csv)
    events = add_time_columns(events)
    if "source_events_file" not in events.columns:
        raise ValueError("master_events.csv must contain source_events_file for enrichment")

    enriched_parts: list[pd.DataFrame] = []
    files_processed = 0
    files_missing_tracks = 0
    files_failed = 0
    colour_rows_attempted = 0
    colour_rows_known = 0

    grouped = events.groupby("source_events_file", dropna=False)
    for events_file_text, event_rows in tqdm(grouped, desc="Privacy feature enrichment", unit="file"):
        events_file = Path(str(events_file_text))
        tracks_csv = events_file.with_name("tracks.csv")
        part = event_rows.copy()
        files_processed += 1

        if not tracks_csv.exists():
            files_missing_tracks += 1
            enriched_parts.append(part)
            continue

        try:
            tracks = prepare_tracks(tracks_csv)
            track_features = aggregate_tracks_for_clip(tracks)
            if track_features.empty:
                enriched_parts.append(part)
                continue

            if RUN_COLOUR_EXTRACTION and cv2 is not None:
                summary = read_tracking_summary_for_events_file(events_file)
                video_path_text = str(summary.get("video_path", "") or "")
                video_path = Path(video_path_text).expanduser() if video_path_text else Path("")
                colours = []
                for _, feature_row in track_features.iterrows():
                    colour_rows_attempted += 1
                    colour = extract_colour_for_row(video_path, feature_row)
                    if colour != "unknown_colour":
                        colour_rows_known += 1
                    colours.append(colour)
                track_features["coarse_colour"] = colours
            else:
                track_features["coarse_colour"] = "unknown_colour"

            part["track_id"] = pd.to_numeric(part["track_id"], errors="coerce").astype("Int64")
            track_features["track_id"] = pd.to_numeric(track_features["track_id"], errors="coerce").astype("Int64")
            part = part.merge(track_features, on="track_id", how="left")
            enriched_parts.append(part)
        except Exception as exc:
            files_failed += 1
            part["feature_enrichment_error"] = str(exc)
            enriched_parts.append(part)

    enriched = pd.concat(enriched_parts, ignore_index=True) if enriched_parts else events

    for col in ["class_name", "route_type", "half_hour_label", "coarse_colour"]:
        if col not in enriched.columns:
            enriched[col] = "unknown"
        enriched[col] = enriched[col].fillna("unknown").astype(str).replace("", "unknown")

    add_quantile_bucket(
        enriched,
        source_col="mean_bbox_area",
        bucket_col="size_bucket",
        quantiles=SIZE_BUCKET_QUANTILES,
        labels=["size_small", "size_medium", "size_large", "size_very_large"],
    )
    add_quantile_bucket(
        enriched,
        source_col="mean_speed_proxy_px_per_sec",
        bucket_col="speed_bucket",
        quantiles=SPEED_BUCKET_QUANTILES,
        labels=["speed_slow", "speed_medium", "speed_fast", "speed_very_fast"],
    )
    enriched["duration_bucket"] = enriched["duration_sec"].map(duration_bucket) if "duration_sec" in enriched.columns else "unknown_duration"  # noqa: E501

    # Signature columns used by the enriched analysis.
    enriched["signature_baseline"] = enriched[["class_name", "route_type",
                                               "half_hour_label"]].astype(str).agg(" | ".join, axis=1)
    enriched["signature_size_motion"] = enriched[["class_name", "route_type",
                                                  "half_hour_label", "size_bucket", "speed_bucket",
                                                  "duration_bucket"]].astype(str).agg(" | ".join, axis=1)
    enriched["signature_size_motion_colour"] = enriched[["class_name", "route_type", "half_hour_label",
                                                         "size_bucket", "speed_bucket", "duration_bucket",
                                                         "coarse_colour"]].astype(str).agg(" | ".join, axis=1)

    enriched.to_csv(enriched_csv, index=False)

    summary = {
        "master_events_csv": str(master_csv),
        "enriched_events_csv": str(enriched_csv),
        "events": int(len(enriched)),
        "files_processed": files_processed,
        "files_missing_tracks": files_missing_tracks,
        "files_failed": files_failed,
        "run_colour_extraction": bool(RUN_COLOUR_EXTRACTION),
        "cv2_available": cv2 is not None,
        "colour_rows_attempted": int(colour_rows_attempted),
        "colour_rows_known": int(colour_rows_known),
        "percent_colour_known": colour_rows_known / colour_rows_attempted * 100 if colour_rows_attempted else 0.0,
        "feature_columns_added": [
            "mean_bbox_width", "mean_bbox_height", "mean_bbox_area", "bbox_aspect_ratio",
            "track_length_px", "mean_speed_proxy_px_per_sec", "size_bucket", "speed_bucket",
            "duration_bucket", "coarse_colour", "signature_baseline", "signature_size_motion",
            "signature_size_motion_colour",
        ],
    }
    write_json(output_dir / "feature_enrichment_summary.json", summary)

    print("Privacy feature enrichment:")
    print(f"  Enriched events: {len(enriched)}")
    print(f"  Output: {enriched_csv}")
    print(f"  Known colour rows: {colour_rows_known}/{colour_rows_attempted}")
    return enriched_csv


# ============================================================
# Enriched privacy analysis
# ============================================================
def prepare_enriched_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_time_columns(df)
    for column in [
        "class_name", "route_type", "half_hour_label", "size_bucket",
        "speed_bucket", "duration_bucket", "coarse_colour",
    ]:
        if column not in df.columns:
            df[column] = "unknown"
        df[column] = df[column].fillna("unknown").astype(str).replace("", "unknown")
    for column in [
        "duration_sec", "mean_confidence", "mean_bbox_area", "bbox_aspect_ratio",
        "track_length_px", "mean_speed_proxy_px_per_sec",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def recurrence_candidates(df: pd.DataFrame, counts: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rec = (
        df.groupby(columns, dropna=False)
        .agg(
            signature_count=(columns[0], "size"),
            distinct_days=("date", "nunique"),
            mean_confidence=("mean_confidence", "mean"),
            median_duration_sec=("duration_sec", "median"),
        )
        .reset_index()
    )
    rec["signature"] = rec[columns].astype(str).agg(" | ".join, axis=1)
    rec = rec[(rec["distinct_days"] >= 2) & (rec["signature_count"] <= RARE_RECURRENCE_MAX_EVENTS)]
    return rec.sort_values(["distinct_days", "signature_count"], ascending=[False, True])


def run_enriched_privacy_results(context: PipelineContext, enriched_csv: Path | None = None) -> dict[str, Any]:
    if enriched_csv is None:
        enriched_csv = analysis_root_from_context(context) / "privacy_features" / "enriched_events.csv"
    if not enriched_csv.exists():
        raise FileNotFoundError(f"Enriched events CSV not found: {enriched_csv}")

    output_dir = analysis_root_from_context(context) / "enriched_privacy_results"
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    df = prepare_enriched_events(enriched_csv)
    total = len(df)

    signature_levels = {
        "baseline": ["class_name", "route_type", "half_hour_label"],
        "size_motion": ["class_name", "route_type", "half_hour_label", "size_bucket",
                        "speed_bucket", "duration_bucket"],
        "size_motion_colour": ["class_name", "route_type", "half_hour_label", "size_bucket",
                               "speed_bucket", "duration_bucket", "coarse_colour"],
    }

    summaries: list[dict[str, Any]] = []
    all_counts: list[pd.DataFrame] = []
    rare_recurrence_by_level: dict[str, int] = {}

    for level_name, columns in signature_levels.items():
        counts = signature_counts(df.copy(), level_name, columns)
        all_counts.append(counts)
        summary = signature_summary(counts, total, columns)
        summary["signature_level"] = level_name
        summaries.append(summary)
        write_csv(tables_dir / f"{level_name}_signature_counts.csv", counts)
        write_csv(tables_dir / f"{level_name}_top_confusable_signatures.csv", counts.head(TOP_N))

        rec = recurrence_candidates(df, counts, columns)
        rare_recurrence_by_level[level_name] = int(len(rec))
        write_csv(tables_dir / f"{level_name}_rare_recurrence_candidates.csv", rec)

        plot_bar(
            counts.head(TOP_N), "signature", "signature_count",
            plots_dir / f"{level_name}_top_confusable_signatures.png",
            f"Top {level_name} signatures",
            rotate=80,
        )
        plot_histogram(
            counts,
            "signature_count",
            plots_dir / f"{level_name}_confusability_distribution.png",
            f"Confusability distribution: {level_name}",
            nbins=50,
        )

    summary_table = pd.DataFrame(summaries)
    summary_table["rare_recurrence_candidate_signatures"] = summary_table["signature_level"].map(
        rare_recurrence_by_level).fillna(0).astype(int)
    write_csv(tables_dir / "signature_summary_comparison.csv", summary_table)
    write_csv(tables_dir / "all_signature_counts.csv", pd.concat(all_counts, ignore_index=True))
    plot_signature_level_comparison(summary_table, plots_dir)

    # Distinctive events under strongest signature.
    strong_counts = signature_counts(df.copy(), "size_motion_colour", signature_levels["size_motion_colour"])
    count_lookup = strong_counts.set_index("signature")["signature_count"].to_dict()
    df["size_motion_colour_signature_count"] = df["signature_size_motion_colour"].map(
        count_lookup).fillna(0).astype(int)
    distinctive = df.sort_values(["size_motion_colour_signature_count",
                                  "mean_confidence", "duration_sec"], ascending=[True, False, False])
    write_csv(tables_dir / "top_distinctive_events_for_manual_review.csv", distinctive.head(200))

    summary = {
        "enriched_events_csv": str(enriched_csv),
        "output_dir": str(output_dir),
        "total_events": int(total),
        "signature_summaries": summaries,
        "rare_recurrence_candidate_counts": rare_recurrence_by_level,
        "strongest_signature_level": "size_motion_colour",
    }
    write_json(output_dir / "enriched_summary_statistics.json", summary)

    plot_bar(
        summary_table,
        "signature_level",
        "percent_events_in_low_confusability_signatures",
        plots_dir / "low_confusability_percent_by_signature_level.png",
        "Low confusability event percentage by signature level",
        rotate=20,
    )

    print("Enriched privacy results:")
    print(f"  Output folder: {output_dir}")
    print(f"  Signature comparison: {tables_dir / 'signature_summary_comparison.csv'}")
    return summary


# ============================================================
# Regenerate Plotly figures from existing outputs only
# ============================================================
def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def add_rare_recurrence_counts_from_summary(summary_table: pd.DataFrame, summary_json_path: Path) -> pd.DataFrame:
    if summary_table.empty or not summary_json_path.exists():
        return summary_table
    try:
        with summary_json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return summary_table

    recurrence_counts = payload.get("rare_recurrence_candidate_counts", {})
    if recurrence_counts and "signature_level" in summary_table.columns:
        summary_table = summary_table.copy()
        summary_table["rare_recurrence_candidate_signatures"] = (
            summary_table["signature_level"].map(recurrence_counts).fillna(0).astype(int)
        )
    return summary_table


def regenerate_existing_plot_exports(context: PipelineContext) -> None:
    """
    Recreate HTML, PNG, and EPS figures from existing CSV/JSON outputs.

    This does not rerun event extraction, merging, enrichment, or statistical analysis.
    Use it when tables already exist and only figure exports are needed.
    """
    analysis_root = analysis_root_from_context(context)
    basic_tables_dir = analysis_root / "tables"
    basic_plots_dir = analysis_root / "plots"
    enriched_root = analysis_root / "enriched_privacy_results"
    enriched_tables_dir = enriched_root / "tables"
    enriched_plots_dir = enriched_root / "plots"

    ensure_dir(basic_plots_dir)
    ensure_dir(enriched_plots_dir)

    route_counts = read_csv_if_exists(basic_tables_dir / "route_counts.csv")
    class_counts = read_csv_if_exists(basic_tables_dir / "class_counts.csv")
    events_by_hour = read_csv_if_exists(basic_tables_dir / "events_by_hour.csv")
    events_by_date = read_csv_if_exists(basic_tables_dir / "events_by_date.csv")
    top_confusable = read_csv_if_exists(basic_tables_dir / "top_confusable_signatures.csv")
    confusability = read_csv_if_exists(basic_tables_dir / "confusability_signatures.csv")

    plot_bar(route_counts, "route_type", "count", basic_plots_dir / "route_counts.png", "Route counts")
    plot_bar(class_counts.head(TOP_N), "class_name", "count", basic_plots_dir / "class_counts_top.png",
             "Class counts", rotate=45)
    plot_line(events_by_hour, "hour", "count", basic_plots_dir / "events_by_hour.png", "Events by hour")
    plot_bar(events_by_date, "date", "count", basic_plots_dir / "events_by_date.png", "Events by date", rotate=45)
    plot_bar(top_confusable.head(TOP_N), "signature", "signature_count", basic_plots_dir / "top_confusable_signatures.png",  # noqa: E501
             "Top baseline signatures", rotate=80)
    plot_histogram(confusability, "signature_count", basic_plots_dir / "confusability_distribution.png",
                   "Confusability distribution", nbins=50)

    summary_table = read_csv_if_exists(enriched_tables_dir / "signature_summary_comparison.csv")
    summary_table = add_rare_recurrence_counts_from_summary(summary_table, enriched_root / "enriched_summary_statistics.json")  # noqa: E501
    if not summary_table.empty:
        write_csv(enriched_tables_dir / "signature_summary_comparison.csv", summary_table)
        plot_bar(
            summary_table,
            "signature_level",
            "percent_events_in_low_confusability_signatures",
            enriched_plots_dir / "low_confusability_percent_by_signature_level.png",
            "Low confusability event percentage by signature level",
            rotate=20,
        )
        plot_signature_level_comparison(summary_table, enriched_plots_dir)

    for level_name in ["baseline", "size_motion", "size_motion_colour"]:
        counts = read_csv_if_exists(enriched_tables_dir / f"{level_name}_signature_counts.csv")
        top_counts = read_csv_if_exists(enriched_tables_dir / f"{level_name}_top_confusable_signatures.csv")
        if top_counts.empty and not counts.empty:
            top_counts = counts.head(TOP_N)
        plot_bar(
            top_counts.head(TOP_N),
            "signature",
            "signature_count",
            enriched_plots_dir / f"{level_name}_top_confusable_signatures.png",
            f"Top {level_name} signatures",
            rotate=80,
        )
        plot_histogram(
            counts,
            "signature_count",
            enriched_plots_dir / f"{level_name}_confusability_distribution.png",
            f"Confusability distribution: {level_name}",
            nbins=50,
        )

    print("Regenerated Plotly figure exports from existing tables:")
    print(f"  Basic plots: {basic_plots_dir}")
    print(f"  Enriched plots: {enriched_plots_dir}")
    print(f"  Final figures: {final_figures_dir()}")


# ============================================================
# Manual review sample
# ============================================================
def select_sample(df: pd.DataFrame, condition: pd.Series, label: str, count: int) -> pd.DataFrame:
    subset = df[condition].copy()
    if subset.empty:
        return subset
    if len(subset) > count:
        subset = subset.sample(n=count, random_state=RANDOM_SEED)
    subset.insert(0, "review_group", label)
    return subset


def sample_manual_review_events(context: PipelineContext) -> Path:
    enriched_csv = analysis_root_from_context(context) / "privacy_features" / "enriched_events.csv"
    if not enriched_csv.exists():
        raise FileNotFoundError(f"Enriched events CSV not found: {enriched_csv}")
    df = prepare_enriched_events(enriched_csv)

    # Signature counts for sampling.
    for col in ["signature_baseline", "signature_size_motion", "signature_size_motion_colour"]:
        if col not in df.columns:
            df[col] = "unknown"
    df["baseline_signature_count"] = df["signature_baseline"].map(
        df["signature_baseline"].value_counts()).fillna(0).astype(int)

    df["size_motion_colour_signature_count"] = df["signature_size_motion_colour"].map(
        df["signature_size_motion_colour"].value_counts()).fillna(0).astype(int)

    samples = [
        select_sample(
            df,
            df["size_motion_colour_signature_count"] <= LOW_CONFUSABILITY_THRESHOLD,
            "distinctive_low_confusability",
            MANUAL_REVIEW_SAMPLE_PER_GROUP,
        ),
        select_sample(
            df,
            df["duration_sec"] >= df["duration_sec"].quantile(0.95),
            "long_duration",
            MANUAL_REVIEW_SAMPLE_PER_GROUP,
        ),
        select_sample(
            df,
            df["mean_confidence"] <= df["mean_confidence"].quantile(0.05),
            "low_confidence",
            MANUAL_REVIEW_SAMPLE_PER_GROUP,
        ),
        select_sample(
            df,
            df["baseline_signature_count"] >= df["baseline_signature_count"].quantile(0.95),
            "common_baseline",
            MANUAL_REVIEW_SAMPLE_PER_GROUP,
        ),
    ]

    sample = pd.concat([part for part in samples if part is not None and not part.empty], ignore_index=True) if samples else pd.DataFrame()  # noqa: E501
    preferred_columns = [
        "review_group", "source_events_file", "video_name", "video_id", "clip_id", "track_id",
        "class_name", "route_type", "wallclock_start", "wallclock_end", "duration_sec",
        "mean_confidence", "size_bucket", "speed_bucket", "duration_bucket", "coarse_colour",
        "signature_baseline", "baseline_signature_count", "signature_size_motion_colour",
        "size_motion_colour_signature_count", "start_frame", "end_frame", "start_time_sec", "end_time_sec",
    ]
    columns = [col for col in preferred_columns if col in sample.columns]
    if columns:
        sample = sample[columns]

    output_path = analysis_root_from_context(context) / "enriched_privacy_results" / "tables" / "manual_review_sample.csv"  # noqa: E501
    write_csv(output_path, sample)
    print("Manual review sample:")
    print(f"  Rows: {len(sample)}")
    print(f"  Output: {output_path}")
    return output_path


# ============================================================
# Mitigation video generation
# ============================================================
def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def run_subprocess(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def read_clip_manifest_paths(context: PipelineContext) -> list[Path]:
    paths: list[Path] = []
    manifest = context.clip_manifest_csv_path
    if manifest.exists():
        try:
            with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    for key in ["standardized_path", "source_path", "preview_path"]:
                        value = str(row.get(key, "") or "").strip()
                        if value:
                            path = Path(value).expanduser()
                            if path.exists() and path.is_file():
                                paths.append(path)
                                break
        except Exception:
            pass

    if not paths and context.merge_master_csv.exists():
        try:
            events = pd.read_csv(context.merge_master_csv, usecols=["source_events_file"])
            for value in events["source_events_file"].dropna().astype(str).unique():
                summary = read_json_maybe(Path(value).with_name("summary.json"))
                video_path = str(summary.get("video_path", "") or "").strip()
                if video_path:
                    path = Path(video_path).expanduser()
                    if path.exists() and path.is_file():
                        paths.append(path)
        except Exception:
            pass

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path.resolve())
    return unique[:MAX_MITIGATION_SOURCE_CLIPS]


def mitigation_filter(variant: str, context: PipelineContext) -> str:
    overlay = (
        f"drawbox=x=iw*{context.crop_x}:y=ih*{context.crop_y}:"
        f"w=iw*{context.crop_w}:h=ih*{context.crop_h}:color=black@1:t=fill"
    )
    low_resolution = "scale=trunc(iw*0.5/2)*2:trunc(ih*0.5/2)*2"
    low_frame_rate = "fps=2"
    heavy_downsample = "fps=1,scale=trunc(iw*0.25/2)*2:trunc(ih*0.25/2)*2"

    if variant == "overlay_masked":
        return overlay
    if variant == "low_resolution":
        return low_resolution
    if variant == "low_frame_rate":
        return low_frame_rate
    if variant == "low_resolution_low_frame_rate":
        return f"{low_frame_rate},{low_resolution}"
    if variant == "heavy_downsample":
        return heavy_downsample
    return ""


def make_mitigation_video(source: Path, output: Path, variant: str, context: PipelineContext) -> tuple[bool, str]:
    ensure_parent(output)
    if output.exists() and not cfg_bool("overwrite_existing_mitigation_videos", False):
        return True, "existing"
    vf = mitigation_filter(variant, context)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-map", "0:v:0",
    ]
    if vf:
        command.extend(["-vf", vf])
    command.extend([
        "-c:v", str(cfg("mitigation_video_codec", "libx264")),
        "-preset", str(cfg("mitigation_preset", "veryfast")),
        "-crf", str(cfg("mitigation_crf", 23)),
        "-pix_fmt", "yuv420p",
        "-an",
        str(output),
    ])
    result = run_subprocess(command)
    if result.returncode != 0:
        return False, result.stderr.strip() or "ffmpeg_failed"
    return True, "ok"


def generate_mitigation_videos(context: PipelineContext) -> Path:
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH for mitigation video generation")

    sources = read_clip_manifest_paths(context)
    if not sources:
        raise FileNotFoundError("Could not find source clips for mitigation generation")

    output_root = cfg_path("mitigation_output_root", context.output_root / "mitigation_videos")
    assert output_root is not None
    output_root = output_root.resolve()
    ensure_dir(output_root)

    rows: list[dict[str, Any]] = []
    for source in tqdm(sources, desc="Mitigation videos", unit="clip"):
        for variant in MITIGATION_VARIANTS:
            output = output_root / variant / source.name
            ok, status = make_mitigation_video(source, output, variant, context)
            rows.append({
                "source_path": str(source),
                "variant": variant,
                "output_path": str(output),
                "status": status if ok else "failed",
                "error": "" if ok else status,
            })

    manifest = output_root / "mitigation_manifest.csv"
    write_csv(manifest, pd.DataFrame(rows))
    print("Mitigation videos:")
    print(f"  Source clips: {len(sources)}")
    print(f"  Output root: {output_root}")
    print(f"  Manifest: {manifest}")
    return manifest


# ============================================================
# Main
# ============================================================
def main() -> None:
    logs(show_level=cfg("logger_level", "info"), show_color=True)
    context = create_context()

    run_event_extraction_flag = cfg_bool("run_event_extraction", RUN_EVENT_EXTRACTION)
    run_event_merge_flag = cfg_bool("run_event_merge", RUN_EVENT_MERGE)
    run_basic_results_flag = cfg_bool("run_basic_statistical_results", RUN_BASIC_STATISTICAL_RESULTS)
    run_enrichment_flag = cfg_bool("run_privacy_feature_enrichment", RUN_PRIVACY_FEATURE_ENRICHMENT)
    run_enriched_results_flag = cfg_bool("run_enriched_privacy_results", RUN_ENRICHED_PRIVACY_RESULTS)
    run_manual_review_flag = cfg_bool("run_manual_review_sample", RUN_MANUAL_REVIEW_SAMPLE)
    run_regenerate_plots_flag = cfg_bool("run_regenerate_existing_plots", RUN_REGENERATE_EXISTING_PLOTS)
    run_mitigation_flag = cfg_bool("run_mitigation_video_generation", RUN_MITIGATION_VIDEO_GENERATION)

    global RUN_COLOUR_EXTRACTION
    RUN_COLOUR_EXTRACTION = cfg_bool("run_colour_extraction", RUN_COLOUR_EXTRACTION)

    stages: list[tuple[str, Any]] = []
    if run_event_extraction_flag:
        stages.append(("event extraction", lambda: run_event_extraction(context)))
    if run_event_merge_flag:
        stages.append(("event merge", lambda: run_event_merge(context)))
    if run_basic_results_flag:
        stages.append(("basic statistical results", lambda: run_basic_statistical_results(context)))
    if run_enrichment_flag:
        stages.append(("privacy feature enrichment", lambda: enrich_privacy_features(context)))
    if run_enriched_results_flag:
        stages.append(("enriched privacy results", lambda: run_enriched_privacy_results(context)))
    if run_manual_review_flag:
        stages.append(("manual review sample", lambda: sample_manual_review_events(context)))
    if run_regenerate_plots_flag:
        stages.append(("regenerate existing plot exports", lambda: regenerate_existing_plot_exports(context)))
    if run_mitigation_flag:
        stages.append(("mitigation video generation", lambda: generate_mitigation_videos(context)))

    logger.info("Starting analysis pipeline")
    logger.info("Annotated YOLO videos are not produced by the analysis runner.")

    for name, fn in tqdm(stages, desc="Analysis stages", unit="stage"):
        logger.info("Running {}", name)
        fn()

    logger.info("Analysis pipeline finished")


if __name__ == "__main__":
    main()
