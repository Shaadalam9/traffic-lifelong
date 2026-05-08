#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any
import common
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CONFIG_PATH = Path("config")
DEFAULT_MASTER_EVENTS_CSV = Path("_output/event_tables/master_events.csv")
DEFAULT_ANALYSIS_OUTPUT_DIR = Path("_output/analysis_results")
TOP_N = 20


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


def cfg_path(key: str, default: Any = None) -> Path | None:
    value = cfg(key, default)
    if value in (None, ""):
        return None
    return Path(value).expanduser()


def candidate_master_paths() -> list[Path]:
    paths: list[Path] = []

    configured = cfg_path("merge_master_csv", None)
    if configured is not None:
        paths.append(configured)

    output_root = cfg_path("output_root", None)
    project_root = cfg_path("project_root", Path(".")) or Path(".")

    if output_root is not None:
        paths.extend([
            output_root / "event_tables" / "master_events.csv",
            output_root / "_output" / "event_tables" / "master_events.csv",
        ])

    paths.extend([
        project_root / "_output" / "event_tables" / "master_events.csv",
        project_root / "event_tables" / "master_events.csv",
        DEFAULT_MASTER_EVENTS_CSV,
    ])

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved_key = str(path.expanduser())
        if resolved_key in seen:
            continue
        seen.add(resolved_key)
        unique_paths.append(path.expanduser())
    return unique_paths


def find_master_events_csv() -> Path:
    for path in candidate_master_paths():
        if path.exists():
            return path.resolve()
    searched = "\n  - ".join(str(path) for path in candidate_master_paths())
    raise FileNotFoundError(
        "Could not find master_events.csv. Searched:\n"
        f"  - {searched}\n"
        "Run run_analysis.py first, or set merge_master_csv in config."
    )


def infer_output_dir(master_csv: Path) -> Path:
    configured = cfg_path("analysis_output_root", None)
    if configured is not None:
        return configured.resolve()
    if master_csv.parent.name == "event_tables":
        return master_csv.parent.parent / "analysis_results"
    return DEFAULT_ANALYSIS_OUTPUT_DIR.resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_label(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip()
    return text if text else default


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return cleaned or "result"


def to_numeric(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


def parse_wallclock(df: pd.DataFrame) -> pd.DataFrame:
    if "wallclock_start" in df.columns:
        df["wallclock_start_dt"] = pd.to_datetime(df["wallclock_start"], errors="coerce")
    else:
        df["wallclock_start_dt"] = pd.NaT

    if "wallclock_end" in df.columns:
        df["wallclock_end_dt"] = pd.to_datetime(df["wallclock_end"], errors="coerce")
    else:
        df["wallclock_end_dt"] = pd.NaT

    has_time = df["wallclock_start_dt"].notna()
    df["date"] = df["wallclock_start_dt"].dt.date.astype("string")
    df.loc[~has_time, "date"] = ""
    df["hour"] = df["wallclock_start_dt"].dt.hour
    df["minute"] = df["wallclock_start_dt"].dt.minute
    df["half_hour_index"] = (df["hour"] * 2 + (df["minute"] // 30)).astype("Int64")
    df["half_hour_label"] = df["half_hour_index"].map(format_half_hour_label)
    df.loc[~has_time, "half_hour_label"] = "unknown_time"
    return df


def format_half_hour_label(value: Any) -> str:
    try:
        if pd.isna(value):
            return "unknown_time"
        index = int(value)
    except Exception:
        return "unknown_time"
    hour = index // 2
    minute = 30 if index % 2 else 0
    return f"{hour:02d}:{minute:02d}"


def prepare_events(master_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(master_csv)
    for column in [
        "duration_sec",
        "mean_confidence",
        "num_points",
        "start_time_sec",
        "end_time_sec",
        "first_crossing_time_sec",
        "last_crossing_time_sec",
    ]:
        to_numeric(df, column)

    for column in ["route_type", "class_name", "entry_zone", "exit_zone", "video_name", "clip_id"]:
        if column not in df.columns:
            df[column] = "unknown"
        df[column] = df[column].map(clean_label)

    df = parse_wallclock(df)
    return df


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def count_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="count")
    total = counts["count"].sum()
    counts["percent"] = counts["count"] / total * 100 if total else 0.0
    return counts


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=1)) if clean.count() > 1 else 0.0,
        "min": float(clean.min()),
        "p10": float(clean.quantile(0.10)),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def grouped_numeric_summary(df: pd.DataFrame, group_column: str, value_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, group_df in df.groupby(group_column, dropna=False):
        row = {group_column: clean_label(group)}
        row.update(numeric_summary(group_df[value_column]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def shannon_entropy_from_counts(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * probabilities.map(lambda p: math.log2(float(p)))).sum())


def cramer_v(table: pd.DataFrame) -> float | None:
    if table.empty:
        return None

    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total <= 0:
        return None

    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total

    valid = expected > 0
    if not valid.any():
        return None

    chi_square_terms = np.zeros_like(observed, dtype=float)
    chi_square_terms[valid] = ((observed[valid] - expected[valid]) ** 2) / expected[valid]
    chi_square = float(chi_square_terms.sum())

    rows, cols = observed.shape
    denom = total * max(1, min(rows - 1, cols - 1))
    if denom <= 0:
        return None
    return float(math.sqrt(chi_square / denom))


def save_current_plot(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_bar(table: pd.DataFrame, x_column: str, y_column: str, title: str, output_path: Path, top_n: int | None = None) -> None:
    if table.empty:
        return
    plot_data = table.copy()
    if top_n is not None:
        plot_data = plot_data.head(top_n)
    plt.figure(figsize=(11, 6))
    plt.bar(plot_data[x_column].astype(str), plot_data[y_column])
    plt.title(title)
    plt.xlabel(x_column.replace("_", " ").title())
    plt.ylabel(y_column.replace("_", " ").title())
    plt.xticks(rotation=35, ha="right")
    save_current_plot(output_path)


def plot_line(table: pd.DataFrame, x_column: str, y_column: str, title: str, output_path: Path) -> None:
    if table.empty:
        return
    plt.figure(figsize=(11, 6))
    plt.plot(table[x_column], table[y_column], marker="o")
    plt.title(title)
    plt.xlabel(x_column.replace("_", " ").title())
    plt.ylabel(y_column.replace("_", " ").title())
    plt.xticks(rotation=35, ha="right")
    save_current_plot(output_path)


def plot_stacked_route_by_hour(route_by_hour: pd.DataFrame, output_path: Path) -> None:
    if route_by_hour.empty:
        return
    pivot = route_by_hour.pivot(index="hour", columns="route_type", values="count").fillna(0).sort_index()
    if pivot.empty:
        return
    plt.figure(figsize=(12, 6))
    bottom = None
    for route in pivot.columns:
        values = pivot[route].to_numpy()
        if bottom is None:
            plt.bar(pivot.index.astype(int), values, label=str(route))
            bottom = values.copy()
        else:
            plt.bar(pivot.index.astype(int), values, bottom=bottom, label=str(route))
            bottom = bottom + values
    plt.title("Events By Hour And Route")
    plt.xlabel("Hour Of Day")
    plt.ylabel("Event Count")
    plt.legend(title="Route")
    save_current_plot(output_path)


def plot_duration_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    clean = df[["route_type", "duration_sec"]].dropna()
    if clean.empty:
        return
    grouped = [group["duration_sec"].to_numpy() for _, group in clean.groupby("route_type")]
    labels = [str(route) for route, _ in clean.groupby("route_type")]
    if not grouped:
        return
    plt.figure(figsize=(10, 6))
    plt.boxplot(grouped, labels=labels, showfliers=False)
    plt.title("Track Duration Distribution By Route")
    plt.xlabel("Route")
    plt.ylabel("Duration Seconds")
    save_current_plot(output_path)


def plot_confusability_histogram(signature_counts: pd.DataFrame, output_path: Path) -> None:
    if signature_counts.empty:
        return
    plt.figure(figsize=(11, 6))
    plt.hist(signature_counts["signature_count"], bins=50)
    plt.title("Confusability Distribution")
    plt.xlabel("Number Of Events Sharing The Same Coarse Signature")
    plt.ylabel("Number Of Signatures")
    save_current_plot(output_path)


def make_route_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    timed = df[df["hour"].notna()].copy()
    if timed.empty:
        return pd.DataFrame(columns=["hour", "route_type", "count"])
    timed["hour"] = timed["hour"].astype(int)
    return timed.groupby(["hour", "route_type"]).size().reset_index(name="count").sort_values(["hour", "route_type"])


def make_events_by_date(df: pd.DataFrame) -> pd.DataFrame:
    timed = df[df["date"].astype(str) != ""].copy()
    if timed.empty:
        return pd.DataFrame(columns=["date", "count"])
    return timed.groupby("date").size().reset_index(name="count").sort_values("date")


def make_events_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    timed = df[df["hour"].notna()].copy()
    if timed.empty:
        return pd.DataFrame(columns=["hour", "count"])
    timed["hour"] = timed["hour"].astype(int)
    return timed.groupby("hour").size().reset_index(name="count").sort_values("hour")


def make_confusability_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    signature_columns = ["class_name", "route_type", "half_hour_label"]
    counts = (
        df.groupby(signature_columns)
        .size()
        .reset_index(name="signature_count")
        .sort_values("signature_count", ascending=False)
    )
    total_events = int(len(df))
    singleton_signatures = int((counts["signature_count"] == 1).sum()) if not counts.empty else 0
    events_in_singletons = int(counts.loc[counts["signature_count"] == 1, "signature_count"].sum()) if not counts.empty else 0
    top_signatures = counts.head(TOP_N).copy()

    summary = {
        "signature_definition": signature_columns,
        "total_events": total_events,
        "unique_signatures": int(len(counts)),
        "singleton_signatures": singleton_signatures,
        "events_in_singleton_signatures": events_in_singletons,
        "percent_events_in_singleton_signatures": (events_in_singletons / total_events * 100) if total_events else 0.0,
        "median_events_per_signature": float(counts["signature_count"].median()) if not counts.empty else None,
        "mean_events_per_signature": float(counts["signature_count"].mean()) if not counts.empty else None,
        "max_events_per_signature": int(counts["signature_count"].max()) if not counts.empty else None,
    }
    return counts, top_signatures, summary


def make_recurrence_candidates(df: pd.DataFrame) -> pd.DataFrame:
    timed = df[(df["date"].astype(str) != "") & (df["half_hour_label"] != "unknown_time")].copy()
    if timed.empty:
        return pd.DataFrame(columns=["class_name", "route_type", "half_hour_label", "event_count", "distinct_days", "first_date", "last_date"])

    grouped = (
        timed.groupby(["class_name", "route_type", "half_hour_label"])
        .agg(
            event_count=("route_type", "size"),
            distinct_days=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
    )
    return grouped[grouped["distinct_days"] >= 2].sort_values(["distinct_days", "event_count"], ascending=False)


def make_top_long_events(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column for column in [
            "video_name", "clip_id", "track_id", "class_name", "route_type", "duration_sec",
            "mean_confidence", "wallclock_start", "wallclock_end", "source_events_file"
        ]
        if column in df.columns
    ]
    if "duration_sec" not in df.columns:
        return pd.DataFrame(columns=columns)
    return df.sort_values("duration_sec", ascending=False).head(50)[columns]


def make_low_confidence_events(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column for column in [
            "video_name", "clip_id", "track_id", "class_name", "route_type", "duration_sec",
            "mean_confidence", "wallclock_start", "wallclock_end", "source_events_file"
        ]
        if column in df.columns
    ]
    if "mean_confidence" not in df.columns:
        return pd.DataFrame(columns=columns)
    return df.sort_values("mean_confidence", ascending=True).head(50)[columns]


def run_results_analysis() -> dict[str, Any]:
    master_csv = find_master_events_csv()
    output_dir = infer_output_dir(master_csv)
    table_dir = output_dir / "tables"
    plot_dir = output_dir / "plots"
    ensure_dir(table_dir)
    ensure_dir(plot_dir)

    font_family = cfg("font_family", None)
    if font_family:
        plt.rcParams["font.family"] = font_family

    df = prepare_events(master_csv)

    route_counts = count_table(df, "route_type")
    class_counts = count_table(df, "class_name")
    route_by_hour = make_route_by_hour(df)
    events_by_hour = make_events_by_hour(df)
    events_by_date = make_events_by_date(df)
    route_by_class = pd.crosstab(df["route_type"], df["class_name"])
    duration_by_route = grouped_numeric_summary(df, "route_type", "duration_sec")
    confidence_by_route = grouped_numeric_summary(df, "route_type", "mean_confidence")
    confidence_by_class = grouped_numeric_summary(df, "class_name", "mean_confidence")
    signature_counts, top_signatures, confusability_summary = make_confusability_tables(df)
    recurrence_candidates = make_recurrence_candidates(df)
    top_long_events = make_top_long_events(df)
    low_confidence_events = make_low_confidence_events(df)

    write_csv(table_dir / "route_counts.csv", route_counts)
    write_csv(table_dir / "class_counts.csv", class_counts)
    write_csv(table_dir / "route_by_hour.csv", route_by_hour)
    write_csv(table_dir / "events_by_hour.csv", events_by_hour)
    write_csv(table_dir / "events_by_date.csv", events_by_date)
    write_csv(table_dir / "route_by_class.csv", route_by_class.reset_index())
    write_csv(table_dir / "duration_statistics_by_route.csv", duration_by_route)
    write_csv(table_dir / "confidence_statistics_by_route.csv", confidence_by_route)
    write_csv(table_dir / "confidence_statistics_by_class.csv", confidence_by_class)
    write_csv(table_dir / "confusability_signatures.csv", signature_counts)
    write_csv(table_dir / "top_confusable_signatures.csv", top_signatures)
    write_csv(table_dir / "recurrence_candidate_signatures.csv", recurrence_candidates)
    write_csv(table_dir / "top_long_duration_events.csv", top_long_events)
    write_csv(table_dir / "low_confidence_events.csv", low_confidence_events)

    route_entropy = shannon_entropy_from_counts(route_counts.set_index("route_type")["count"])
    class_entropy = shannon_entropy_from_counts(class_counts.set_index("class_name")["count"])
    route_class_association = cramer_v(route_by_class)

    timed_events = int(df["wallclock_start_dt"].notna().sum())
    total_events = int(len(df))
    summary = {
        "master_events_csv": str(master_csv),
        "analysis_output_dir": str(output_dir),
        "total_events": total_events,
        "events_with_wallclock_time": timed_events,
        "percent_with_wallclock_time": timed_events / total_events * 100 if total_events else 0.0,
        "unique_videos": int(df["video_name"].nunique()) if "video_name" in df.columns else None,
        "unique_clips": int(df["clip_id"].nunique()) if "clip_id" in df.columns else None,
        "unique_routes": int(df["route_type"].nunique()),
        "unique_classes": int(df["class_name"].nunique()),
        "first_wallclock_time": str(df["wallclock_start_dt"].min()) if timed_events else "",
        "last_wallclock_time": str(df["wallclock_start_dt"].max()) if timed_events else "",
        "route_entropy_bits": route_entropy,
        "class_entropy_bits": class_entropy,
        "route_class_cramers_v": route_class_association,
        "duration_seconds": numeric_summary(df["duration_sec"]),
        "mean_confidence": numeric_summary(df["mean_confidence"]),
        "confusability": confusability_summary,
        "recurrence_candidate_signature_count": int(len(recurrence_candidates)),
    }
    write_json(output_dir / "summary_statistics.json", summary)

    plot_bar(route_counts, "route_type", "count", "Route Counts", plot_dir / "route_counts.png")
    plot_bar(class_counts, "class_name", "count", f"Top {TOP_N} Vehicle Classes", plot_dir / "class_counts_top.png", top_n=TOP_N)
    plot_line(events_by_hour, "hour", "count", "Events By Hour Of Day", plot_dir / "events_by_hour.png")
    plot_line(events_by_date, "date", "count", "Events By Date", plot_dir / "events_by_date.png")
    plot_stacked_route_by_hour(route_by_hour, plot_dir / "route_by_hour_stacked.png")
    plot_duration_boxplot(df, plot_dir / "duration_by_route_boxplot.png")
    plot_confusability_histogram(signature_counts, plot_dir / "confusability_distribution.png")
    plot_bar(top_signatures.assign(signature=top_signatures[["class_name", "route_type", "half_hour_label"]].agg(" | ".join, axis=1)),
             "signature", "signature_count", "Top Confusable Coarse Signatures", plot_dir / "top_confusable_signatures.png", top_n=TOP_N)

    if not recurrence_candidates.empty:
        recurrence_plot = recurrence_candidates.head(TOP_N).copy()
        recurrence_plot["signature"] = recurrence_plot[["class_name", "route_type", "half_hour_label"]].agg(" | ".join, axis=1)
        plot_bar(recurrence_plot, "signature", "event_count", "Top Recurrence Candidate Signatures", plot_dir / "recurrence_candidate_signatures.png", top_n=TOP_N)

    print("Result analysis complete.")
    print(f"  Master events: {master_csv}")
    print(f"  Total events: {summary['total_events']}")
    print(f"  Output folder: {output_dir}")
    print(f"  Summary JSON: {output_dir / 'summary_statistics.json'}")
    print(f"  Plots folder: {plot_dir}")
    print(f"  Tables folder: {table_dir}")
    return summary


def main() -> None:
    run_results_analysis()


if __name__ == "__main__":
    main()
