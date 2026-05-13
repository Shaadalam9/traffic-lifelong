from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================
# Edit only these values
# ============================================================

IMAGE_PATH = Path("/Users/alam/Repos/traffic-lifelong/data/cvat_annotation/images/frame_05_31_36.png")
SCENE_JSON_PATH = Path("/Users/alam/Repos/traffic-lifelong/data/scene_regions.json")
OUTPUT_PATH = Path("scene_regions_overlay_clean.png")


# ============================================================
# Display options
# ============================================================

SHOW_VALID_ROAD = True
SHOW_OVERLAY_REGIONS = True
SHOW_IGNORE_REGIONS = True
SHOW_STOP_LINE = True
SHOW_BOUNDARIES = True

VALID_ROAD_ALPHA = 0.30
OVERLAY_REGION_ALPHA = 0.26
IGNORE_REGION_ALPHA = 0.14
STOP_LINE_ALPHA = 0.18

REGION_LINE_THICKNESS = 4
BOUNDARY_THICKNESS = 7

FONT_SCALE = 0.85
FONT_THICKNESS = 2


# OpenCV uses BGR colour order
COLOURS = {
    "valid_road_region": (70, 200, 70),
    "overlay_region": (0, 170, 255),
    "ignore_region": (80, 80, 80),
    "stop_line_near": (255, 255, 0),
    "queue_area_near": (255, 0, 255),
    "boundary_bottom": (0, 0, 255),
    "boundary_far_left": (255, 80, 40),
    "boundary_far_center": (0, 255, 255),
    "boundary_far_right": (255, 0, 255),
}

PRETTY_NAMES = {
    "valid_road_region": "Valid road region",
    "overlay_region": "Overlay region",
    "ignore_region": "Ignore region",
    "stop_line_near": "Stop line",
    "queue_area_near": "Queue area",
    "boundary_bottom": "Entry boundary",
    "boundary_far_left": "Left exit",
    "boundary_far_center": "Straight exit",
    "boundary_far_right": "Right exit",
}


# The raw bottom centre line in the JSON is visually zigzagged.
# This override keeps the entry boundary clean for the paper figure.
BOUNDARY_OVERRIDES = {
    "boundary_bottom": [
        [19.15, 679.75],
        [1913.50, 899.90],
    ],
}


# Manual label positions for a cleaner figure.
LABEL_POSITIONS = {
    "valid_road_region": (1020, 720),

    "overlay_region_1": (240, 35),
    "overlay_region_2": (1350, 45),
    "overlay_region_3": (1810, 50),
    "overlay_region_4": (160, 910),
    "overlay_region_5": (1600, 950),

    "ignore_region_1": (1245, 290),
    "ignore_region_2": (210, 595),

    "stop_line_near": (1325, 585),

    "boundary_bottom": (680, 700),
    "boundary_far_left": (500, 505),
    "boundary_far_center": (1490, 390),
    "boundary_far_right": (1710, 500),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_point(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def is_point_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(is_point(item) for item in value)


def normalise_polygons(value: Any) -> list[list[list[float]]]:
    if is_point_list(value):
        return [value]

    polygons: list[list[list[float]]] = []
    if isinstance(value, list):
        for item in value:
            if is_point_list(item):
                polygons.append(item)

    return polygons


def to_cv_points(points: list[list[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    array = np.rint(array).astype(np.int32)
    return array.reshape((-1, 1, 2))


def draw_text_box(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    colour: tuple[int, int, int],
) -> None:
    x, y = position

    text_size, baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        FONT_THICKNESS,
    )
    text_width, text_height = text_size

    pad_x = 10
    pad_y = 7

    x = max(0, min(image.shape[1] - text_width - (2 * pad_x), x))
    y = max(text_height + (2 * pad_y), min(image.shape[0] - pad_y, y))

    top_left = (x, y - text_height - baseline - (2 * pad_y))
    bottom_right = (x + text_width + (2 * pad_x), y + baseline + pad_y)

    cv2.rectangle(image, top_left, bottom_right, colour, -1)
    cv2.putText(
        image,
        text,
        (x + pad_x, y - pad_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def apply_polygon_fill(
    image: np.ndarray,
    polygons: list[list[list[float]]],
    colour: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = image.copy()

    for polygon in polygons:
        if len(polygon) < 3:
            continue
        points = to_cv_points(polygon)
        cv2.fillPoly(overlay, [points], colour)

    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, dst=image)


def draw_polygon_outlines(
    image: np.ndarray,
    region_name: str,
    polygons: list[list[list[float]]],
) -> None:
    colour = COLOURS.get(region_name, (220, 220, 220))

    for index, polygon in enumerate(polygons, start=1):
        if len(polygon) < 3:
            continue

        points = to_cv_points(polygon)

        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=colour,
            thickness=REGION_LINE_THICKNESS,
            lineType=cv2.LINE_AA,
        )

        if region_name == "valid_road_region":
            label_key = region_name
        else:
            label_key = f"{region_name}_{index}"

        if label_key in LABEL_POSITIONS:
            label_text = PRETTY_NAMES.get(region_name, region_name)

            if region_name == "overlay_region":
                label_text = f"Overlay region {index}"
            elif region_name == "ignore_region":
                label_text = f"Ignore region {index}"

            draw_text_box(
                image=image,
                text=label_text,
                position=LABEL_POSITIONS[label_key],
                colour=colour,
            )


def draw_region(
    image: np.ndarray,
    region_name: str,
    polygons: list[list[list[float]]],
    alpha: float,
) -> None:
    if not polygons:
        return

    colour = COLOURS.get(region_name, (220, 220, 220))
    apply_polygon_fill(image, polygons, colour, alpha)
    draw_polygon_outlines(image, region_name, polygons)


def draw_boundary(
    image: np.ndarray,
    boundary_name: str,
    payload: dict[str, Any],
) -> None:
    colour = COLOURS.get(boundary_name, (255, 255, 255))
    points = BOUNDARY_OVERRIDES.get(boundary_name, payload.get("centerline", []))

    if len(points) < 2:
        return

    cv_points = to_cv_points(points)

    cv2.polylines(
        image,
        [cv_points],
        isClosed=False,
        color=(255, 255, 255),
        thickness=BOUNDARY_THICKNESS + 5,
        lineType=cv2.LINE_AA,
    )

    cv2.polylines(
        image,
        [cv_points],
        isClosed=False,
        color=colour,
        thickness=BOUNDARY_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    if boundary_name in LABEL_POSITIONS:
        draw_text_box(
            image=image,
            text=PRETTY_NAMES.get(boundary_name, boundary_name),
            position=LABEL_POSITIONS[boundary_name],
            colour=colour,
        )


def draw_scene_regions() -> None:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    scene = load_json(SCENE_JSON_PATH)
    output = image.copy()

    regions = scene.get("regions", {})

    if SHOW_VALID_ROAD and "valid_road_region" in regions:
        draw_region(
            image=output,
            region_name="valid_road_region",
            polygons=normalise_polygons(regions["valid_road_region"].get("polygons", [])),
            alpha=VALID_ROAD_ALPHA,
        )

    if SHOW_IGNORE_REGIONS and "ignore_region" in regions:
        draw_region(
            image=output,
            region_name="ignore_region",
            polygons=normalise_polygons(regions["ignore_region"].get("polygons", [])),
            alpha=IGNORE_REGION_ALPHA,
        )

    if SHOW_STOP_LINE and "stop_line_near" in regions:
        draw_region(
            image=output,
            region_name="stop_line_near",
            polygons=normalise_polygons(regions["stop_line_near"].get("polygons", [])),
            alpha=STOP_LINE_ALPHA,
        )

    if SHOW_OVERLAY_REGIONS and "overlay_region" in regions:
        draw_region(
            image=output,
            region_name="overlay_region",
            polygons=normalise_polygons(regions["overlay_region"].get("polygons", [])),
            alpha=OVERLAY_REGION_ALPHA,
        )

    if SHOW_BOUNDARIES:
        for boundary_name, payload in scene.get("boundaries", {}).items():
            draw_boundary(output, boundary_name, payload)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(OUTPUT_PATH), output)
    if not saved:
        raise RuntimeError(f"Could not save output image: {OUTPUT_PATH}")

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    draw_scene_regions()