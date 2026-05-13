#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


# Edit these values only
ANNOTATIONS_XML = "annotations.xml"
OUTPUT_JSON = "scene_regions.json"
IMAGE_NAME = None  # set to an image name like "frame_05_31_36.png" or leave as None for the first image
ROUND_DIGITS = 2

# Labels expected from CVAT
BOUNDARY_LABELS = [
    "gate_bottom",
    "gate_far_left",
    "gate_far_center",
    "gate_far_right",
]
OPTIONAL_REGION_LABELS = [
    "stop_line_near",
    "queue_area_near",
    "ignore_region",
    "overlay_region",
    "valid_road_region",
]


Point = tuple[float, float]
Polyline = list[Point]
Polygon = list[Point]


def parse_points(points_text: str) -> list[Point]:
    points: list[Point] = []
    for item in points_text.strip().split(";"):
        if not item:
            continue
        x_text, y_text = item.split(",")
        points.append((float(x_text), float(y_text)))
    return points


def round_point(point: Point, digits: int = ROUND_DIGITS) -> list[float]:
    return [round(point[0], digits), round(point[1], digits)]


def round_points(points: list[Point], digits: int = ROUND_DIGITS) -> list[list[float]]:
    return [round_point(point, digits=digits) for point in points]


def polygon_area(points: Polygon) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def dedupe_consecutive(points: list[Point], epsilon: float = 1e-6) -> list[Point]:
    if not points:
        return []
    deduped = [points[0]]
    for point in points[1:]:
        prev = deduped[-1]
        if math.hypot(point[0] - prev[0], point[1] - prev[1]) > epsilon:
            deduped.append(point)
    if len(deduped) > 1:
        first = deduped[0]
        last = deduped[-1]
        if math.hypot(first[0] - last[0], first[1] - last[1]) <= epsilon:
            deduped.pop()
    return deduped


def polygon_to_centerline(points: Polygon) -> Polyline:
    """
    Convert a thin strip polygon exported from CVAT into a usable centerline.

    This works well for the gate polygons in this project because they were drawn as
    long narrow strips with vertices ordered around the perimeter.
    """
    clean = dedupe_consecutive(points)
    if len(clean) < 2:
        return clean

    if len(clean) == 2:
        return clean

    midpoints: list[Point] = []
    half = len(clean) // 2
    for index in range(half):
        p1 = clean[index]
        p2 = clean[-(index + 1)]
        midpoint = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
        midpoints.append(midpoint)

    if len(clean) % 2 == 1:
        midpoints.append(clean[half])

    # Sort along dominant axis so the polyline is ordered end to end
    xs = [point[0] for point in midpoints]
    ys = [point[1] for point in midpoints]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    if x_span >= y_span:
        midpoints.sort(key=lambda point: point[0])
    else:
        midpoints.sort(key=lambda point: point[1])

    return dedupe_consecutive(midpoints)


def polyline_length(points: Polyline) -> float:
    if len(points) < 2:
        return 0.0
    return sum(
        math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
        for index in range(len(points) - 1)
    )


def select_image(root: ET.Element, image_name: str | None) -> ET.Element:
    images = root.findall(".//image")
    if not images:
        raise ValueError("No <image> elements found in the CVAT XML.")

    if image_name is None:
        return images[0]

    for image in images:
        if image.attrib.get("name") == image_name:
            return image

    available = [image.attrib.get("name", "") for image in images]
    raise ValueError(f"Image name {image_name!r} not found. Available images: {available}")


def collect_shapes(image: ET.Element) -> dict[str, list[dict[str, Any]]]:
    shapes: dict[str, list[dict[str, Any]]] = {}
    for element in image:
        if element.tag not in {"polygon", "polyline"}:
            continue
        label = element.attrib["label"]
        shape = {
            "type": element.tag,
            "points": parse_points(element.attrib["points"]),
            "z_order": int(element.attrib.get("z_order", "0")),
        }
        shapes.setdefault(label, []).append(shape)
    return shapes


def select_largest_polygon(shapes: list[dict[str, Any]]) -> dict[str, Any]:
    polygons = [shape for shape in shapes if shape["type"] == "polygon"]
    if not polygons:
        raise ValueError("Expected at least one polygon shape.")
    return max(polygons, key=lambda shape: polygon_area(shape["points"]))


def build_scene_config(xml_path: Path, image_name: str | None = None) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    image = select_image(root, image_name=image_name)
    shapes_by_label = collect_shapes(image)

    scene: dict[str, Any] = {
        "source": {
            "annotations_xml": str(xml_path),
            "image_name": image.attrib.get("name"),
        },
        "frame": {
            "width": int(image.attrib["width"]),
            "height": int(image.attrib["height"]),
        },
        "regions": {},
        "boundaries": {},
    }

    # Regions
    if "valid_road_region" not in shapes_by_label:
        raise ValueError("Missing valid_road_region in the exported annotations.")
    scene["regions"]["valid_road_region"] = {
        "polygons": round_points(select_largest_polygon(shapes_by_label["valid_road_region"])["points"])
    }

    for label in ["overlay_region", "ignore_region"]:
        if label in shapes_by_label:
            polygons = [
                round_points(shape["points"])
                for shape in shapes_by_label[label]
                if shape["type"] == "polygon"
            ]
            scene["regions"][label] = {"polygons": polygons}

    for label in ["stop_line_near", "queue_area_near"]:
        if label in shapes_by_label:
            scene["regions"][label] = {
                "polygons": [
                    round_points(shape["points"])
                    for shape in shapes_by_label[label]
                    if shape["type"] == "polygon"
                ]
            }

    # Boundaries converted from thin polygons into centerlines
    for label in BOUNDARY_LABELS:
        if label not in shapes_by_label:
            continue
        shape = select_largest_polygon(shapes_by_label[label])
        centerline = polygon_to_centerline(shape["points"])
        scene["boundaries"][label.replace("gate_", "boundary_")] = {
            "strip_polygon": round_points(shape["points"]),
            "centerline": round_points(centerline),
            "length_px": round(polyline_length(centerline), ROUND_DIGITS),
        }

    return scene


def save_scene_config(scene: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(scene, handle, indent=2)


def main() -> None:
    xml_path = Path(ANNOTATIONS_XML).expanduser()
    output_path = Path(OUTPUT_JSON).expanduser()

    if not xml_path.exists():
        raise FileNotFoundError(f"Could not find annotations XML: {xml_path}")

    scene = build_scene_config(xml_path, image_name=IMAGE_NAME)
    save_scene_config(scene, output_path)

    print(f"Saved scene config to: {output_path}")
    print("Boundaries found:", ", ".join(scene["boundaries"].keys()))
    print("Region keys:", ", ".join(scene["regions"].keys()))


if __name__ == "__main__":
    main()


from utils.base import PipelineContext, PipelineStage


class SceneConfigBuilder(PipelineStage):
    def run(self) -> None:
        global ANNOTATIONS_XML, OUTPUT_JSON, IMAGE_NAME, ROUND_DIGITS

        ANNOTATIONS_XML = str(self.context.annotations_xml)
        OUTPUT_JSON = str(self.context.scene_regions_json)
        IMAGE_NAME = self.context.scene_image_name
        ROUND_DIGITS = self.context.scene_round_digits

        self.logger.info("Starting scene config export for {}.", ANNOTATIONS_XML)
        main()
