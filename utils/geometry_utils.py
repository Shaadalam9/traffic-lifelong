from __future__ import annotations

import math
from typing import Iterable


def bbox_center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def point_to_polyline_distance(point: tuple[float, float], polyline: list[list[float]]) -> float:
    px, py = point
    distances = []
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        distances.append(point_to_segment_distance(px, py, ax, ay, bx, by))
    return min(distances) if distances else float("inf")
