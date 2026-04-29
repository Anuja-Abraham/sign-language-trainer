from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Deque, Dict, Iterable, Tuple


Point = Tuple[float, float]


@dataclass
class TrailStats:
    x_range: float
    y_range: float
    path: float
    x_turns: int


def _trail_stats(trail: Iterable[Point]) -> TrailStats:
    points = list(trail)
    if not points:
        return TrailStats(0.0, 0.0, 0.0, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    path = 0.0
    x_turns = 0
    last_sign = 0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        path += hypot(dx, dy)
        sign = 1 if dx > 0.004 else -1 if dx < -0.004 else 0
        if sign and last_sign and sign != last_sign:
            x_turns += 1
        if sign:
            last_sign = sign
    return TrailStats(max(xs) - min(xs), max(ys) - min(ys), path, x_turns)


class DynamicMatcher:
    def __init__(self, max_points: int = 24) -> None:
        self.max_points = max_points
        self.index_tip: Deque[Point] = deque(maxlen=max_points)
        self.pinky_tip: Deque[Point] = deque(maxlen=max_points)

    def reset(self) -> None:
        self.index_tip.clear()
        self.pinky_tip.clear()

    def update(self, landmarks: Dict[int, Point]) -> None:
        self.index_tip.append(landmarks[8])
        self.pinky_tip.append(landmarks[20])

    def matches_motion(self, target: str) -> bool:
        if target == "J":
            s = _trail_stats(self.pinky_tip)
            return s.path > 0.16 and s.x_range > 0.045 and s.y_range > 0.06
        if target == "Z":
            s = _trail_stats(self.index_tip)
            return s.path > 0.2 and s.x_range > 0.09 and s.y_range > 0.05 and s.x_turns >= 2
        return True
