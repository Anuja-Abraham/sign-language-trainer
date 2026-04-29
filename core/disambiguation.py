from __future__ import annotations

from typing import Dict


def refine_sign(candidate: str, meta: Dict[str, float | bool]) -> str:
    m = meta or {}

    if (
        candidate in {"U", "V", "R"}
        or (
            m.get("openCount") == 2
            and m.get("indexUp")
            and m.get("middleUp")
            and not m.get("ringUp")
            and not m.get("pinkyUp")
        )
    ):
        if m.get("indexMiddleCrossed") or float(m.get("indexMiddleGap", 1.0)) < 0.16:
            return "R"
        if float(m.get("indexMiddleGap", 0.0)) > 0.34:
            return "V"
        return "U"

    if candidate in {"M", "N", "T"} or (m.get("openCount") == 0 and not m.get("thumbOut", True)):
        distances = [
            ("T", float(m.get("thumbToIndexMcp", 10.0))),
            ("N", float(m.get("thumbToMiddleMcp", 10.0))),
            ("M", float(m.get("thumbToRingMcp", 10.0))),
        ]
        distances.sort(key=lambda x: x[1])
        if distances[0][1] < 0.55:
            return distances[0][0]

    if candidate in {"6", "7", "8", "9"} or (m.get("openCount") == 3 and not m.get("thumbOut", True)):
        touches = [
            ("9", float(m.get("thumbToIndexTip", 10.0))),
            ("8", float(m.get("thumbToMiddleTip", 10.0))),
            ("7", float(m.get("thumbToRingTip", 10.0))),
            ("6", float(m.get("thumbToPinkyTip", 10.0))),
        ]
        touches.sort(key=lambda x: x[1])
        if touches[0][1] < 0.45:
            return touches[0][0]

    return candidate
