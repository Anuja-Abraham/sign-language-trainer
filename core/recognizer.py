from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import Deque, Dict, List, Optional, Tuple

from .content import ALL_SIGNS, SIGN_GUIDE_TEXT
from .disambiguation import refine_sign


@dataclass
class DetectionResult:
    sign: str
    confidence: float
    hint: str
    meta: Dict[str, float | bool]


class Recognizer:
    def __init__(self, smoothing_window: int = 6) -> None:
        self.smoothing_window = smoothing_window
        self.recent: Deque[DetectionResult] = deque(maxlen=smoothing_window)
        self.profiles = self._profiles()

    def reset(self) -> None:
        self.recent.clear()

    @staticmethod
    def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)

    def _finger_extended(self, landmarks: List[Tuple[float, float, float]], tip: int, pip: int, mcp: int) -> bool:
        tip_to_wrist = self._distance(landmarks[tip], landmarks[0])
        pip_to_wrist = self._distance(landmarks[pip], landmarks[0])
        mcp_to_wrist = self._distance(landmarks[mcp], landmarks[0])
        return tip_to_wrist > pip_to_wrist > mcp_to_wrist

    def _thumb_out(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        return self._distance(landmarks[4], landmarks[0]) > self._distance(landmarks[2], landmarks[0]) * 1.2

    @staticmethod
    def _guide(sign: str) -> str:
        return SIGN_GUIDE_TEXT.get(sign, f"Practice {sign} handshape")

    def detect_sign(self, landmarks: List[Tuple[float, float, float]]) -> DetectionResult:
        index_up = self._finger_extended(landmarks, 8, 6, 5)
        middle_up = self._finger_extended(landmarks, 12, 10, 9)
        ring_up = self._finger_extended(landmarks, 16, 14, 13)
        pinky_up = self._finger_extended(landmarks, 20, 18, 17)
        thumb_out = self._thumb_out(landmarks)
        open_count = sum([index_up, middle_up, ring_up, pinky_up])
        palm_scale = max(self._distance(landmarks[5], landmarks[17]), 0.01)

        def norm(a: int, b: int) -> float:
            return self._distance(landmarks[a], landmarks[b]) / palm_scale

        avg_curl = (norm(8, 5) + norm(12, 9) + norm(16, 13) + norm(20, 17)) / 4.0
        meta: Dict[str, float | bool] = {
            "indexUp": index_up,
            "middleUp": middle_up,
            "ringUp": ring_up,
            "pinkyUp": pinky_up,
            "thumbOut": thumb_out,
            "openCount": open_count,
            "avgCurl": avg_curl,
            "pinchIndex": norm(4, 8),
            "thumbMiddle": norm(4, 12),
            "indexMiddleGap": norm(8, 12),
            "thumbToIndexMcp": norm(4, 5),
            "thumbToMiddleMcp": norm(4, 9),
            "thumbToRingMcp": norm(4, 13),
            "thumbToIndexTip": norm(4, 8),
            "thumbToMiddleTip": norm(4, 12),
            "thumbToRingTip": norm(4, 16),
            "thumbToPinkyTip": norm(4, 20),
            "indexMiddleCrossed": (landmarks[8][0] - landmarks[12][0]) * (landmarks[5][0] - landmarks[9][0]) < 0,
        }

        fingers = [index_up, middle_up, ring_up, pinky_up]
        best_sign = "-"
        best_score = 0.0
        for sign in ALL_SIGNS:
            profile = self.profiles.get(sign)
            score = self._score_profile(profile, fingers, thumb_out, open_count, avg_curl, meta)
            if score > best_score:
                best_score, best_sign = score, sign

        if best_score < 0.45:
            result = DetectionResult("-", 0.2, "Adjust hand to match the target sign.", meta)
        else:
            refined = refine_sign(best_sign, meta)
            result = DetectionResult(refined, max(0.35, min(0.95, best_score)), self._guide(refined), meta)
        return self._smooth(result)

    def _smooth(self, result: DetectionResult) -> DetectionResult:
        self.recent.append(result)
        buckets: Dict[str, Dict[str, object]] = {}
        for item in self.recent:
            bucket = buckets.setdefault(item.sign, {"count": 0, "sum": 0.0, "meta": item.meta})
            bucket["count"] = int(bucket["count"]) + 1
            bucket["sum"] = float(bucket["sum"]) + item.confidence
            bucket["meta"] = item.meta
        best_sign = result.sign
        best_count = -1
        best_sum = -1.0
        best_meta = result.meta
        for sign, value in buckets.items():
            count = int(value["count"])
            conf_sum = float(value["sum"])
            if count > best_count or (count == best_count and conf_sum > best_sum):
                best_sign, best_count, best_sum, best_meta = sign, count, conf_sum, value["meta"]  # type: ignore
        if best_sign == "-":
            return DetectionResult("-", 0.2, result.hint, best_meta)
        avg_conf = best_sum / max(best_count, 1)
        return DetectionResult(best_sign, max(0.3, min(0.95, avg_conf)), self._guide(best_sign), best_meta)

    @staticmethod
    def _score_profile(
        profile: Dict[str, object],
        fingers: List[bool],
        thumb_out: bool,
        open_count: int,
        avg_curl: float,
        meta: Dict[str, float | bool],
    ) -> float:
        if not profile:
            return 0.0
        score = 0.2
        checks = 1
        p_fingers = profile.get("fingers")
        if isinstance(p_fingers, list):
            for i, expected in enumerate(p_fingers):
                if expected is None:
                    continue
                checks += 1
                if bool(expected) == fingers[i]:
                    score += 1
        if "thumbOut" in profile:
            checks += 1
            if bool(profile["thumbOut"]) == thumb_out:
                score += 1
        open_range = profile.get("openRange")
        if isinstance(open_range, list) and len(open_range) == 2:
            checks += 1
            if open_range[0] <= open_count <= open_range[1]:
                score += 1
        curve_range = profile.get("curveRange")
        if isinstance(curve_range, list) and len(curve_range) == 2:
            checks += 1
            if curve_range[0] <= avg_curl <= curve_range[1]:
                score += 1
        pinch_range = profile.get("pinchIndex")
        if isinstance(pinch_range, list) and len(pinch_range) == 2:
            checks += 1
            if pinch_range[0] <= float(meta["pinchIndex"]) <= pinch_range[1]:
                score += 1
        thumb_middle = profile.get("thumbMiddle")
        if isinstance(thumb_middle, list) and len(thumb_middle) == 2:
            checks += 1
            if thumb_middle[0] <= float(meta["thumbMiddle"]) <= thumb_middle[1]:
                score += 1
        return score / checks

    @staticmethod
    def _profiles() -> Dict[str, Dict[str, object]]:
        return {
            "A": {"fingers": [0, 0, 0, 0], "thumbOut": True, "openRange": [0, 1], "curveRange": [0.2, 0.5]},
            "B": {"fingers": [1, 1, 1, 1], "thumbOut": True, "openRange": [4, 4], "curveRange": [0.5, 1.4]},
            "C": {"fingers": [None, None, None, None], "thumbOut": True, "curveRange": [0.65, 1.1]},
            "D": {"fingers": [1, 0, 0, 0], "thumbOut": True, "openRange": [1, 2]},
            "E": {"fingers": [0, 0, 0, 0], "thumbOut": False, "openRange": [0, 0], "curveRange": [0.15, 0.55]},
            "F": {"fingers": [0, 1, 1, 1], "pinchIndex": [0, 0.35], "openRange": [2, 3]},
            "G": {"fingers": [1, 0, 0, 0], "pinchIndex": [0.35, 0.9], "openRange": [1, 2]},
            "H": {"fingers": [1, 1, 0, 0], "openRange": [2, 2]},
            "I": {"fingers": [0, 0, 0, 1], "openRange": [1, 1]},
            "J": {"fingers": [0, 0, 0, 1], "openRange": [1, 1]},
            "K": {"fingers": [1, 1, 0, 0], "thumbMiddle": [0, 0.45], "openRange": [2, 3]},
            "L": {"fingers": [1, 0, 0, 0], "thumbOut": True, "openRange": [1, 2]},
            "M": {"fingers": [0, 0, 0, 0], "thumbOut": False, "openRange": [0, 0]},
            "N": {"fingers": [0, 0, 0, 0], "thumbOut": False, "openRange": [0, 0]},
            "O": {"fingers": [None, None, None, None], "curveRange": [0.45, 0.8], "pinchIndex": [0, 0.35]},
            "P": {"fingers": [1, 1, 0, 0], "thumbMiddle": [0, 0.45], "openRange": [2, 3]},
            "Q": {"fingers": [1, 0, 0, 0], "pinchIndex": [0.35, 0.9], "openRange": [1, 2]},
            "R": {"fingers": [1, 1, 0, 0], "openRange": [2, 2]},
            "S": {"fingers": [0, 0, 0, 0], "thumbOut": True, "openRange": [0, 1]},
            "T": {"fingers": [0, 0, 0, 0], "thumbOut": False, "openRange": [0, 0]},
            "U": {"fingers": [1, 1, 0, 0], "openRange": [2, 2]},
            "V": {"fingers": [1, 1, 0, 0], "openRange": [2, 2]},
            "W": {"fingers": [1, 1, 1, 0], "openRange": [3, 3]},
            "X": {"fingers": [1, 0, 0, 0], "openRange": [1, 2], "curveRange": [0.3, 0.75]},
            "Y": {"fingers": [0, 0, 0, 1], "thumbOut": True, "openRange": [1, 2]},
            "Z": {"fingers": [1, 0, 0, 0], "openRange": [1, 2]},
            "0": {"fingers": [None, None, None, None], "curveRange": [0.45, 0.8], "pinchIndex": [0, 0.35]},
            "1": {"fingers": [1, 0, 0, 0], "openRange": [1, 1]},
            "2": {"fingers": [1, 1, 0, 0], "openRange": [2, 2]},
            "3": {"fingers": [1, 1, 0, 0], "thumbOut": True, "openRange": [2, 3]},
            "4": {"fingers": [1, 1, 1, 1], "thumbOut": False, "openRange": [4, 4]},
            "5": {"fingers": [1, 1, 1, 1], "thumbOut": True, "openRange": [4, 4]},
            "6": {"fingers": [1, 1, 1, 0], "thumbOut": False, "openRange": [3, 3]},
            "7": {"fingers": [1, 1, 0, 1], "thumbOut": False, "openRange": [3, 3]},
            "8": {"fingers": [1, 0, 1, 1], "thumbOut": False, "openRange": [3, 3]},
            "9": {"fingers": [0, 1, 1, 1], "pinchIndex": [0, 0.35], "openRange": [2, 3]},
        }
