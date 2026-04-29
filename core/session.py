from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from random import choice
from typing import Dict, List, Optional

from .content import CONTENT_SETS, DYNAMIC_SIGNS
from .dynamic_signs import DynamicMatcher
from .recognizer import DetectionResult


class Mode(str, Enum):
    LEARN = "learn"
    QUIZ = "quiz"
    CHALLENGE = "challenge"
    DRILL_AE = "drill-ae"
    DRILL_BD = "drill-bd"


DIFFICULTY_PRESETS = {
    "beginner": {"confidence": 55, "holdFrames": 14},
    "intermediate": {"confidence": 60, "holdFrames": 20},
    "advanced": {"confidence": 70, "holdFrames": 28},
}

# Per-sign threshold tuning for practical accuracy without model training.
# Negative values reduce required confidence for hard/confusable signs.
SIGN_THRESHOLD_OFFSETS = {
    "M": -8,
    "N": -8,
    "T": -8,
    "U": -6,
    "V": -6,
    "R": -6,
    "6": -8,
    "7": -8,
    "8": -8,
    "9": -8,
    "J": -10,
    "Z": -10,
}


@dataclass
class Stats:
    attempts: int = 0
    correct: int = 0
    streak: int = 0
    best_streak: int = 0
    level_completions: int = 0
    challenge_score: int = 0


@dataclass
class SessionState:
    content_key: str = "asl-ae"
    mode: Mode = Mode.LEARN
    difficulty: str = "intermediate"
    challenge_duration: int = 30
    instructor_mode: bool = False
    dual_view: bool = True
    target_index: int = 0
    hold_frames: int = 0
    target_success_count: int = 0
    previous_detected_sign: str = "-"
    distance_scale: float = 1.0
    completed_signs: set[str] = field(default_factory=set)
    stats: Stats = field(default_factory=Stats)
    timed_remaining: int = 0

    @property
    def current_signs(self) -> List[str]:
        return CONTENT_SETS.get(self.content_key, CONTENT_SETS["asl-ae"])

    @property
    def target(self) -> str:
        signs = self.current_signs
        return signs[self.target_index] if signs else "A"


class SessionController:
    def __init__(self) -> None:
        self.state = SessionState()
        self.dynamic = DynamicMatcher()

    def set_content_set(self, key: str) -> None:
        self.state.content_key = key if key in CONTENT_SETS else "asl-ae"
        self.state.target_index = 0
        self.state.hold_frames = 0
        self.state.target_success_count = 0
        self.state.completed_signs.clear()
        self.dynamic.reset()

    def set_mode(self, mode: Mode) -> None:
        self.state.mode = mode
        if mode == Mode.LEARN:
            self.state.timed_remaining = 0

    def set_target(self, index: int) -> None:
        signs = self.state.current_signs
        if not signs:
            return
        self.state.target_index = index % len(signs)
        self.state.hold_frames = 0
        self.state.target_success_count = 0
        self.dynamic.reset()

    def set_random_target(self, only_from: Optional[List[str]] = None) -> None:
        pool = only_from if only_from else self.state.current_signs
        if not pool:
            return
        target = choice(pool)
        self.set_target(self.state.current_signs.index(target))

    def set_difficulty(self, level: str) -> None:
        self.state.difficulty = level if level in DIFFICULTY_PRESETS else "intermediate"

    def set_calibration_scale(self, scale: float) -> None:
        self.state.distance_scale = max(0.8, min(1.25, scale))

    def start_timed_mode(self, mode: Mode, seconds: int, random_pool: Optional[List[str]] = None) -> None:
        self.set_mode(mode)
        self.state.timed_remaining = max(0, seconds)
        self.set_random_target(random_pool)
        if mode == Mode.CHALLENGE:
            self.state.stats.challenge_score = 0

    def tick_timer(self) -> bool:
        if self.state.timed_remaining <= 0:
            return False
        self.state.timed_remaining -= 1
        if self.state.timed_remaining <= 0:
            self.set_mode(Mode.LEARN)
            return True
        return False

    def process_result(self, result: DetectionResult, motion_ok: bool) -> Dict[str, object]:
        target = self.state.target
        diff = DIFFICULTY_PRESETS[self.state.difficulty]
        threshold = round(diff["confidence"] * self.state.distance_scale)
        threshold += SIGN_THRESHOLD_OFFSETS.get(target, 0)
        threshold = max(35, min(95, threshold))

        if result.sign != "-" and result.sign != self.state.previous_detected_sign:
            self._record_attempt(result.sign == target)
        self.state.previous_detected_sign = result.sign

        is_dynamic = target in DYNAMIC_SIGNS
        is_correct = result.sign == target and round(result.confidence * 100) >= threshold and (motion_ok or not is_dynamic)
        if is_correct:
            self.state.hold_frames = 0
            self.state.target_success_count = 0
            self.state.completed_signs.add(target)
            self.state.stats.level_completions += 1
            status = f"Threshold reached for {target}. Moving to next sign."
            auto_advanced = False
            if self.state.mode == Mode.CHALLENGE:
                self.state.stats.challenge_score += 1
            if self.state.mode in {Mode.QUIZ, Mode.CHALLENGE, Mode.DRILL_AE, Mode.DRILL_BD}:
                pool = self.resolve_target_pool_for_mode()
                self.set_random_target(pool)
                auto_advanced = True
            elif self.state.mode == Mode.LEARN:
                self.set_target(self.state.target_index + 1)
                auto_advanced = True
            return {"progress": 100, "status": status, "mastered": True, "auto_advanced": auto_advanced}

        self.state.hold_frames = 0
        return {
            "progress": 0,
            "status": (
                f"Practice {target}: hold handshape and draw motion ({self.state.target_success_count}/3 completed)"
                if is_dynamic
                else f"Practice {target} ({self.state.target_success_count}/3 completed)"
            ),
            "mastered": False,
            "auto_advanced": False,
        }

    def resolve_target_pool_for_mode(self) -> Optional[List[str]]:
        if self.state.mode == Mode.DRILL_AE:
            return ["A", "E"]
        if self.state.mode == Mode.DRILL_BD:
            return ["B", "D"]
        return None

    def _record_attempt(self, correct: bool) -> None:
        self.state.stats.attempts += 1
        if correct:
            self.state.stats.correct += 1
            self.state.stats.streak += 1
            self.state.stats.best_streak = max(self.state.stats.best_streak, self.state.stats.streak)
        else:
            self.state.stats.streak = 0
