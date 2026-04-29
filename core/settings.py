from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AppSettings:
    difficulty: str = "intermediate"
    content_set: str = "asl-ae"
    theme: str = "dark"
    challenge_duration: int = 30
    dual_view: bool = True
    instructor_mode: bool = False
    ai_coach_enabled: bool = False


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return AppSettings(**data)
        except Exception:
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
