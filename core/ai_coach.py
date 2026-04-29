from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


@dataclass
class AICoach:
    api_key: str
    model: str = "gemini-2.0-flash"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key.strip())

    def get_tip(self, payload: Dict[str, object]) -> str:
        if not self.enabled:
            return "AI coach unavailable: missing GEMINI_API_KEY."

        target = str(payload.get("target", "-"))
        detected = str(payload.get("detected", "-"))
        confidence = int(payload.get("confidence", 0))
        status = str(payload.get("status", ""))
        coach_tip = str(payload.get("coach_tip", ""))

        prompt = (
            "You are an ASL hand-sign coach. Give one short corrective hint (max 14 words). "
            "Focus on actionable hand-shape or movement correction.\n"
            f"Target sign: {target}\n"
            f"Detected sign: {detected}\n"
            f"Confidence: {confidence}%\n"
            f"Current app status: {status}\n"
            f"Current heuristic coach tip: {coach_tip}\n"
            "Return only the hint sentence."
        )

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 40},
        }
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        req = Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req, timeout=12) as res:
                response = json.loads(res.read().decode("utf-8"))
            candidates = response.get("candidates", [])
            if not candidates:
                return "AI coach: no response. Keep hand centered and retry."
            parts = candidates[0].get("content", {}).get("parts", [])
            text = (parts[0].get("text", "") if parts else "").strip()
            return text or "AI coach: no usable tip. Keep hand steady."
        except (HTTPError, URLError, TimeoutError):
            return "AI coach unavailable right now. Check internet/API key."
        except Exception:
            return "AI coach error. Continue with on-screen coach tip."
