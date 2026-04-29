from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from urllib.request import Request, urlopen
import cv2
import numpy as np

CONTENT_SETS: Dict[str, List[str]] = {
    "asl-ae": list("ABCDE"),
    "asl-az": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "numbers": list("0123456789"),
}

ALL_SIGNS = CONTENT_SETS["asl-az"] + CONTENT_SETS["numbers"]
DYNAMIC_SIGNS = {"J", "Z"}

SIGN_GUIDE_TEXT: Dict[str, str] = {
    "A": "Fist + thumb outside",
    "B": "Fingers up together",
    "C": "Curved hand shape",
    "D": "Index up, others folded",
    "E": "All fingers folded",
    "F": "Index + thumb make circle, others up",
    "G": "Index and thumb point sideways",
    "H": "Index + middle extended together",
    "I": "Pinky up",
    "J": "Draw J with pinky",
    "K": "Index + middle up, thumb touches middle",
    "L": "Index up + thumb out",
    "M": "Thumb tucked under three fingers",
    "N": "Thumb tucked under two fingers",
    "O": "Round O with fingers and thumb",
    "P": "Like K but angled down",
    "Q": "Like G but angled down",
    "R": "Cross index and middle",
    "S": "Fist with thumb across",
    "T": "Thumb tucked under index",
    "U": "Index + middle together up",
    "V": "Index + middle spread",
    "W": "Three fingers up",
    "X": "Index bent hook",
    "Y": "Thumb + pinky out",
    "Z": "Trace Z with index",
    "0": "Closed O handshape",
    "1": "Index finger up",
    "2": "Index + middle up",
    "3": "Thumb + index + middle up",
    "4": "Four fingers up, thumb in",
    "5": "Open palm",
    "6": "Thumb touches pinky",
    "7": "Thumb touches ring",
    "8": "Thumb touches middle",
    "9": "Thumb touches index",
}

def _sign_image_url(sign: str) -> str:
    if sign in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return f"https://commons.wikimedia.org/wiki/Special:FilePath/Sign_language_{sign}.svg?width=320"
    if sign == "0":
        return (
            "https://commons.wikimedia.org/wiki/"
            "Special:FilePath/Asl_alphabet_gallaudet_%28zero%29.svg?width=300"
        )
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/Sign_language_{sign}.jpg?width=320"


def ensure_local_reference_assets(base_dir: Path) -> Dict[str, str]:
    """Download real sign references; fallback to generated local cards."""
    refs_dir = base_dir / "assets" / "sign_refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, str] = {}
    for sign in ALL_SIGNS:
        target_png = refs_dir / f"{sign}.png"
        target_jpg = refs_dir / f"{sign}.jpg"
        if not target_png.exists() and not target_jpg.exists():
            try:
                request = Request(
                    _sign_image_url(sign),
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                with urlopen(request, timeout=20) as response:
                    content_type = response.headers.get("Content-Type", "").lower()
                    payload = response.read()
                    target = target_jpg if "jpeg" in content_type or "jpg" in content_type else target_png
                    target.write_bytes(payload)
            except Exception:
                pass
        if target_png.exists():
            mapping[sign] = str(target_png)
            continue
        if target_jpg.exists():
            mapping[sign] = str(target_jpg)
            continue
        if not target_png.exists() and not target_jpg.exists():
            fallback_png = refs_dir / f"{sign}.png"
            card = np.full((260, 320, 3), 255, dtype=np.uint8)
            cv2.rectangle(card, (1, 1), (318, 258), (225, 232, 240), 2)
            cv2.putText(card, sign, (125, 125), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (15, 23, 42), 4, cv2.LINE_AA)
            cv2.putText(card, "ASL Reference", (82, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (51, 65, 85), 2, cv2.LINE_AA)
            cv2.imwrite(str(fallback_png), card)
            mapping[sign] = str(fallback_png)
            continue
    return mapping
