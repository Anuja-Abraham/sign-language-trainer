from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ml_features import extract_features


def init_landmarker(model_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            model_path,
        )
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture landmark samples for ASL model training.")
    parser.add_argument("--label", required=True, help="Sign label to capture (e.g. A, B, 1, 2)")
    parser.add_argument("--out", default="data/landmarks.csv", help="Output CSV path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path("assets/models/hand_landmarker.task")
    landmarker = init_landmarker(model_path)
    cap = cv2.VideoCapture(0)

    header = ["label"] + [f"f{i}" for i in range(63)]
    file_exists = out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if not file_exists:
            writer.writerow(header)

        samples = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            has_hand = bool(result.hand_landmarks)

            msg = f"Label={args.label} Samples={samples} | SPACE=save | Q=quit"
            cv2.putText(frame, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 80), 2)
            cv2.putText(
                frame,
                "Hand detected" if has_hand else "No hand",
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 0) if has_hand else (0, 0, 255),
                2,
            )
            cv2.imshow("Capture Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and has_hand:
                lm = result.hand_landmarks[0]
                landmarks = [(p.x, p.y, p.z) for p in lm]
                feats = extract_features(landmarks)
                writer.writerow([args.label] + feats)
                samples += 1
                fp.flush()

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
