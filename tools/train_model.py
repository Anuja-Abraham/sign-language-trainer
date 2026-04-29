from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ASL landmark classifier.")
    parser.add_argument("--data", default="data/landmarks.csv", help="Training CSV path")
    parser.add_argument("--out", default="models/asl_landmark_model.joblib", help="Output model path")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if "label" not in df.columns:
        raise SystemExit("CSV must include 'label' column.")
    x = df.drop(columns=["label"]).values
    y = df["label"].astype(str).values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    print(f"Validation accuracy: {acc:.4f}")
    print(classification_report(y_test, pred))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Saved model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
