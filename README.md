# Python Sign Language Trainer (Desktop)

Python desktop rewrite of the sign-language trainer with:

- MediaPipe hand landmarks + OpenCV camera pipeline
- Recognition heuristics for `A-Z` and `0-9`
- Smoothing and lookalike disambiguation (`U/V/R`, `M/N/T`, `6/7/8/9`)
- Dynamic sign motion gates for `J` and `Z`
- Learn/Quiz/Challenge/Drill modes
- Calibration scaling, mastery progression, stats
- Local reference sign cards generated in `assets/sign_refs/`

## Project Layout

- `main.py` - application entry point
- `ui/main_window.py` - desktop UI and event wiring
- `core/recognizer.py` - sign detection + smoothing
- `core/disambiguation.py` - confusion-group refinement
- `core/dynamic_signs.py` - `J`/`Z` motion matching
- `core/session.py` - mode/timer/progression logic
- `core/settings.py` - settings persistence
- `core/content.py` - content sets, guide text, local assets bootstrap
- `tests/test_session.py` - regression tests

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Test

```bash
pytest -q
```

## Build Executable (Windows)

```bash
pyinstaller --noconfirm --onefile --windowed --name sign-language-trainer-desktop main.py
```

Output binary will be under `dist/`.

## Notes

- First run generates local reference SVG cards for all signs.
- Webcam permission is required.
- Theme support is intentionally simple (`dark`/`light`) in v1 desktop rewrite.
