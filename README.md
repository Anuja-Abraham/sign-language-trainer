# Python Sign Language Trainer (Desktop)

Desktop ASL trainer built with Python + PySide6 + MediaPipe.

## Features

- Live webcam hand tracking
- Targeted sign practice for `A-Z` and `0-9`
- Auto-calibration on camera start
- Auto-advance to next sign once threshold is met
- Learn / Quiz / Challenge / Drill modes
- Previous / Next sign controls in Learn mode
- AI Coach Tips (Gemini API, optional)
- Local sign reference assets with fallback rendering

## Requirements

- Windows 10/11
- Python 3.11+ (3.12+ recommended)
- Webcam
- Internet (first run downloads MediaPipe model and sign references)

## Quick Start (New System)

### 1) Clone

```powershell
git clone https://github.com/Anuja-Abraham/sign-language-trainer.git
cd sign-language-trainer
```

### 2) Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
pip install -r requirements.txt
```

### 4) (Optional) Configure Gemini AI Coach

Create `.env` in project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5) Run

```powershell
python main.py
```

## Usage Flow

1. Click `Start Camera`
2. Wait for auto-calibration countdown to finish
3. Follow `Targeted Sign`
4. Hold correct sign until threshold is crossed
5. App shifts automatically to next sign

## Project Structure

- `main.py` - app bootstrap + `.env` loading
- `ui/main_window.py` - desktop UI and workflow wiring
- `core/content.py` - content sets + reference assets
- `core/recognizer.py` - sign recognition logic
- `core/disambiguation.py` - confusion-group rules
- `core/dynamic_signs.py` - motion checks for `J` / `Z`
- `core/session.py` - progression, thresholds, timers, modes
- `core/ai_coach.py` - Gemini AI hint integration
- `core/settings.py` - persisted app settings
- `config/settings.json` - runtime preferences
- `tests/test_session.py` - session/disambiguation tests

## Testing

```powershell
pytest -q
```

## Build Executable (Windows)

```powershell
pyinstaller --noconfirm --onefile --windowed --name sign-language-trainer-desktop main.py
```

Generated output:

- `dist/sign-language-trainer-desktop.exe`

## Troubleshooting

- `ModuleNotFoundError`: activate `.venv` first.
- Blank/missing reference image: restart app to regenerate local assets.
- Camera not opening: close other apps using webcam and retry.
- AI coach not responding: verify `GEMINI_API_KEY` and internet access.

## Maintenance

- Keep `.env` local and never commit API keys.
