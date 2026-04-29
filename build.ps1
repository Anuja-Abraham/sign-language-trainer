param(
  [string]$Python = "python"
)

& $Python -m PyInstaller --noconfirm --onefile --windowed --name sign-language-trainer-desktop main.py
