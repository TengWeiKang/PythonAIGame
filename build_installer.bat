@echo off
REM Simple PyInstaller example. Adjust for data files and icons.

REM Activate venv first:
REM .\.venv\Scripts\activate

pip install pyinstaller
pyinstaller --noconfirm --clean --onefile --windowed --add-data "data;data" main.py
echo Build done. Check the dist folder for main.exe
pause
