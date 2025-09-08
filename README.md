# Webcam Master Checker (Windows)

**Purpose:** local Windows desktop app for schools. Capture webcam video, train object classes, set a "master image" with expected object positions/angles, and run live detection that compares webcam frames to the master image. Outputs kid-friendly feedback and coordinates/angle for each trained object.

## Quick start (Windows)
1. Install Python 3.13.7 and Git.
2. Create venv and install:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
