# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python desktop application called "Webcam Master Checker" - a local Windows app for schools that captures webcam video, trains object detection models, compares live frames to reference images, and provides educational feedback.

## Common Development Commands

### Running the Application
```bash
# Activate virtual environment (if not already active)
.\.venv\Scripts\activate

# Run the main application
python main.py
```

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Configuration
- **`config.json`** - Runtime configuration with camera, detection, and AI settings
- **`requirements.txt`** - Python dependencies including OpenCV, PyTorch, Ultralytics

### ML/AI Integration
- YOLO models for object detection (custom model)
- Google Gemini API integration for AI analysis
- Custom training pipeline for object classification
- Performance optimization with GPU support
- Model file named as `model.pt`

### UI Architecture
- Modern Tkinter-based GUI with themes
- Separate dialogs for different workflows
- Canvas-based image display with annotations