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
# OR use the batch file
run.bat
```

### Testing
```bash
# Run all tests with pytest
pytest

# Run specific test files
pytest tests/test_matching.py
pytest tests/test_core_matching.py

# Run individual test modules
python test_imports.py
python test_dialogs.py
```

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Build/Distribution
```bash
# Build installer (Windows)
build_installer.bat
```

## Architecture Overview

### Core Structure
- **`app/`** - Main application package with clean layered architecture
  - **`app/main.py`** - Application entry point with modern UI
  - **`app/config/`** - Configuration management with dataclasses
  - **`app/core/`** - Core entities, constants, and utilities
  - **`app/services/`** - Business logic services (detection, inference, training)
  - **`app/backends/`** - ML model backends (YOLO integration)
  - **`app/ui/`** - Modern Tkinter UI components and windows
  - **`app/utils/`** - Utility functions (geometry, crypto, image processing)

### Key Configuration
- **`config.json`** - Runtime configuration with camera, detection, and AI settings
- **`requirements.txt`** - Python dependencies including OpenCV, PyTorch, Ultralytics
- Configuration managed through `app/config/settings.py` with type-safe dataclasses

### ML/AI Integration
- YOLO models for object detection (YOLOv8, YOLO11, YOLO12)
- Google Gemini API integration for AI analysis
- Custom training pipeline for object classification
- Performance optimization with GPU support

### UI Architecture
- Modern Tkinter-based GUI with themes
- Component-based architecture in `app/ui/components/`
- Separate dialogs for different workflows
- Canvas-based image display with annotations

## Key Services

- **DetectionService** - Object detection using YOLO models
- **InferenceService** - Real-time inference on webcam frames
- **TrainingService** - Custom model training workflows
- **WebcamService** - Camera capture and management
- **GeminiService** - AI-powered image analysis and comparison

## Testing Strategy
- Unit tests in `tests/` directory using pytest
- Integration tests for UI components (`test_dialogs.py`)
- Import validation (`test_imports.py`)
- Core matching algorithm tests (`test_core_matching.py`)

## Data Management
- **`data/`** - Application data directory
  - **`data/models/`** - Trained ML models
  - **`data/master/`** - Reference images
  - **`data/results/`** - Export results
- Settings automatically saved to `config.json`

# Development Guidelines

## Philosophy

### Core Beliefs

- **Incremental progress over big bangs** - Small changes that compile and pass tests
- **Learning from existing code** - Study and plan before implementing
- **Pragmatic over dogmatic** - Adapt to project reality
- **Clear intent over clever code** - Be boring and obvious

### Simplicity Means

- Single responsibility per function/class
- Avoid premature abstractions
- No clever tricks - choose the boring solution
- If you need to explain it, it's too complex

## Process

### 1. Planning & Staging

Break complex work into 3-5 stages. Document in `IMPLEMENTATION_PLAN.md`:

```markdown
## Stage N: [Name]
**Goal**: [Specific deliverable]
**Success Criteria**: [Testable outcomes]
**Tests**: [Specific test cases]
**Status**: [Not Started|In Progress|Complete]
```
- Update status as you progress
- Remove file when all stages are done

### 2. Implementation Flow

1. **Understand** - Study existing patterns in codebase
2. **Test** - Write test first (red)
3. **Implement** - Minimal code to pass (green)
4. **Refactor** - Clean up with tests passing
5. **Commit** - With clear message linking to plan

### 3. When Stuck (After 3 Attempts)

**CRITICAL**: Maximum 3 attempts per issue, then STOP.

1. **Document what failed**:
   - What you tried
   - Specific error messages
   - Why you think it failed

2. **Research alternatives**:
   - Find 2-3 similar implementations
   - Note different approaches used

3. **Question fundamentals**:
   - Is this the right abstraction level?
   - Can this be split into smaller problems?
   - Is there a simpler approach entirely?

4. **Try different angle**:
   - Different library/framework feature?
   - Different architectural pattern?
   - Remove abstraction instead of adding?

## Technical Standards

### Architecture Principles

- **Composition over inheritance** - Use dependency injection
- **Interfaces over singletons** - Enable testing and flexibility
- **Explicit over implicit** - Clear data flow and dependencies
- **Test-driven when possible** - Never disable tests, fix them

### Code Quality

- **Every commit must**:
  - Compile successfully
  - Pass all existing tests
  - Include tests for new functionality
  - Follow project formatting/linting

- **Before committing**:
  - Run formatters/linters
  - Self-review changes
  - Ensure commit message explains "why"

### Error Handling

- Fail fast with descriptive messages
- Include context for debugging
- Handle errors at appropriate level
- Never silently swallow exceptions

## Decision Framework

When multiple valid approaches exist, choose based on:

1. **Testability** - Can I easily test this?
2. **Readability** - Will someone understand this in 6 months?
3. **Consistency** - Does this match project patterns?
4. **Simplicity** - Is this the simplest solution that works?
5. **Reversibility** - How hard to change later?

## Project Integration

### Learning the Codebase

- Find 3 similar features/components
- Identify common patterns and conventions
- Use same libraries/utilities when possible
- Follow existing test patterns

### Tooling

- Use project's existing build system
- Use project's test framework
- Use project's formatter/linter settings
- Don't introduce new tools without strong justification

## Quality Gates

### Definition of Done

- [ ] Tests written and passing
- [ ] Code follows project conventions
- [ ] No linter/formatter warnings
- [ ] Commit messages are clear
- [ ] Implementation matches plan
- [ ] No TODOs without issue numbers

### Test Guidelines

- Test behavior, not implementation
- One assertion per test when possible
- Clear test names describing scenario
- Use existing test utilities/helpers
- Tests should be deterministic

## Important Reminders

**NEVER**:
- Use `--no-verify` to bypass commit hooks
- Disable tests instead of fixing them
- Commit code that doesn't compile
- Make assumptions - verify with existing code

**ALWAYS**:
- Commit working code incrementally
- Update plan documentation as you go
- Learn from existing implementations
- Stop after 3 failed attempts and reassess
