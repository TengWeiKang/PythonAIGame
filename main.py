"""New main entry point using refactored structure."""

import sys
import os
from pathlib import Path

# Add the current directory to path for app imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the new main function
from app.main import main

if __name__ == "__main__":
    sys.exit(main())