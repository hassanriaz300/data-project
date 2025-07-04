# src/config.py

# … your existing imports/constants …

from pathlib import Path

# e.g. DATA_DIR, API keys, etc.

# -----------------------------------------------------------------------------
# Add this:
MODELS_DIR = Path(__file__).parent.parent / "models"
