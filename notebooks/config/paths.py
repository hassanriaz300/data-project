import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
CLEANED_DIR = os.path.join(INTERIM_DIR, "processed")

SEMANTIC_DIR = os.path.join(INTERIM_DIR, "semantic_mapped")
os.makedirs(SEMANTIC_DIR, exist_ok=True)
