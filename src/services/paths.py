# This module defines the directory paths used in the project, ensuring that all file operations are consistent and organized.
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
CLEANED_DIR = os.path.join(INTERIM_DIR, "processed")
SEMANTIC_DIR = os.path.join(INTERIM_DIR, "semantic_mapped")
VISUALIZATION_DIR = os.path.join(INTERIM_DIR, "visualization")

os.makedirs(SEMANTIC_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)