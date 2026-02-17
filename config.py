"""Configuration management for the OCR Japanese Translator project."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure directories exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")

# Translation settings
TARGET_LANGUAGE = "English"
SOURCE_LANGUAGE = "Japanese"

# Font settings for rendering text
DEFAULT_FONT_SIZE = 20
FONT_COLOR = (0, 0, 0)  # Black text

# Image processing settings
MIN_CONFIDENCE = 0.5  # Minimum confidence for text detection
PADDING_RATIO = 0.1  # Padding around text bounding boxes