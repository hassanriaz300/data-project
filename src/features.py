# Cleaning and Tokenization Functions in progress

import re
import string

# Emoji ranges (common blocks)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "\U0001f780-\U0001f7ff"  # Geometric shapes extended
    "\U0001f800-\U0001f8ff"  # Supplemental arrows-C
    "\U0001f900-\U0001f9ff"  # Supplemental symbols & pictographs
    "\U0001fa00-\U0001fa6f"  # Chess symbols, etc.
    "\U0001fa70-\U0001faff"  # Symbols & pictographs extended-A
    "]+",
    flags=re.UNICODE,
)


def remove_emojis(text: str) -> str:
    """Strip emojis from a string."""
    return EMOJI_PATTERN.sub("", text)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. Strip emojis
    text = remove_emojis(text)
    # 2. Keep only ASCII (drops accented chars, etc.)
    text = text.encode("ascii", errors="ignore").decode()
    # 3. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).lower().strip()
    return text


def tokenize(text: str) -> list[str]:
    """
    Split cleaned text into tokens on whitespace.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()
