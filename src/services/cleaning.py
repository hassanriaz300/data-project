import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords

# -----------------------------------------------------------------------------
# 1) Stopwords setup
# -----------------------------------------------------------------------------
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

# -----------------------------------------------------------------------------
# 2) Emoji & Punctuation regexes
# -----------------------------------------------------------------------------
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols/pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\u2700-\u27bf"  # dingbats
    "\u24c2-\U0001f251"  # enclosed chars
    "]+",
    flags=re.UNICODE,
)
PUNCT_PATTERN = re.compile(r"[^\w\s]")  # remove any non-alphanumeric, non-space


# -----------------------------------------------------------------------------
# 3) Cleaning helpers
# -----------------------------------------------------------------------------
def count_emojis(text: str) -> int:
    return len(EMOJI_PATTERN.findall(str(text)))


def has_emoji(text: str) -> bool:
    return count_emojis(text) > 0


def normalize_text(text: str) -> str:
    txt = str(text).lower()
    txt = EMOJI_PATTERN.sub("", txt)  # strip emojis
    txt = PUNCT_PATTERN.sub(" ", txt)  # strip punctuation/symbols
    txt = re.sub(r"\s+", " ", txt).strip()  # collapse spaces
    return txt


def has_stopword(text: str) -> bool:
    tokens = normalize_text(text).split()
    return any(tok in STOP_WORDS for tok in tokens)


def remove_stopwords(text: str) -> str:
    tokens = normalize_text(text).split()
    return " ".join(tok for tok in tokens if tok not in STOP_WORDS)


# -----------------------------------------------------------------------------
# 4) Main processing function
# -----------------------------------------------------------------------------
def prepare_reviews(input_path: str, out_dir: str):
    """
    1) Drop columns
    2) Flag & split by emoji
    3) Normalize, remove stop-words & flag
    4) Write two enriched files:
       - edeka_with_emojis_enriched.xlsx
       - edeka_without_emojis_enriched.xlsx
    5) Aggregate long→wide on cleaned_review, preserving lists of:
       - sentiment
       - theme_tra, cateex_tra, cat_tra
    6) Write wide file: edeka_reviews_wide.xlsx
    """
    df = pd.read_excel(input_path)

    # Drop unwanted columns
    for c in [
        "review_text",
        "review_text_german",
        "Translated_review1",
        "theme",
        "category_explanation",
        "category",
    ]:
        if c in df.columns:
            df = df.drop(columns=c)

    # Emoji flags
    df["emoji_count"] = df["Translated_review2"].apply(count_emojis)
    df["Has_emoji"] = df["emoji_count"] > 0

    # Stop-word flags & cleaned text
    df["Has_Stop_word"] = df["Translated_review2"].apply(has_stopword)
    df["cleaned_review"] = df["Translated_review2"].apply(remove_stopwords)

    # Split enriched
    df_with = df[df["Has_emoji"]].reset_index(drop=True)
    df_without = df[~df["Has_emoji"]].reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    path_w = os.path.join(out_dir, "edeka_with_emojis_enriched.xlsx")
    path_wo = os.path.join(out_dir, "edeka_without_emojis_enriched.xlsx")
    df_with.to_excel(path_w, index=False)
    df_without.to_excel(path_wo, index=False)

    print(f"Wrote enriched (with emojis)    → {path_w}")
    print(f"Wrote enriched (without emojis) → {path_wo}")

    # Now collapse to wide format
    dedup_cols = [
        "date",
        "Translated_review2",
        "rating",
        "store",
        "address",
        "city",
        "latitude",
        "longitude",
        "emoji_count",
        "Has_emoji",
        "Has_Stop_word",
        "cleaned_review",
    ]

    # For multilabel columns, aggregate all unique values into lists
    agg_dict = {col: "first" for col in dedup_cols}
    # For other key fields you want to preserve as lists:
    for col in ["sentiment", "theme_tra", "cateex_tra", "cat_tra"]:
        if col in df.columns:
            agg_dict[col] = lambda s: list(dict.fromkeys(s.dropna()))
    # For any extra columns, just keep first occurrence
    for col in df.columns:
        if col not in agg_dict:
            agg_dict[col] = "first"

    # Group by all dedup columns: only perfect duplicates are merged
    wide = df.groupby(dedup_cols, as_index=False).agg(agg_dict)
    wide_path = os.path.join(out_dir, "edeka_reviews_wide.xlsx")
    wide.to_excel(wide_path, index=False)
    print(f"Wrote wide-format file           → {wide_path}")


# -----------------------------------------------------------------------------
# 5) CLI entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Clean, split & reshape Edeka reviews")
    p.add_argument(
        "--input",
        default="data/interim/edeka_with_emojis.xlsx",
        help="Input Excel file",
    )
    p.add_argument(
        "--out-dir",
        default="data/interim/cleaned_reviews",
        help="Directory for output files",
    )
    args = p.parse_args()
    prepare_reviews(args.input, args.out_dir)
