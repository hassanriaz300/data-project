import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords


try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))


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


# This function counts the number of emojis in the text and returns the count might be usefull for fake reviews detection.
def count_emojis(text: str) -> int:
    return len(EMOJI_PATTERN.findall(str(text)))


# This function checks if the text contains any emojis and returns a boolean value.
def has_emoji(text: str) -> bool:
    return count_emojis(text) > 0


# This function normalizes the text by converting it to lowercase, removing emojis,
# punctuation, and collapsing multiple spaces into a single space.
def normalize_text(text: str) -> str:
    txt = str(text).lower()
    txt = EMOJI_PATTERN.sub("", txt)  # strip emojis
    txt = PUNCT_PATTERN.sub(" ", txt)  # strip punctuation/symbols
    txt = re.sub(r"\s+", " ", txt).strip()  # collapse spaces
    return txt


# This function checks if the text contains any stopwords and returns a boolean value.
def has_stopword(text: str) -> bool:
    tokens = normalize_text(text).split()
    return any(tok in STOP_WORDS for tok in tokens)


# This function removes stopwords from the text, returning a cleaned version without them.
def remove_stopwords(text: str) -> str:
    tokens = normalize_text(text).split()
    return " ".join(tok for tok in tokens if tok not in STOP_WORDS)


# This function prepares the reviews by cleaning, splitting, and reshaping them into a wide format for further analysis.Files are saved in the specified output directory.
def prepare_reviews(input_path: str, out_dir: str):
    df = pd.read_excel(input_path)

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

    df["emoji_count"] = df["Translated_review2"].apply(count_emojis)
    df["Has_emoji"] = df["emoji_count"] > 0

    df["Has_Stop_word"] = df["Translated_review2"].apply(has_stopword)
    df["cleaned_review"] = df["Translated_review2"].apply(remove_stopwords)

    df_with = df[df["Has_emoji"]].reset_index(drop=True)
    df_without = df[~df["Has_emoji"]].reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    path_w = os.path.join(out_dir, "edeka_with_emojis_enriched.xlsx")
    path_wo = os.path.join(out_dir, "edeka_without_emojis_enriched.xlsx")
    df_with.to_excel(path_w, index=False)
    df_without.to_excel(path_wo, index=False)

    print(f"Wrote enriched (with emojis)    → {path_w}")
    print(f"Wrote enriched (without emojis) → {path_wo}")

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

    agg_dict = {col: "first" for col in dedup_cols}
    for col in ["sentiment", "theme_tra", "cateex_tra", "cat_tra"]:
        if col in df.columns:
            agg_dict[col] = lambda s: list(dict.fromkeys(s.dropna()))

    for col in df.columns:
        if col not in agg_dict:
            agg_dict[col] = "first"

    wide = df.groupby(dedup_cols, as_index=False).agg(agg_dict)
    wide_path = os.path.join(out_dir, "edeka_reviews_wide.xlsx")
    wide.to_excel(wide_path, index=False)
    print(f"Wrote wide-format file           → {wide_path}")


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
