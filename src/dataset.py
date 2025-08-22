import ast
import pandas as pd


def parse_accusations(cell):
    """
    Safely parse the 'accusations' column which may contain
    string-encoded Python lists. Returns an empty list on failure.
    """
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return cell
    try:
        return ast.literal_eval(cell)
    except Exception:
        return []


def load_raw_reviews(
    path: str = "data/interim/edeka_with_hf_accusations.xlsx",
    text_col: str = "review_text",
    acc_col: str = "accusations",
) -> pd.DataFrame:
    df = pd.read_excel(path)
    # ensure text column exists
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in {path}")

    # parse accusations
    df["true_accusations"] = df[acc_col].apply(parse_accusations)
    # rename text column for consistency
    df = df.rename(columns={text_col: "review_text"})
    return df


def save_enriched(
    df: pd.DataFrame, path: str = "data/processed/edeka_analysis_output.xlsx"
) -> None:
    """
    Save the enriched DataFrame (with semantic matches,
    keyword counts, etc.) to an Excel file.
    """
    # create directory if needed
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False)


if __name__ == "__main__":
    df = load_raw_reviews()
    print(f"Loaded {len(df)} reviews. Sample:")
    print(df.head())
    save_enriched(df)
    print("Sample data saved to default path.")
