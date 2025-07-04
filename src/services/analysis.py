# src/analysis.py

import os
import ast
import yaml
import argparse
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer

from src.dataset import load_raw_reviews, save_enriched
from src.features import clean_text


def load_config():
    with open("config/labels.yaml") as f:
        labels = yaml.safe_load(f)
    with open("config/keywords.yaml") as f:
        keywords = yaml.safe_load(f)
    return labels, keywords


def embed_labels(labels, model):
    """Compute embeddings for each label string."""
    return model.encode(labels, convert_to_tensor=True, show_progress_bar=False)


def compute_semantic_matches(doc_embeds, label_embeds, labels, top_k=3):
    """
    For each document embedding, compute cosine similarities to all label embeddings.
    Return lists of top-1 and top-k labels.
    """
    sims = cosine_similarity(doc_embeds, label_embeds)  # shape (n_docs, n_labels)
    top_indices = sims.argsort(axis=1)[:, ::-1]  # descending
    top1 = [labels[idxs[0]] for idxs in top_indices]
    topk = [[labels[i] for i in idxs[:top_k]] for idxs in top_indices]
    return top1, topk


def match_keywords(text, keywords: dict[str, list[str]]):
    """
    For a single document text, find which keywords appear per label.
    Returns a dict {label: [matched_keywords]}, and a total count.
    """
    text_low = text.lower()
    matched = {}
    total = 0
    for label, kw_list in keywords.items():
        hits = [kw for kw in kw_list if kw in text_low]
        if hits:
            matched[label] = hits
            total += len(hits)
    return matched, total


def evaluate(true_lists, pred_top1, labels):
    """
    Binarize true vs. predicted and compute precision, recall, f1.
    """
    mlb = MultiLabelBinarizer(classes=labels)
    y_true = mlb.fit_transform(true_lists)
    y_pred = mlb.transform([[p] for p in pred_top1])
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return p, r, f1


def enrich_dataframe(
    df: pd.DataFrame,
    model_name: str,
    labels: list[str],
    keywords: dict[str, list[str]],
    top_k: int,
):
    # 1) Clean text
    df["cleaned_text"] = df["review_text"].astype(str).map(clean_text)

    # 2) Embed all reviews
    model = SentenceTransformer(model_name)
    doc_embeds = model.encode(
        df["cleaned_text"].tolist(), convert_to_tensor=True, show_progress_bar=True
    )

    # 3) Embed labels once
    label_embeds = embed_labels(labels, model)

    # 4) Semantic matching
    top1, topk = compute_semantic_matches(doc_embeds, label_embeds, labels, top_k)
    df["top_accusation"] = top1
    df["top3_accusations"] = topk

    # 5) Keyword matching
    kw_matches = (
        df["review_text"].astype(str).map(lambda t: match_keywords(t, keywords))
    )
    df["matched_keywords"] = kw_matches.map(lambda mk: mk[0])
    df["keyword_count"] = kw_matches.map(lambda mk: mk[1])

    # 6) Optional evaluation
    if "true_accusations" in df.columns:
        # assume true_accusations is a list per row
        p, r, f1 = evaluate(df["true_accusations"].tolist(), top1, labels)
        print(
            f"Top-1 classification — Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}"
        )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run semantic+keyword analysis on reviews."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to raw Excel file with columns ['review_text','accusations']",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Path to write enriched Excel"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--topk",
        "-k",
        type=int,
        default=3,
        help="How many semantic labels to keep (top-k)",
    )
    args = parser.parse_args()

    # 1) Load configs
    labels, keywords = load_config()

    # 2) Load data
    df = load_raw_reviews(args.input)

    # 3) Enrich
    enriched = enrich_dataframe(df, args.model, labels, keywords, args.topk)

    # 4) Save results
    save_enriched(enriched, args.output)
    print(f"Enriched data saved to {args.output}")


if __name__ == "__main__":
    main()
