# src/modeling/predict.py

import argparse
import joblib
from sentence_transformers import SentenceTransformer

from src.features import clean_text


def predict(args):
    # 1) Load bundle
    bundle = joblib.load(args.model)
    model_name = bundle["embedder_name"]
    mlb = bundle["mlb"]
    clf = bundle["clf"]

    # 2) Prepare input texts
    texts = [args.text] if args.text else open(args.input).read().splitlines()
    cleaned = [clean_text(t) for t in texts]

    # 3) Embed
    embedder = SentenceTransformer(model_name)
    X = embedder.encode(cleaned, convert_to_tensor=False, show_progress_bar=False)

    # 4) Predict probabilities
    probs = clf.predict_proba(X)

    # 5) Threshold or top-k
    for text, row in zip(texts, probs):
        # Pick all classes above thresh
        above = mlb.classes_[row >= args.thresh]
        # If none, take top-k
        if len(above) == 0:
            topk_idx = row.argsort()[::-1][: args.topk]
            above = mlb.classes_[topk_idx]
        print(f"\nText: {text}")
        for cls, p in zip(mlb.classes_, row):
            print(f"  {cls:20s}: {p:.3f}")
        print("→ Predicted:", list(above))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load model bundle and predict labels")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Single review text to classify")
    group.add_argument("--input", help="File with one review per line")
    parser.add_argument(
        "-m",
        "--model",
        default="models/multilabel_model.joblib",
        help="Path to saved joblib bundle",
    )
    parser.add_argument(
        "-t",
        "--thresh",
        type=float,
        default=0.5,
        help="Probability threshold for selecting labels",
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        default=3,
        help="If no labels pass threshold, pick this many highest-scoring",
    )
    args = parser.parse_args()
    predict(args)
