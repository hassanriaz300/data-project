# src/modeling/train.py

import argparse
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

from src.dataset import load_raw_reviews
from src.features import clean_text


def prepare_features(texts, model_name):
    """Clean + embed a list of raw texts."""
    model = SentenceTransformer(model_name)
    cleaned = [clean_text(t) for t in texts]
    embeddings = model.encode(cleaned, convert_to_tensor=False, show_progress_bar=True)
    return embeddings


def train(args):
    # 1) Load data
    df = load_raw_reviews(args.input)
    X_text = df[args.text_col].astype(str).tolist()
    y_lists = df["true_accusations"].tolist()

    # 2) Featurize
    print("Embedding texts with", args.model)
    X = prepare_features(X_text, args.model)

    # 3) Binarize labels
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_lists)
    print(f"Classes: {mlb.classes_.tolist()}")

    # 4) Fit a One-vs-Rest logistic regression
    clf = OneVsRestClassifier(LogisticRegression(solver="liblinear"))
    clf.fit(X, Y)
    print("Training complete.")

    # 5) Dump bundle
    bundle = {"embedder_name": args.model, "mlb": mlb, "clf": clf}
    joblib.dump(bundle, args.output)
    print(f"Model bundle saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multilabel classifier on reviews"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to Excel file with ['review_text','accusations']",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="models/multilabel_model.joblib",
        help="Where to dump the joblib bundle",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model to use for embeddings",
    )
    parser.add_argument(
        "--text-col",
        default="review_text",
        help="Name of the text column in your input file",
    )
    args = parser.parse_args()
    train(args)
