# This file contains functions for mapping accusations to semantic labels using a pre-trained model.
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from src.services.paths import SEMANTIC_DIR


# Input and output paths for the semantic accusation mapping service.
DEFAULT_INPUT = Path("data/interim/with_top5_accusations.xlsx")
DEFAULT_OUTPUT = Path(SEMANTIC_DIR) / "top5_semantic_mappedweb.xlsx"

MODEL_NAME = "all-mpnet-base-v2"  # stronger than MiniLM
TOP_N = 5  # how many labels to retain
THRESH = 0.45  # min cosine similarity to accept


# All accusations to be used in the mapping.
ALL_ACCS: List[str] = [
    "Rude or disrespectful staff",
    "Moldy or expired food sold",
    "Hidden price increases / price manipulation",
    "Dirty store or poor hygiene",
    "Discrimination or unfair treatment",
    "Misleading product information or labeling",
    "Long waiting times or not enough checkouts",
    "Goods out of stock or unavailable",
    "Excessive packaging or environmental waste",
    "Food waste or edible food thrown away",
    "Labor exploitation in supply chains",
    "Severe migrant worker exploitation",
    "Seasonal farm labor abuse",
    "Retailer price-fixing scandal",
    "Vertical price-fixing",
    "Human rights due diligence failures",
    "Underpaid farm labor in supply chains",
    "Plantation worker rights violations",
    "Antitrust concerns in grocery delivery",
    "Labor & living condition abuses",
    "Defective cooling, mice infestation and mold—severe hygiene breaches",
    "Massive cleanliness failures exposed in undercover reports",
    "Ignoring human-rights and labor violations in supplier plantations",
    "Alleged breaches of Germany’s supply-chain law: underpaid labor and unsafe conditions",
    "Environmental destruction and rights abuses on palm-oil plantations",
    "Misleading use of RSPO eco-label despite rights and environmental breaches",
    "Product-safety hazard: metal fragments found in packaged food",
    "Illegal internal surveillance and “spy” operations against employees",
    "Poor transparency, labor and women’s rights violations, neglect of small-scale producers",
    "Intransparent discount schemes—special app-only prices not clearly disclosed",
    "Excessive single-use packaging and lack of reuse options",
    "Misleading bonus-app presentation: discounts shown without final prices",
    "Raised minimum-spend thresholds for coupon redemption, disadvantaging shoppers",
    "Charging customers for the weight of packaging (bag weight passed onto buyer)",
    "Bonus-app only displays discount amounts, not original prices—considered deceptive",
    "Privacy violations: unauthorized collection of employee medical and personal data",
    "Secret price-fixing agreements leading to hefty fines",
    "Abuse of market power: anticompetitive retaliation via antitrust complaints",
    "Overuse of single-use plastics despite sustainability pledges",
    "Investigations into possible price-collusion among discounters",
    "High rates of plastic packaging in produce aisles",
    "Collusion among potato processors to fix prices, hurting consumers",
]


THEME_TO_ACCS: Dict[str, List[str]] = {acc: [acc] for acc in ALL_ACCS}
CATEEX_TO_ACCS: Dict[str, List[str]] = {
    "Hygiene and cleanliness (e.g. cleanliness)": [
        "Dirty store or poor hygiene",
        "Moldy or expired food sold",
        "Defective cooling, mice infestation and mold—severe hygiene breaches",
    ],
    "Long queues": ["Long waiting times or not enough checkouts"],
    "Pricing policy and bargains (e.g. price, special offers, discount campaign)": [
        "Hidden price increases / price manipulation"
    ],
    "Pricing policy and bargains (e.g. price, special offers, discount campaigns)": [
        "Hidden price increases / price manipulation"
    ],
    "Product availability and quality (e.g. fish counter, counter, bread, cake, yoghurt, meat and sausage counter, missing stocks)": [
        "Goods out of stock or unavailable",
        "Moldy or expired food sold",
    ],
    "Customer service and employee friendliness (e.g. staff, unprofessionalism, incompetence)": [
        "Rude or disrespectful staff",
        "Discrimination or unfair treatment",
    ],
    "Shopping experience and atmosphere": [
        "Rude or disrespectful staff",
        "Discrimination or unfair treatment",
    ],
    "Social responsibility and ethics (e.g. racism, political messages)": [
        "Discrimination or unfair treatment",
        "Excessive packaging or environmental waste",
    ],
    "Store layout and infrastructure (e.g. product presentation, store layout, interior design, renovation, renovation, size of the store)": [
        "Dirty store or poor hygiene",
        "Long waiting times or not enough checkouts",
    ],
    "Technological aspects and functionality (e.g. APP, Internet, self-service checkouts)": [
        "Hidden price increases / price manipulation",
        "Long waiting times or not enough checkouts",
    ],
    "Access, location and accessibility (e.g. access, surroundings, opening times)": [
        "Long waiting times or not enough checkouts",
    ],
    "#VALUE!": ALL_ACCS,  # corrupted cell fallback
}
CAT_TO_ACCS: Dict[str, List[str]] = {
    "Hygiene and cleanliness": [
        "Dirty store or poor hygiene",
        "Moldy or expired food sold",
        "Defective cooling, mice infestation and mold—severe hygiene breaches",
    ],
    "Pricing policy and bargains": ["Hidden price increases / price manipulation"],
    "Customer service and employee friendliness": [
        "Rude or disrespectful staff",
        "Discrimination or unfair treatment",
    ],
    "Product availability and quality": [
        "Goods out of stock or unavailable",
        "Moldy or expired food sold",
    ],
    "Access, location and accessibility": [
        "Long waiting times or not enough checkouts",
    ],
    "Store layout and infrastructure": [
        "Dirty store or poor hygiene",
        "Long waiting times or not enough checkouts",
    ],
    "Shopping experience and atmosphere": [
        "Rude or disrespectful staff",
        "Discrimination or unfair treatment",
    ],
    "Social responsibility and ethics": [
        "Discrimination or unfair treatment",
        "Excessive packaging or environmental waste",
    ],
    "Technological aspects and functionality": [
        "Hidden price increases / price manipulation",
        "Long waiting times or not enough checkouts",
    ],
}


----------------------------------------------------------------------
def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

# Narrows down the universe of accusations based on theme, catex, and cat.
def _get_candidate_accs(theme, catex, cat) -> List[str]:
    
    cands: List[str] = []
    for t in _as_list(theme):
        cands += THEME_TO_ACCS.get(t, [])
    if not cands:
        for c in _as_list(catex):
            cands += CATEEX_TO_ACCS.get(c, [])
    if not cands:
        for g in _as_list(cat):
            cands += CAT_TO_ACCS.get(g, [])
    if not cands:
        cands = ALL_ACCS.copy()

    return list(dict.fromkeys(cands))



def map_accusations(
    input_path: os.PathLike | str = DEFAULT_INPUT,
    output_path: os.PathLike | str = DEFAULT_OUTPUT,
    model_name: str = MODEL_NAME,
    top_n: int = TOP_N,
    thresh: float = THRESH,
):
   
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    
    print(f"loading {input_path}")
    df = pd.read_excel(input_path)
    for col in ["sentiment", "theme_tra", "cateex_tra", "cat_tra"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

    
    print(f"loading model: {model_name} …")
    model = SentenceTransformer(model_name)
    acc_embs = model.encode(ALL_ACCS, convert_to_tensor=True, normalize_embeddings=True)

    
    def semantic_top_n(row) -> Tuple[List[str], List[float]]:
        # skip positive reviews
        sentiments = row.get("sentiment", [])
        is_relevant = (
            any(s in ("negative", "mixed") for s in sentiments)
            if isinstance(sentiments, list)
            else sentiments in ("negative", "mixed")
        )
        if not is_relevant:
            return [None] * top_n, [None] * top_n

        cands = _get_candidate_accs(
            row.get("theme_tra", []), row.get("cateex_tra", []), row.get("cat_tra", [])
        )
        idxs = [ALL_ACCS.index(lbl) for lbl in cands]

        emb = model.encode(
            str(row.get("cleaned_review", "")),
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        sims = util.cos_sim(emb, acc_embs[idxs])[0]
        scores, topk = sims.topk(k=min(top_n, len(cands)))

        labels, vals = [], []
        for i, s in zip(topk.cpu().tolist(), scores.cpu()):
            if float(s) >= thresh:
                labels.append(cands[i])
                vals.append(float(s))
        
        while len(labels) < top_n:
            labels.append(None)
            vals.append(None)
        return labels, vals

    
    print(" scoring reviews …")
    tqdm.pandas()
    tops = df.progress_apply(semantic_top_n, axis=1)

    label_cols = [f"SemTop{i}" for i in range(1, top_n + 1)]
    score_cols = [f"SemTop{i}_score" for i in range(1, top_n + 1)]
    df[label_cols] = tops.apply(lambda t: pd.Series(t[0]))
    df[score_cols] = tops.apply(lambda t: pd.Series(t[1]))

   
    df.to_excel(output_path, index=False)
    print(f" saving → {output_path}")
    print(f" done!  Added {2 * top_n} semantic columns.")



if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Semantic accusation mapper")
    p.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Input Excel file (cleaned & with candidate cols)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output Excel file with SemTop* columns",
    )
    p.add_argument("--model", default=MODEL_NAME, help="Sentence-Transformer name")
    p.add_argument("--top_n", type=int, default=TOP_N, help="Top-N labels to keep")
    p.add_argument("--thresh", type=float, default=THRESH, help="Min cosine similarity")
    args = p.parse_args()

    map_accusations(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        top_n=args.top_n,
        thresh=args.thresh,
    )
