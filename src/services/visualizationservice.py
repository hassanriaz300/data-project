# This file contains functions for visualizing and analyzing accusations data from Excel files.

import ast
import os
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from src.services.paths import VISUALIZATION_DIR as OUT_DIR
from collections import defaultdict, Counter
from wordcloud import WordCloud


# Output directory for saved plots
OUT_DIR = "data/interim/visualization"
os.makedirs(OUT_DIR, exist_ok=True)
WORDCLOUD_DIR = "data/interim/wordclouds"
os.makedirs(WORDCLOUD_DIR, exist_ok=True)

# Mapping similar accusations into unified category labels
CATEGORY_MAPPING = {
    # 1. Employee Conduct & Customer Service
    "Rude or disrespectful staff": "Employee Conduct & Customer Service",
    "Discrimination or unfair treatment": "Employee Conduct & Customer Service",
    # 2. Store Hygiene & Cleanliness
    "Dirty store or poor hygiene": "Store Hygiene & Cleanliness",
    "Defective cooling, mice infestation and mold—severe hygiene breaches": "Store Hygiene & Cleanliness",
    "Massive cleanliness failures exposed in undercover reports": "Store Hygiene & Cleanliness",
    # 3. Product Safety & Quality
    "Moldy or expired food sold": "Product Safety & Quality",
    "Product-safety hazard: metal fragments found in packaged food": "Product Safety & Quality",
    # 4. Pricing & Discount Practices
    "Hidden price increases / price manipulation": "Pricing & Discount Practices",
    "Bonus-app only displays discount amounts, not original prices — considered deceptive": "Pricing & Discount Practices",
    "Misleading bonus-app presentation: discounts shown without final prices": "Pricing & Discount Practices",
    "Intransparent discount schemes — special app-only prices not clearly disclosed": "Pricing & Discount Practices",
    "Charging customers for the weight of packaging (bag weight passed onto buyer)": "Pricing & Discount Practices",
    "Raised minimum-spend thresholds for coupon redemption, disadvantaging shoppers": "Pricing & Discount Practices",
    # 5. Competition & Antitrust Issues
    "Retailer price-fixing scandal": "Competition & Antitrust Issues",
    "Vertical price-fixing": "Competition & Antitrust Issues",
    "Secret price-fixing agreements leading to hefty fines": "Competition & Antitrust Issues",
    "Collusion among potato processors to fix prices, hurting consumers": "Competition & Antitrust Issues",
    "Investigations into possible price-collusion among discounters": "Competition & Antitrust Issues",
    "Antitrust concerns in grocery delivery": "Competition & Antitrust Issues",
    "Abuse of market power: anticompetitive retaliation via antitrust complaints": "Competition & Antitrust Issues",
    # 6. Stock & Service Availability
    "Goods out of stock or unavailable": "Stock & Service Availability",
    "Long waiting times or not enough checkouts": "Stock & Service Availability",
    # 7. Packaging & Environmental Impact
    "Excessive packaging or environmental waste": "Packaging & Environmental Impact",
    "Excessive single-use packaging and lack of reuse options": "Packaging & Environmental Impact",
    "Overuse of single-use plastics despite sustainability pledges": "Packaging & Environmental Impact",
    "High rates of plastic packaging in produce aisles": "Packaging & Environmental Impact",
    # 8. Food Waste Practices
    "Food waste or edible food thrown away": "Food Waste Practices",
    # 9. Labor & Human Rights
    "Labor exploitation in supply chains": "Labor & Human Rights",
    "Underpaid farm labor in supply chains": "Labor & Human Rights",
    "Seasonal farm labor abuse": "Labor & Human Rights",
    "Severe migrant worker exploitation": "Labor & Human Rights",
    "Labor & living condition abuses": "Labor & Human Rights",
    "Plantation worker rights violations": "Labor & Human Rights",
    # 10. Supply-Chain Transparency & Ethical Sourcing
    "Human rights due diligence failures": "Supply-Chain Transparency & Ethical Sourcing",
    "Alleged breaches of Germany’s supply-chain law: underpaid labor and unsafe conditions": "Supply-Chain Transparency & Ethical Sourcing",
    "Ignoring human-rights and labor violations in supplier plantations": "Supply-Chain Transparency & Ethical Sourcing",
    "Environmental destruction and rights abuses on palm-oil plantations": "Supply-Chain Transparency & Ethical Sourcing",
    "Poor transparency, labor and women’s rights violations, neglect of small-scale producers": "Supply-Chain Transparency & Ethical Sourcing",
    # 11. Privacy & Data Surveillance
    'Illegal internal surveillance and "spy" operations against employees': "Privacy & Data Surveillance",
    "Privacy violations: unauthorized collection of employee medical and personal data": "Privacy & Data Surveillance",
    # 12. Product Labeling & Information
    "Misleading product information or labeling": "Product Labeling & Information",
    "Misleading use of RSPO eco-label despite rights and environmental breaches": "Product Labeling & Information",
}


# This function maps a single accusation value to its unified category label.
def _map_value(val):
    return CATEGORY_MAPPING.get(val, val)


# This function normalizes a list of accusations by mapping each to its unified category label.
def _normalize_list(acc_list):
    if not isinstance(acc_list, list):
        return acc_list
    return [_map_value(item) for item in acc_list]


def plot_top10_accusation_heatmap(input_path: str) -> dict:
    df = pd.read_excel(input_path)

    # Parse and normalize list-like columns
    for col in ["most_accusations", "medium_accusations", "less_accusations"]:
        df[col] = df[col].apply(
            lambda x: _normalize_list(ast.literal_eval(x)) if isinstance(x, str) else []
        )

    # Compute % of reviews per tier
    def pct_series(col):
        return df[[col]].explode(col).dropna()[col].value_counts(normalize=True) * 100

    most = pct_series("most_accusations")
    medium = pct_series("medium_accusations")
    least = pct_series("less_accusations")

    combined = pd.concat([most, medium, least], axis=1).fillna(0)
    combined.columns = ["Most", "Medium", "Least"]

    top10 = combined.sum(axis=1).nlargest(10).index
    m = combined.loc[top10]

    wrapped_labels = ["\n".join(textwrap.wrap(lbl, width=30)) for lbl in m.index]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(
        m.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=m.values.max()
    )

    ax.set_yticks(np.arange(len(wrapped_labels)))
    ax.set_yticklabels(wrapped_labels, fontsize=9)
    ax.set_xticks(np.arange(len(m.columns)))
    ax.set_xticklabels(m.columns, fontsize=10)

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(
                j,
                i,
                f"{m.iat[i, j]:.1f}%",
                ha="center",
                va="center",
                color="white" if m.iat[i, j] > (m.values.max() / 2) else "black",
                fontsize=8,
            )

    plt.colorbar(im, label="Percent of all reviews", fraction=0.046, pad=0.04)
    plt.title("Top 10 Accusations by Evidence Level (%)", pad=12)
    plt.tight_layout()

    # 5. Save plot
    plot_filename = f"accusation_heatmap_{uuid.uuid4().hex[:6]}.png"
    plot_path = os.path.join(OUT_DIR, plot_filename)
    fig.savefig(plot_path)
    plt.close(fig)

    return {
        "plot": plot_path,
        "top10_labels": top10.tolist(),
        "data_table": m.reset_index().to_dict(orient="records"),
    }


# This function plots a heatmap of accusation categories by their rank position (Top1–Top5).ALL Models
def plot_top10_by_rank_heatmap(input_path: str) -> dict:
    """
    Loads Excel with Top1–Top5 accusations, maps to categories,
    and plots a heatmap showing their frequency by rank position.
    """
    df = pd.read_excel(input_path)

    records = []
    for rank in range(1, 6):
        col = f"Top{rank}"
        if col in df.columns:
            temp = df[[col]].rename(columns={col: "accusation"}).copy()
            temp["rank"] = rank
            temp["accusation"] = temp["accusation"].apply(_map_value)
            records.append(temp)

    long = pd.concat(records, ignore_index=True).dropna(subset=["accusation"])

    # Compute percentage per (category, rank)
    counts = long.groupby(["accusation", "rank"]).size().unstack(fill_value=0)
    pct = counts.div(len(df)) * 100

    # Focus on top 10 by total share
    top10 = pct.sum(axis=1).nlargest(10).index
    heat = pct.loc[top10]
    wrapped_labels = ["\n".join(textwrap.wrap(lbl, width=25)) for lbl in heat.index]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        heat.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=heat.values.max()
    )

    ax.set_yticks(np.arange(len(wrapped_labels)))
    ax.set_yticklabels(wrapped_labels, fontsize=10)
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels([f"TOP {i}" for i in heat.columns], fontsize=11)

    max_val = heat.values.max()
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iat[i, j]
            ax.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                color="white" if val > max_val * 0.6 else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "Percent of all reviews", size=12
    )
    ax.set_title("Top 10 Accusation Categories by Rank (%)", fontsize=14, pad=20)
    plt.tight_layout()

    # Save plot
    plot_filename = f"rank_heatmap_{uuid.uuid4().hex[:6]}.png"
    plot_path = os.path.join(OUT_DIR, plot_filename)
    fig.savefig(plot_path)
    plt.close(fig)

    return {
        "plot": plot_path,
        "top10_labels": top10.tolist(),
        "data_table": heat.reset_index().to_dict(orient="records"),
    }


# This function plots a heatmap of semantic topics (SemTop1–SemTop5) by their frequency across reviews.
def plot_semantic_topic_heatmap(input_path: str) -> dict:
    """
    Loads Excel with SemTop1–SemTop5 topics, maps each via _map_value,
    and plots a heatmap showing their frequency by SEM TOP position.
    """
    import uuid
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import textwrap

    from src.services.visualizationservice import _map_value
    from src.services.paths import VISUALIZATION_DIR as OUT_DIR

    df = pd.read_excel(input_path)

    records = []
    for pos in range(1, 6):
        col = f"SemTop{pos}"
        if col in df.columns:
            temp = df[[col]].rename(columns={col: "topic"}).copy()
            temp["position"] = pos
            # map each topic string intocategory map if it matches exactly
            temp["topic"] = temp["topic"].apply(_map_value)
            records.append(temp)

    long = pd.concat(records, ignore_index=True).dropna(subset=["topic"])

    # Compute percentage per (topic, position)
    counts = long.groupby(["topic", "position"]).size().unstack(fill_value=0)
    pct = counts.div(len(df)) * 100

    # Focus on top 10 by total share
    top10 = pct.sum(axis=1).nlargest(10).index
    heat = pct.loc[top10]

    wrapped_labels = ["\n".join(textwrap.wrap(lbl, width=25)) for lbl in heat.index]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        heat.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=heat.values.max()
    )

    # Y-axis: topics
    ax.set_yticks(np.arange(len(wrapped_labels)))
    ax.set_yticklabels(wrapped_labels, fontsize=10)
    # X-axis: SEM TOP positions
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels([f"SEM TOP {i}" for i in heat.columns], fontsize=11)

    # Annotate cells
    max_val = heat.values.max()
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iat[i, j]
            ax.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                color="white" if val > max_val * 0.6 else "black",
                fontsize=9,
            )

    # Colorbar & styling
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percent of all reviews", size=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title("Top 10 Semantic Topics by SEM TOP Position (%)", fontsize=14, pad=20)
    plt.tight_layout()

    # Save plot
    plot_filename = f"semantic_heatmap_{uuid.uuid4().hex[:6]}.png"
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUT_DIR, plot_filename)
    fig.savefig(plot_path)
    plt.close(fig)

    return {
        "plot": plot_path,
        "top10_labels": top10.tolist(),
        "data_table": heat.reset_index().to_dict(orient="records"),
    }


# This function plots the top-5 accusations by their presence across reviews.
def plot_top5_accusations(input_path: str) -> dict:
    df = pd.read_excel(input_path)

    def gather_pairs(row):
        pairs = []
        # list‐columns
        for acc_col, score_col in [
            ("most_accusations", "most_scores"),
            ("medium_accusations", "medium_scores"),
            ("less_accusations", "less_scores"),
        ]:
            accs = row.get(acc_col) or []
            scores = row.get(score_col) or []
            if isinstance(accs, str):
                accs, scores = ast.literal_eval(accs), ast.literal_eval(scores)
            for a, s in zip(accs, scores):
                pairs.append((_map_value(a), s))

        # Top1–Top5 columns
        for i in range(1, 6):
            a = row.get(f"Top{i}")
            s = row.get(f"Top{i}_score")
            if pd.notna(a) and pd.notna(s):
                pairs.append((_map_value(a), s))

        # SemTop1–SemTop5 columns
        for i in range(1, 6):
            a = row.get(f"SemTop{i}")
            s = row.get(f"SemTop{i}_score")
            if pd.notna(a) and pd.notna(s):
                pairs.append((_map_value(a), s))

        return pairs

    # 3. For each review, take its top-5 by score
    top5_per_review = []
    for _, row in df.iterrows():
        pairs = gather_pairs(row)
        top5 = [a for a, _ in sorted(pairs, key=lambda x: x[1], reverse=True)[:5]]
        top5_per_review.append(top5)

    # 4. Compute frequencies across reviews
    all_top5 = pd.Series([a for sub in top5_per_review for a in sub])
    counts = (all_top5.value_counts(normalize=True) * 100).head(10).sort_values()

    # 5. Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.barh(ax=ax)
    ax.set_xlabel("Percentage of Reviews (%)")
    ax.set_title("Top 10 Accusations by Presence (Mapped Categories)")
    plt.tight_layout()

    # 6. Save to disk
    os.makedirs(OUT_DIR, exist_ok=True)
    fname = f"top5_accusations_{uuid.uuid4().hex[:6]}.png"
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path)
    plt.close(fig)

    # 7. Build table for frontend
    table = [{"category": idx, "percentage": val} for idx, val in counts.items()]
    return {"plot": path, "data_table": table}


# Returns monthly trends of Top-1 accusation categories
def get_top1_category_trends(input_path: str) -> dict:
    """
    Reads the Excel at input_path, picks each review's single highest-score
    label across all sources, maps it into one of the 12 categories,
    then computes the monthly % of reviews whose Top-1 falls in each category.
    Returns a JSON-serializable dict with:
      - chartData: list of { month: 'YYYY-MM', '<CategoryName>': pct, ... }
      - categories: list of category names (so the frontend can pick colors/legends)
    """
    df = pd.read_excel(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)

    def gather_pairs(row):
        pairs = []
        # 1) your three tier columns
        for acc_col, score_col in [
            ("most_accusations", "most_scores"),
            ("medium_accusations", "medium_scores"),
            ("less_accusations", "less_scores"),
        ]:
            accs = row.get(acc_col) or []
            scores = row.get(score_col) or []
            if isinstance(accs, str):
                accs = ast.literal_eval(accs)
                scores = ast.literal_eval(scores)
            pairs += list(zip(accs, scores))
        # 2) Top1–Top5
        for i in range(1, 6):
            a, s = row.get(f"Top{i}"), row.get(f"Top{i}_score")
            if pd.notna(a) and pd.notna(s):
                pairs.append((a, s))
        # 3) SemTop1–SemTop5
        for i in range(1, 6):
            a, s = row.get(f"SemTop{i}"), row.get(f"SemTop{i}_score")
            if pd.notna(a) and pd.notna(s):
                pairs.append((a, s))
        return pairs

    out = []
    for _, row in df.iterrows():
        pairs = gather_pairs(row)
        if not pairs:
            continue
        best = max(pairs, key=lambda x: x[1])[0]
        cat = CATEGORY_MAPPING.get(best, "Other")
        out.append({"month": row["month"], "category": cat})

    long = pd.DataFrame(out)

    grp = long.groupby(["month", "category"]).size().reset_index(name="count")
    total = df.groupby("month").size().reset_index(name="n_reviews")
    avg_rating = df.groupby("month")["rating"].mean().reset_index(name="avg_rating")
    merged = grp.merge(total, on="month")
    merged["pct"] = merged["count"] / merged["n_reviews"] * 100
    merged = merged.merge(avg_rating, on="month")

    pivot = merged.pivot(index="month", columns="category", values="pct").fillna(0)
    pivot = pivot.sort_index()

    chartData = pivot.reset_index().to_dict(orient="records")
    categories = list(pivot.columns)
    ratingSeries = avg_rating.sort_values("month").to_dict(orient="records")

    return {
        "chartData": chartData,
        "categories": categories,
        "ratingSeries": ratingSeries,
    }


# Returns benchmarks for each store based on accusations scores from different tiers and BART based Top1–Top5 , Semantic match based Top1–Top5
def get_store_benchmarks(input_path: str) -> dict:
    df = pd.read_excel(input_path)
    df["date"] = pd.to_datetime(df["date"])

    def gather_pairs(row):
        pairs = []
        # three tier lists
        for acc_col, score_col in [
            ("most_accusations", "most_scores"),
            ("medium_accusations", "medium_scores"),
            ("less_accusations", "less_scores"),
        ]:
            accs = row.get(acc_col) or []
            scores = row.get(score_col) or []
            if isinstance(accs, str):
                accs = ast.literal_eval(accs)
                scores = ast.literal_eval(scores)
            pairs += list(zip(accs, scores))
        # Top1–Top5
        for i in range(1, 6):
            a, s = row.get(f"Top{i}"), row.get(f"Top{i}_score")
            if pd.notna(a) and pd.notna(s):
                pairs.append((a, s))
        # SemTop1–SemTop5
        for i in range(1, 6):
            a, s = row.get(f"SemTop{i}"), row.get(f"SemTop{i}_score")
            if pd.notna(a) and pd.notna(s):
                pairs.append((a, s))
        return pairs

    records = []
    for _, row in df.iterrows():
        pairs = gather_pairs(row)
        if not pairs:
            continue
        top1 = max(pairs, key=lambda x: x[1])[0]

        city = row.get("city", "Unknown")
        rating = row.get("rating", np.nan)
        records.append({"city": city, "top1": _map_value(top1), "rating": rating})

    city_df = pd.DataFrame(records)
    agg = (
        city_df.groupby("city")
        .agg(
            avg_rating=("rating", "mean"),
            complaint_diversity=("top1", lambda s: s.nunique()),
            top1_category=(
                "top1",
                lambda s: s.mode().iat[0] if not s.mode().empty else None,
            ),
        )
        .reset_index()
    )

    return {"benchmarks": agg.to_dict(orient="records")}


# Group accusations by a specified column (e.g., tier, TopK, semantic)
def compare_accusation_by_group(
    input_path: str, group_by: str, source: str = "tier"
) -> dict:
    import ast
    import pandas as pd

    df = pd.read_excel(input_path)

    if group_by not in df.columns:
        raise ValueError(f"Grouping column '{group_by}' not found in uploaded file")

    required_columns = {
        "tier": [
            "most_accusations",
            "most_scores",
            "medium_accusations",
            "medium_scores",
            "less_accusations",
            "less_scores",
        ],
        "topk": [f"Top{i}" for i in range(1, 6)]
        + [f"Top{i}_score" for i in range(1, 6)],
        "semantic": [f"SemTop{i}" for i in range(1, 6)]
        + [f"SemTop{i}_score" for i in range(1, 6)],
    }

    missing = [col for col in required_columns[source] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for source '{source}': {missing}")

    def get_top3(row):
        pairs = []
        if source == "tier":
            for acc_col, score_col in [
                ("most_accusations", "most_scores"),
                ("medium_accusations", "medium_scores"),
                ("less_accusations", "less_scores"),
            ]:
                accs = row.get(acc_col)
                scores = row.get(score_col)
                if isinstance(accs, str):
                    accs = ast.literal_eval(accs)
                if isinstance(scores, str):
                    scores = ast.literal_eval(scores)
                if isinstance(accs, list) and isinstance(scores, list):
                    pairs += list(zip(accs, scores))
        elif source in {"topk", "semantic"}:
            prefix = "Top" if source == "topk" else "SemTop"
            for i in range(1, 6):
                a = row.get(f"{prefix}{i}")
                s = row.get(f"{prefix}{i}_score")
                if pd.notna(a) and pd.notna(s):
                    pairs.append((a, s))
        if not pairs:
            return []
        top3 = sorted(pairs, key=lambda x: x[1], reverse=True)[:3]
        return [_map_value(a) for a, _ in top3]

    records = []
    skipped = 0
    for _, row in df.iterrows():
        group = row.get(group_by)
        if pd.isna(group):
            continue
        top3 = get_top3(row)
        if not top3:
            skipped += 1
            continue
        for cat in top3:
            records.append({"group": group, "category": cat})

    if not records:
        raise ValueError(f"No valid accusations found using source '{source}'")

    print(
        f"[Grouped Breakdown] Total reviews: {len(df)}, "
        f"Skipped reviews (no accusations): {skipped}, "
        f"Top-3 accusation entries extracted: {len(records)}"
    )

    result_df = pd.DataFrame(records)
    counts = result_df.groupby(["group", "category"]).size().unstack(fill_value=0)
    percents = counts.div(counts.sum(axis=1), axis=0) * 100

    print("Grouped categories per group:")
    print(result_df.groupby(["group", "category"]).size())
    print("Result matrix (percentages):")
    print(percents.head())

    return {
        "group_col": group_by,
        "categories": percents.columns.tolist(),
        "groups": percents.index.tolist(),
        "matrix": percents.reset_index().to_dict(orient="records"),
    }


def extract_category_snippets(input_path: str):
    """
    Returns: (dict[category: List[str]], List[dict])
    1. category → evidence sentences (for word clouds)
    2. snippet_tables: [{review_index, evidence: {category: [sentences]}}]
    """
    df = pd.read_excel(input_path)
    result = defaultdict(list)
    snippet_tables = []

    for idx, row in df.iterrows():
        raw = row.get("evidence")
        if not isinstance(raw, str):
            continue
        try:
            evid_dict = ast.literal_eval(raw)
        except Exception:
            continue
        if not isinstance(evid_dict, dict):
            continue

        review_snippets = {}
        for label, sentences in evid_dict.items():
            if not isinstance(sentences, list):
                continue
            cat = _map_value(label)
            valid_sents = [
                s.strip() for s in sentences if isinstance(s, str) and len(s) > 3
            ]
            if not valid_sents:
                continue
            result[cat].extend(valid_sents)
            review_snippets[cat] = valid_sents
        snippet_tables.append({"review_index": idx, "evidence": review_snippets})

    return dict(result), snippet_tables


# Generate wordclouds from category snippets
def generate_wordclouds(input_path: str):
    snippets, _ = extract_category_snippets(input_path)
    results = {}
    for category, sents in snippets.items():
        text = " ".join(sents)
        if not text.strip():
            continue
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fname = f"{category.replace(' ', '_').replace('&', '').replace('/', '')[:30]}_{uuid.uuid4().hex[:6]}.png"
        path = os.path.join(WORDCLOUD_DIR, fname)
        wc.to_file(path)
        results[category] = path
    return results


# Embeded drilldown analysis for group_by column
def drilldown_analysis(input_path: str, group_by: str, group_value: str = None):
    df = pd.read_excel(input_path)
    if group_by not in df.columns:
        raise ValueError(f"Grouping column '{group_by}' not found in uploaded file")

    if group_value is not None:
        df = df[df[group_by] == group_value]

    n_reviews = len(df)
    avg_rating = float(df["rating"].mean()) if "rating" in df.columns else None

    category_counts = Counter()
    for _, row in df.iterrows():
        raw = row.get("evidence")
        if isinstance(raw, str):
            try:
                evid_dict = ast.literal_eval(raw)
            except Exception:
                continue
            for label, sentences in (evid_dict or {}).items():
                cat = _map_value(label)
                category_counts[cat] += (
                    len(sentences) if isinstance(sentences, list) else 1
                )
        else:
            top1 = row.get("Top1")
            if pd.notna(top1):
                cat = _map_value(top1)
                category_counts[cat] += 1

    top_categories = [cat for cat, _ in category_counts.most_common(3)]

    return {
        "group_by": group_by,
        "group_value": group_value,
        "n_reviews": n_reviews,
        "avg_rating": avg_rating,
        "top_categories": top_categories,
        "category_distribution": dict(category_counts),
    }


# Embeded wordcloud generation from top accusation using evidence and fallback to review text
def deep_analyze_service(
    input_path: str, group_by: str = None, group_value: str = None
):
    wordclouds = generate_wordclouds(input_path)
    snippets, snippet_tables = extract_category_snippets(input_path)
    dropdown = None
    if group_by:
        dropdown = drilldown_analysis(input_path, group_by, group_value)

    if not wordclouds:
        wordclouds = generate_wordclouds_from_top_accusation(
            input_path, text_col="cleaned_review"
        )

    return {
        "wordclouds": wordclouds,
        "snippet_tables": snippet_tables,
        "dropdown_summary": dropdown,
    }


# Wordclouds from top accusation using review text
def generate_wordclouds_from_top_accusation(
    input_path: str, text_col: str = "cleaned_review"
):
    df = pd.read_excel(input_path)

    # Use your gather_pairs definition
    def gather_pairs(row):
        pairs = []
        for acc_col, score_col in [
            ("most_accusations", "most_scores"),
            ("medium_accusations", "medium_scores"),
            ("less_accusations", "less_scores"),
        ]:
            accs = row.get(acc_col) or []
            scores = row.get(score_col) or []
            if isinstance(accs, str):
                accs, scores = ast.literal_eval(accs), ast.literal_eval(scores)
            for a, s in zip(accs, scores):
                pairs.append((_map_value(a), s))
        for i in range(1, 6):
            a = row.get(f"Top{i}")
            s = row.get(f"Top{i}_score")
            if pd.notna(a) and pd.notna(s):
                pairs.append((_map_value(a), s))
        for i in range(1, 6):
            a = row.get(f"SemTop{i}")
            s = row.get(f"SemTop{i}_score")
            if pd.notna(a) and pd.notna(s):
                pairs.append((_map_value(a), s))
        return pairs

    cat_to_texts = defaultdict(list)
    for _, row in df.iterrows():
        pairs = gather_pairs(row)
        if not pairs:
            continue
        top1 = sorted(pairs, key=lambda x: x[1], reverse=True)[0][0]
        text = row.get(text_col, "")
        if isinstance(text, str) and text.strip():
            cat_to_texts[top1].append(text.strip())

    results = {}
    for cat, texts in cat_to_texts.items():
        blob = " ".join(texts)
        if not blob.strip():
            continue
        wc = WordCloud(width=800, height=400, background_color="white").generate(blob)
        fname = f"{cat.replace(' ', '_').replace('&', '').replace('/', '')[:30]}_{uuid.uuid4().hex[:6]}.png"
        path = os.path.join("data/interim/wordclouds", fname)
        wc.to_file(path)
        results[cat] = path
    return results
