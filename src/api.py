
# =============================================================================
# Standard Library Imports
# =============================================================================

import io
import os
import uuid
#from contextlib import asynccontextmanager
#from typing import List, Optional

# =============================================================================
# Third-Party Imports
# =============================================================================
from typing import Optional
#import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
#from pydantic import BaseModel
#from sentence_transformers import SentenceTransformer

# =============================================================================
# Local Application Imports
# =============================================================================

#from src.features import clean_text
from src.services.analysis import enrich_dataframe, load_config
from src.services.cleaning import prepare_reviews
from src.services.paths import CLEANED_DIR, DATA_DIR, RAW_DATA_DIR, SEMANTIC_DIR
from src.services.semanticservice import map_accusations
from src.services.visualizationservice import (
    compare_accusation_by_group,
    deep_analyze_service,
    get_store_benchmarks,
    get_top1_category_trends,
    plot_semantic_topic_heatmap,
    plot_top10_accusation_heatmap,
    plot_top10_by_rank_heatmap,
    plot_top5_accusations,
)


# =============================================================================
# Project Paths
# =============================================================================

BASE_DIR = os.path.dirname(__file__)  # .../src
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
BUILD_DIR = os.path.join(PROJECT_DIR, "frontend", "build")

# =============================================================================
# FastAPI App Setup
# =============================================================================

app = FastAPI(
    title="Supermarket Review Analysis API",
    description="Cleaning, semantic mapping, and visualization of customer reviews",
    version="1.0.0",
)


# =============================================================================
# Helper Functions
# =============================================================================

async def save_upload_file(
    file: UploadFile,
    prefix: str,
    upload_dir: str = RAW_DATA_DIR,
) -> str:
    os.makedirs(upload_dir, exist_ok=True)

    original_name = file.filename or "uploaded.xlsx"
    _, ext = os.path.splitext(original_name)

    if not ext:
        ext = ".xlsx"

    filename = f"{prefix}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(upload_dir, filename)

    with open(path, "wb") as out:
        out.write(await file.read())

    return path

# =============================================================================
# Data Preparation Routes
# =============================================================================

@app.post("/prepare")
async def prepare_endpoint(file: UploadFile = File(...)):
    # Ensure the raw folder exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    original_name = file.filename
    raw_path = os.path.join(RAW_DATA_DIR, original_name)

    if os.path.exists(raw_path):
        # Append a UUID to the base name, before the extension
        name, ext = os.path.splitext(original_name)
        unique_name = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        raw_path = os.path.join(RAW_DATA_DIR, unique_name)

    # 1) Write the upload to disk
    try:
        contents = await file.read()
        with open(raw_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to save uploaded file: {e}")

    # 2) Run cleaning and splitting service
    try:
        prepare_reviews(input_path=raw_path, out_dir=CLEANED_DIR)
    except Exception as e:
        raise HTTPException(500, detail=f"Prepare service failed: {e}")

    # 3) List the generated files
    generated = os.listdir(CLEANED_DIR)

    return JSONResponse(
        {
            "detail": "Prepared reviews successfully",
            "raw_file": os.path.basename(raw_path),
            "generated_files": generated,
        }
    )




# =============================================================================
# Analysis Routes
# =============================================================================

@app.post("/analysis")
def analysis(
    file: UploadFile = File(...),
    model: Optional[str] = None,
    topk: Optional[int] = 3,
):
    try:
        content = file.file.read()
        df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Excel file: {e}")

    df["true_accusations"] = df["accusations"].apply(
        lambda x: x
        if isinstance(x, list)
        else (__import__("ast").literal_eval(x) if isinstance(x, str) else [])
    )

    labels, keywords = load_config()

    enriched = enrich_dataframe(df, model, labels, keywords, topk)

    return enriched.to_dict(orient="records")


# =============================================================================
# Semantic Mapping Routes
# =============================================================================

@app.post("/semantic/map")
async def semantic_map(file: UploadFile = File(...)):
    # 1) Persist the raw upload
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    original = file.filename or "uploaded.xlsx"
    base, ext = os.path.splitext(original)

    raw_path = os.path.join(RAW_DATA_DIR, f"{base}_{uuid.uuid4().hex[:8]}{ext}")

    with open(raw_path, "wb") as fh:
        fh.write(await file.read())

    # 2) Choose output name inside SEMANTIC_DIR
    os.makedirs(SEMANTIC_DIR, exist_ok=True)

    out_name = f"{base}_mapped_{uuid.uuid4().hex[:6]}.xlsx"
    out_path = os.path.join(SEMANTIC_DIR, out_name)

    # 3) Run mapper
    try:
        map_accusations(input_path=raw_path, output_path=out_path)
    except Exception as e:
        raise HTTPException(500, f"Semantic mapping failed: {e}")

    # 4) Respond with list of semantic files
    generated = os.listdir(SEMANTIC_DIR)

    return JSONResponse(
        {
            "detail": "Semantic mapping complete",
            "raw_file": os.path.basename(raw_path),
            "generated_files": generated,
        }
    )


# =============================================================================
# Visualization Routes
# =============================================================================


@app.post("/visualize")
async def visualize(file: UploadFile = File(...)):
    raw_path = await save_upload_file(file, prefix="viz")

    try:
        result = plot_top10_accusation_heatmap(raw_path)
    except Exception as e:
        raise HTTPException(500, f"Visualization failed: {e}")

    return {
        "plot": result["plot"],
        "top10_labels": result["top10_labels"],
        "data_table": result["data_table"],
    }


@app.post("/visualize/by-rank")
async def visualize_by_rank(file: UploadFile = File(...)):
    raw_path = await save_upload_file(file, prefix="top5rank")

    try:
        result = plot_top10_by_rank_heatmap(raw_path)
    except Exception as e:
        raise HTTPException(500, f"Rank-based visualization failed: {e}")

    return {
        "plot": result["plot"],
        "top10_labels": result["top10_labels"],
        "data_table": result["data_table"],
    }


@app.post("/visualize/semantic")
async def visualize_semantic(file: UploadFile = File(...)):
    raw_path = await save_upload_file(file, prefix="semantic")

    try:
        result = plot_semantic_topic_heatmap(raw_path)
    except Exception as e:
        raise HTTPException(500, f"Semantic heatmap generation failed: {e}")

    return {
        "plot": result["plot"],
        "top10_labels": result["top10_labels"],
        "data_table": result["data_table"],
    }


@app.post("/visualize/top-accusations")
async def visualize_top5_accusations(file: UploadFile = File(...)):
    raw = await save_upload_file(file, prefix="top5")

    try:
        result = plot_top5_accusations(raw)
    except Exception as e:
        raise HTTPException(500, f"Top-5 accusations failed: {e}")

    return JSONResponse(
        {
            "plot": result["plot"],
            "data_table": result["data_table"],
        }
    )


@app.post("/visualize/top1-category-trends")
async def top1_category_trends(file: UploadFile = File(...)):
    raw_path = await save_upload_file(file, prefix="trend")

    try:
        return get_top1_category_trends(raw_path)
    except Exception as e:
        raise HTTPException(500, f"Error generating trends: {e}")


@app.post("/visualize/store-benchmarks")
async def store_benchmarks(file: UploadFile = File(...)):
    raw_path = await save_upload_file(file, prefix="bench")
    try:
        return get_store_benchmarks(raw_path)
    except Exception as e:
        raise HTTPException(500, f"Error: {e}")


@app.post("/visualize/grouped-breakdown")
async def visualize_grouped_breakdown(
    file: UploadFile = File(...),
    group_by: str = "city",
    source: str = "tier",
):
    path = await save_upload_file(file, prefix="grouped")

    try:
        result = compare_accusation_by_group(path, group_by, source)
    except Exception as e:
        raise HTTPException(400, f"Grouped breakdown failed: {e}")

    return result


@app.post("/visualize/deep-analysis")
async def deep_analysis(
    file: UploadFile = File(...),
    group_by: str = None,
    group_value: str = None,
):
    raw_path = await save_upload_file(file, prefix="deep")
    result = deep_analyze_service(raw_path, group_by, group_value)

    return result


# =============================================================================
# Static Files
# =============================================================================

app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")


# app.mount("/", StaticFiles(directory=BUILD_DIR, html=True), name="frontend")