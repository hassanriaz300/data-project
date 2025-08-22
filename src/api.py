# src/api.py
import uuid
import os
import io
import joblib
import yaml
import pandas as pd
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from src.services.cleaning import prepare_reviews
from src.services.paths import RAW_DATA_DIR, CLEANED_DIR, SEMANTIC_DIR

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.features import clean_text
from src.dataset import load_raw_reviews
from src.services.analysis import enrich_dataframe, load_config
from src.services.semanticservice import map_accusations
from fastapi.responses import StreamingResponse
import tempfile
from src.services.visualizationservice import plot_top10_accusation_heatmap
from src.services.visualizationservice import plot_top10_by_rank_heatmap
from src.services.visualizationservice import plot_semantic_topic_heatmap
from src.services.visualizationservice import plot_top5_accusations
from src.services.visualizationservice import get_top1_category_trends
from src.services.visualizationservice import get_store_benchmarks
from src.services.visualizationservice import compare_accusation_by_group


from src.services.visualizationservice import deep_analyze_service


BASE_DIR = os.path.dirname(__file__)  # …/src
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
BUILD_DIR = os.path.join(PROJECT_DIR, "frontend", "build")


class ModelBundle:
    def __init__(self, path: str):
        bundle = joblib.load(path)
        self.embedder_name = bundle["embedder_name"]
        self.mlb = bundle["mlb"]
        self.clf = bundle["clf"]
        self.embedder = SentenceTransformer(self.embedder_name)


model_bundle: ModelBundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the model bundle
    global model_bundle
    try:
        model_bundle = ModelBundle("models/multilabel_model.joblib")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    yield
    # Shutdown: (nothing special to do)


app = FastAPI(
    title="Supermarket Review Classifier",
    description="Multilabel classification & analysis of customer reviews",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    texts: List[str]
    thresh: Optional[float] = 0.5
    topk: Optional[int] = 3


class PredictResponseItem(BaseModel):
    text: str
    scores: dict
    predicted: List[str]


class PredictResponse(BaseModel):
    predictions: List[PredictResponseItem]


# This endpoint prepares the reviews for classification and returns the cleaned files
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

    # 2) Run your cleaning & splitting service
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


# This endpoint performs multilabel classification on the provided text future work not implemeted yet
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    cleaned = [clean_text(t) for t in req.texts]
    embeds = model_bundle.embedder.encode(cleaned, convert_to_tensor=False)
    probs = model_bundle.clf.predict_proba(embeds)

    results = []
    classes = model_bundle.mlb.classes_
    for text, row in zip(req.texts, probs):
        sel = [cls for cls, p in zip(classes, row) if p >= req.thresh]
        if not sel:
            top_idxs = row.argsort()[::-1][: req.topk]
            sel = [classes[i] for i in top_idxs]

        results.append(
            PredictResponseItem(
                text=text,
                scores={cls: float(p) for cls, p in zip(classes, row)},
                predicted=sel,
            )
        )

    return PredictResponse(predictions=results)


# Endpoint for analysis of uploaded Excel files
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
    model_name = model or model_bundle.embedder_name

    enriched = enrich_dataframe(df, model_name, labels, keywords, topk)
    return enriched.to_dict(orient="records")


# Endpoint for semantic mapping of accusations
@app.post("/semantic/map")
async def semantic_map(file: UploadFile = File(...)):
    # 1.  Persist the raw upload just like /prepare ----------------
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    original = file.filename or "uploaded.xlsx"
    base, ext = os.path.splitext(original)
    raw_path = os.path.join(RAW_DATA_DIR, f"{base}_{uuid.uuid4().hex[:8]}{ext}")
    with open(raw_path, "wb") as fh:
        fh.write(await file.read())

    # 2.  Choose an output name inside SEMANTIC_DIR ----------------
    os.makedirs(SEMANTIC_DIR, exist_ok=True)
    out_name = f"{base}_mapped_{uuid.uuid4().hex[:6]}.xlsx"
    out_path = os.path.join(SEMANTIC_DIR, out_name)

    # 3.  Run the mapper ------------------------------------------
    try:
        map_accusations(input_path=raw_path, output_path=out_path)
    except Exception as e:
        raise HTTPException(500, f"Semantic mapping failed: {e}")

    # 4.  Respond with the *list* of semantic files ---------------
    generated = os.listdir(SEMANTIC_DIR)
    return JSONResponse(
        {
            "detail": "Semantic mapping complete",
            "raw_file": os.path.basename(raw_path),
            "generated_files": generated,
        }
    )


# Endpoint for visualization of accusation heatmaps
@app.post("/visualize")
async def visualize(file: UploadFile = File(...)):
    # Save uploaded file to disk
    raw_path = os.path.join("data/raw", f"viz_{uuid.uuid4().hex[:8]}.xlsx")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    try:
        result = plot_top10_accusation_heatmap(raw_path)
    except Exception as e:
        raise HTTPException(500, f"Visualization failed: {e}")

    return {
        "plot": result["plot"],
        "top10_labels": result["top10_labels"],
        "data_table": result["data_table"],
    }


# Endpoint for various visualizations
@app.post("/visualize/by-rank")
async def visualize_by_rank(file: UploadFile = File(...)):
    # Save uploaded file to disk
    raw_path = os.path.join("data/raw", f"top5rank_{uuid.uuid4().hex[:8]}.xlsx")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    try:
        result = plot_top10_by_rank_heatmap(raw_path)
    except Exception as e:
        raise HTTPException(500, f"Rank-based visualization failed: {e}")

    return {
        "plot": result["plot"],
        "top10_labels": result["top10_labels"],
        "data_table": result["data_table"],
    }


# Endpoint for semantic topic heatmap visualization
@app.post("/visualize/semantic")
async def visualize_semantic(file: UploadFile = File(...)):
    # 1. Save the upload
    raw_path = os.path.join("data/raw", f"semantic_{uuid.uuid4().hex[:8]}.xlsx")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    # 2. Call the new service
    try:
        from src.services.visualizationservice import plot_semantic_topic_heatmap

        result = plot_semantic_topic_heatmap(raw_path)
    except Exception as e:
        raise HTTPException(500, f"Semantic heatmap generation failed: {e}")

    # 3. Return the plot path + data
    return {
        "plot": result["plot"],
        "top10_labels": result["top10_labels"],
        "data_table": result["data_table"],
    }


#
@app.post("/visualize/top-accusations")
async def visualize_top5_accusations(file: UploadFile = File(...)):
    raw = os.path.join("data/raw", f"top5_{uuid.uuid4().hex[:8]}.xlsx")
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    with open(raw, "wb") as out:
        out.write(await file.read())

    try:
        result = plot_top5_accusations(raw)
    except Exception as e:
        raise HTTPException(500, f"Top-5 accusations failed: {e}")

    fname = os.path.basename(result["plot"])
    return JSONResponse(
        {
            "plot": result["plot"],
            "data_table": result["data_table"],
        }
    )


# Endpoint for top-1 category trends visualization
@app.post("/visualize/top1-category-trends")
async def top1_category_trends(file: UploadFile = File(...)):
    raw_path = f"data/raw/trend_{uuid.uuid4().hex[:8]}.xlsx"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(await file.read())
    try:
        return get_top1_category_trends(raw_path)
    except Exception as e:
        raise HTTPException(500, f"Error generating trends: {e}")


#
@app.post("/visualize/store-benchmarks")
async def store_benchmarks(file: UploadFile = File(...)):
    raw_path = f"data/raw/bench_{uuid.uuid4().hex[:8]}.xlsx"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(await file.read())
    try:
        return get_store_benchmarks(raw_path)
    except Exception as e:
        raise HTTPException(500, f"Error: {e}")


#
@app.post("/visualize/grouped-breakdown")
async def visualize_grouped_breakdown(
    file: UploadFile = File(...),
    group_by: str = "city",
    source: str = "tier",  # can be: "tier", "topk", or "semantic"
):
    path = os.path.join("data/raw", f"grouped_{uuid.uuid4().hex[:8]}.xlsx")
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        result = compare_accusation_by_group(path, group_by, source)
    except Exception as e:
        raise HTTPException(400, f"Grouped breakdown failed: {e}")
    return result


def _save_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1]
    path = f"data/raw/tmp_{uuid.uuid4().hex[:8]}{ext}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as out:
        out.write(file.file.read())
    return path


# This endpoint performs a deep analysis of the uploaded file DeepAnalysis web UI
@app.post("/visualize/deep-analysis")
async def deep_analysis(
    file: UploadFile = File(...), group_by: str = None, group_value: str = None
):
    # Save uploaded file to disk
    raw_path = os.path.join("data/raw", f"deep_{uuid.uuid4().hex[:8]}.xlsx")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(await file.read())
    result = deep_analyze_service(raw_path, group_by, group_value)
    return result


app.mount("/static", StaticFiles(directory="data"), name="static")
app.mount("/", StaticFiles(directory=BUILD_DIR, html=True), name="frontend")
