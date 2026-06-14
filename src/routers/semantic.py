# =============================================================================
# Standard Library Imports
# =============================================================================
import os
import uuid

# =============================================================================
# Third-Party Imports
# =============================================================================
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# =============================================================================
# Local Application Imports
# =============================================================================
from src.services.paths import RAW_DATA_DIR, SEMANTIC_DIR
from src.services.semanticservice import map_accusations


router = APIRouter()


@router.post("/semantic/map")
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
