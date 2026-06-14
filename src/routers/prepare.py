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
from src.services.cleaning import prepare_reviews
from src.services.paths import CLEANED_DIR, RAW_DATA_DIR


router = APIRouter()


@router.post("/prepare")
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
