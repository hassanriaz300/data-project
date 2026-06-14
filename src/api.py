# =============================================================================
# Third-Party Imports
# =============================================================================
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# =============================================================================
# Local Application Imports
# =============================================================================
from src.routers.prepare import router as prepare_router
from src.routers.semantic import router as semantic_router
from src.routers.visualization import router as visualization_router
from src.services.paths import DATA_DIR


# =============================================================================
# FastAPI App Setup
# =============================================================================
app = FastAPI(
    title="Supermarket Review Analysis API",
    description="Cleaning, semantic mapping, and visualization of customer reviews",
    version="1.0.0",
)


# =============================================================================
# Routers
# =============================================================================
app.include_router(prepare_router)
app.include_router(semantic_router)
app.include_router(visualization_router)


# =============================================================================
# Static Files
# =============================================================================
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")
