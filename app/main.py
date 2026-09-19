"""FaceMatch AI — FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app import __version__
from app.config import load_config
from app.models.database import init_db
from app.services.face_detector import get_detector
from app.services.face_index import get_face_index
from app.api import search, enrollment, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown."""
    cfg = load_config()

    # Configure logging
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    logger = logging.getLogger("facematch")

    logger.info("Starting FaceMatch AI v%s", __version__)

    # Ensure data directories exist
    Path(cfg.paths.image_repository).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.faiss_index).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.upload_temp).mkdir(parents=True, exist_ok=True)

    # Initialize database
    init_db(cfg.paths.database)
    logger.info("Database initialized at %s", cfg.paths.database)

    # Initialize face detector (loads ML models)
    detector = get_detector()
    detector.initialize()

    # Initialize FAISS index (load from disk if exists)
    index = get_face_index()
    index.initialize()
    logger.info("FAISS index loaded with %d embeddings.", index.size)

    logger.info("FaceMatch AI ready — serving on %s:%d", cfg.server.host, cfg.server.port)

    yield

    # Shutdown
    logger.info("Shutting down FaceMatch AI...")


app = FastAPI(
    title="FaceMatch AI",
    description=(
        "Intelligent Face Search & Recognition Platform. "
        "Detect faces, extract embeddings, and search against a repository of enrolled identities."
    ),
    version=__version__,
    lifespan=lifespan,
)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(search.router)
app.include_router(enrollment.router)
app.include_router(admin.router)

# Serve static files (web UI)
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the web UI."""
    index_html = static_dir / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return {
        "name": "FaceMatch AI",
        "version": __version__,
        "docs": "/docs",
        "health": "/admin/health",
    }
