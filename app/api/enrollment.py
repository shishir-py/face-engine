"""Enrollment API endpoints with source directory management."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.config import get_config
from app.models.schemas import (
    EnrollmentResponse,
    PersonEnrollResponse,
    SourceDirectoryInfo,
    SourceListResponse,
)
from app.services.enrollment import EnrollmentPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/enrollment", tags=["Enrollment"])


# ─── Directory Browser ───────────────────────────────────────

@router.get("/browse")
async def browse_directories(
    path: str = Query("/", description="Directory path to list contents of"),
):
    """
    Browse the server filesystem to select a directory for enrollment.

    Returns subdirectories and basic info about the given path,
    allowing the UI to render a folder picker/navigator.
    """
    target = Path(path).expanduser().resolve()

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    # Count image files at this level (to help user decide)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    image_count = sum(
        1 for f in target.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    )

    # Count subdirectories that contain images (structured layout hint)
    subdirs_with_images = 0

    entries = []
    try:
        for item in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if item.name.startswith("."):
                continue  # Skip hidden files/dirs
            if item.is_dir():
                # Count images inside this subdir (one level)
                try:
                    sub_images = sum(
                        1 for f in item.iterdir()
                        if f.is_file() and f.suffix.lower() in image_extensions
                    )
                except PermissionError:
                    sub_images = 0
                if sub_images > 0:
                    subdirs_with_images += 1
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory",
                    "image_count": sub_images,
                })
            elif item.is_file() and item.suffix.lower() in image_extensions:
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "image",
                    "size": item.stat().st_size,
                })
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

    # Detect likely layout
    if subdirs_with_images >= 2:
        suggested_layout = "structured"
    elif image_count > 0:
        suggested_layout = "flat"
    else:
        suggested_layout = "auto"

    return {
        "current_path": str(target),
        "parent_path": str(target.parent) if target != target.parent else None,
        "entries": entries,
        "image_count": image_count,
        "subdirs_with_images": subdirs_with_images,
        "suggested_layout": suggested_layout,
    }


# ─── Source Directory Management ──────────────────────────────

@router.post("/sources/add", response_model=EnrollmentResponse)
async def add_source_directory(
    path: str = Form(..., description="Absolute path to image directory"),
    layout: str = Form("auto", description="Layout: auto | structured | flat"),
    alias: Optional[str] = Form(None, description="Friendly name for this source"),
):
    """
    Add and scan a new image source directory.

    The directory will be registered and immediately scanned for faces.

    **Layout modes:**
    - `auto` (default): Automatically detect if the directory is
      structured (person-per-folder) or flat (all images loose).
    - `structured`: Each subdirectory is treated as one person.
    - `flat`: All images are indexed individually for image-level search.
    """
    if layout not in ("auto", "structured", "flat"):
        raise HTTPException(status_code=400, detail=f"Invalid layout: {layout}. Use auto, structured, or flat.")

    pipeline = EnrollmentPipeline()
    try:
        result = pipeline.scan_directory(path, layout=layout, alias=alias)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Source directory scan failed")
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

    return EnrollmentResponse(
        status="completed",
        stats=result["stats"],
        failed_images=result["failed_images"],
    )


@router.get("/sources", response_model=SourceListResponse)
async def list_source_directories():
    """List all registered image source directories with their scan status."""
    pipeline = EnrollmentPipeline()
    sources = pipeline.list_sources()
    return SourceListResponse(
        sources=[SourceDirectoryInfo(**s) for s in sources],
        total=len(sources),
    )


@router.post("/sources/{source_id}/scan", response_model=EnrollmentResponse)
async def rescan_source(source_id: int):
    """Re-scan a specific source directory for new images."""
    from app.models.database import SourceDirectory, get_session

    session = get_session()
    source = session.query(SourceDirectory).filter_by(id=source_id).first()
    if source is None:
        session.close()
        raise HTTPException(status_code=404, detail=f"Source directory {source_id} not found.")

    source_path = source.path
    source_layout = source.layout
    session.close()

    pipeline = EnrollmentPipeline()
    try:
        result = pipeline.scan_directory(source_path, layout=source_layout)
    except Exception as e:
        logger.exception("Source rescan failed")
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

    return EnrollmentResponse(
        status="completed",
        stats=result["stats"],
        failed_images=result["failed_images"],
    )


# ─── Standard Enrollment ──────────────────────────────────────

@router.post("/scan", response_model=EnrollmentResponse)
async def scan_repository(
    repository_path: Optional[str] = Form(None, description="Path to image repository. Defaults to configured path."),
):
    """
    Scan the image repository and enroll all people.

    If a path is provided, it is scanned with auto-detected layout.
    Otherwise, the default configured repository is scanned in structured mode.
    """
    pipeline = EnrollmentPipeline()
    try:
        if repository_path:
            # Use the new scan_directory which auto-detects layout
            result = pipeline.scan_directory(repository_path)
        else:
            result = pipeline.scan_repository()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Enrollment scan failed")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")

    return EnrollmentResponse(
        status="completed",
        stats=result["stats"],
        failed_images=result["failed_images"],
    )


@router.post("/person", response_model=PersonEnrollResponse)
async def enroll_person(
    person_id: str = Form(..., description="Unique identifier for the person"),
    name: Optional[str] = Form(None, description="Display name for the person"),
    images: List[UploadFile] = File(..., description="Face images to enroll"),
):
    """
    Enroll a new person or add images to an existing person.

    Upload one or more images of the same person. The system will detect
    faces, extract embeddings, and add them to the search index.
    """
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    cfg = get_config()

    # Save uploaded images to a temporary directory
    temp_dir = tempfile.mkdtemp()
    saved_paths = []

    try:
        for img in images:
            if img.content_type and not img.content_type.startswith("image/"):
                continue

            content = await img.read()
            if len(content) == 0:
                continue

            # Save to person's directory in the repository
            person_dir = Path(cfg.paths.image_repository) / person_id
            person_dir.mkdir(parents=True, exist_ok=True)

            dest = person_dir / img.filename
            with open(dest, "wb") as f:
                f.write(content)
            saved_paths.append(dest)

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid images provided.")

        pipeline = EnrollmentPipeline()
        result = pipeline.enroll_person(person_id, saved_paths, name=name)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Person enrollment failed")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return PersonEnrollResponse(
        status="completed",
        person_id=result["person_id"],
        images_processed=result["images_processed"],
        faces_detected=result["faces_detected"],
        failed_images=result["failed_images"],
    )


@router.post("/reprocess", response_model=EnrollmentResponse)
async def reprocess_all():
    """
    Reprocess all images and rebuild the search index.

    Use this after changing the face detection/recognition model,
    or when existing embeddings need to be regenerated.
    """
    pipeline = EnrollmentPipeline()
    try:
        result = pipeline.reprocess_all()
    except Exception as e:
        logger.exception("Reprocessing failed")
        raise HTTPException(status_code=500, detail=f"Reprocessing failed: {str(e)}")

    return EnrollmentResponse(
        status="completed",
        stats=result["stats"],
        failed_images=result["failed_images"],
    )
