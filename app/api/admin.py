"""Admin and monitoring API endpoints."""

import logging
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.config import get_config
from app.models.database import Face, Image, Person, ProcessingLog, SearchLog, SourceDirectory, get_session
from app.models.schemas import HealthResponse, SystemStats

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin & Monitoring"])

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check with statistics."""
    from app import __version__
    from app.services.face_index import get_face_index

    session = get_session()
    cfg = get_config()

    total_people = session.query(Person).count()
    total_images = session.query(Image).filter_by(processed=True).count()
    total_faces = session.query(Face).count()
    failed_images = session.query(Image).filter(Image.error.isnot(None)).count()
    source_dirs = session.query(SourceDirectory).count()

    index = get_face_index()

    session.close()

    return HealthResponse(
        status="healthy",
        version=__version__,
        uptime_seconds=round(time.time() - _start_time, 1),
        stats=SystemStats(
            total_people=total_people,
            total_images=total_images,
            total_faces=total_faces,
            index_size=index.size,
            model_version=cfg.face_detection.model_pack,
            index_type=cfg.search.index_type,
            failed_images=failed_images,
            source_directories=source_dirs,
        ),
    )


@router.get("/people")
async def list_people(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List all enrolled people with their image and face counts."""
    session = get_session()

    people = (
        session.query(Person)
        .order_by(Person.person_id)
        .offset(offset)
        .limit(limit)
        .all()
    )

    total = session.query(Person).count()
    session.close()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "people": [
            {
                "person_id": p.person_id,
                "name": p.name,
                "image_count": p.image_count,
                "face_count": p.face_count,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in people
        ],
    }


@router.get("/people/{person_id}")
async def get_person(person_id: str):
    """Get details about a specific enrolled person."""
    session = get_session()

    person = session.query(Person).filter_by(person_id=person_id).first()
    if not person:
        session.close()
        raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")

    images = session.query(Image).filter_by(person_id_fk=person.id).all()
    session.close()

    return {
        "person_id": person.person_id,
        "name": person.name,
        "image_count": person.image_count,
        "face_count": person.face_count,
        "created_at": person.created_at.isoformat() if person.created_at else None,
        "images": [
            {
                "file_path": img.file_path,
                "processed": img.processed,
                "face_count": img.face_count,
                "error": img.error,
                "width": img.width,
                "height": img.height,
            }
            for img in images
        ],
    }


@router.get("/failed-images")
async def list_failed_images(
    limit: int = Query(100, ge=1, le=1000),
):
    """List all images that failed processing, with error reasons."""
    session = get_session()

    failed = (
        session.query(Image)
        .filter(Image.error.isnot(None))
        .order_by(Image.created_at.desc())
        .limit(limit)
        .all()
    )

    session.close()

    return {
        "count": len(failed),
        "images": [
            {
                "file_path": img.file_path,
                "error": img.error,
                "person_id": None,
            }
            for img in failed
        ],
    }


@router.get("/logs/processing")
async def processing_logs(limit: int = Query(20, ge=1, le=100)):
    """View recent processing/enrollment logs."""
    session = get_session()

    logs = (
        session.query(ProcessingLog)
        .order_by(ProcessingLog.started_at.desc())
        .limit(limit)
        .all()
    )

    session.close()

    return {
        "logs": [
            {
                "id": log.id,
                "run_type": log.run_type,
                "started_at": log.started_at.isoformat() if log.started_at else None,
                "finished_at": log.finished_at.isoformat() if log.finished_at else None,
                "total_images": log.total_images,
                "images_processed": log.images_processed,
                "faces_detected": log.faces_detected,
                "images_skipped": log.images_skipped,
                "images_failed": log.images_failed,
                "model_version": log.model_version,
            }
            for log in logs
        ],
    }


@router.get("/logs/searches")
async def search_logs(limit: int = Query(50, ge=1, le=500)):
    """View recent search request logs."""
    session = get_session()

    logs = (
        session.query(SearchLog)
        .order_by(SearchLog.timestamp.desc())
        .limit(limit)
        .all()
    )

    session.close()

    return {
        "logs": [
            {
                "request_id": log.request_id,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "faces_in_query": log.faces_in_query,
                "top_k": log.top_k,
                "total_candidates": log.total_candidates,
                "processing_time_ms": log.processing_time_ms,
            }
            for log in logs
        ],
    }


@router.get("/config")
async def get_current_config():
    """View the current system configuration."""
    cfg = get_config()
    return {
        "face_detection": {
            "model_pack": cfg.face_detection.model_pack,
            "detection_threshold": cfg.face_detection.detection_threshold,
            "min_face_size": cfg.face_detection.min_face_size,
        },
        "search": {
            "default_top_k": cfg.search.default_top_k,
            "index_type": cfg.search.index_type,
        },
        "thresholds": {
            "high_confidence": cfg.thresholds.high_confidence,
            "possible_match": cfg.thresholds.possible_match,
            "low_confidence": cfg.thresholds.low_confidence,
        },
    }


@router.get("/image")
async def serve_image(path: str = Query(..., description="Absolute path to the image file")):
    """
    Serve an enrolled image file for preview.

    Takes an absolute file path and returns the image if it exists.
    Only serves image files for security.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    # Only serve image files
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}
    if file_path.suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Not an image file")

    return FileResponse(
        str(file_path),
        media_type=f"image/{file_path.suffix.lower().strip('.')}",
    )
