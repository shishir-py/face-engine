"""Face search API endpoints."""

import io
import logging
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.models.schemas import SearchResponse
from app.services.search import FaceSearchService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/face", tags=["Face Search"])


@router.post("/search", response_model=SearchResponse)
async def search_face(
    image: UploadFile = File(..., description="Image to search for matching faces"),
    top_k: Optional[int] = Form(None, description="Number of candidates to return per face"),
    search_mode: Optional[str] = Form("person", description="Search mode: person | image | both"),
    high_confidence: Optional[float] = Form(None, description="Threshold for high-confidence match"),
    possible_match: Optional[float] = Form(None, description="Threshold for possible match"),
    low_confidence: Optional[float] = Form(None, description="Threshold for low-confidence match"),
):
    """
    Upload an image and search for matching people/images in the repository.

    **Search modes:**
    - `person` (default): Returns ranked person-level matches with aggregated scores.
    - `image`: Returns individual matching images with file paths — useful for
      finding all images where a face appears across unsorted directories.
    - `both`: Returns both person-level and image-level results.

    **Match statuses (person mode):**
    - `high_confidence`: Strong match — very likely the same person.
    - `possible_match`: Moderate similarity — may be the same person.
    - `low_confidence`: Weak similarity — unlikely but not impossible.
    - `no_match`: Below threshold — not a match.
    """
    # Validate file type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {image.content_type}. Expected an image.")

    # Read image bytes
    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {str(e)}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file.")

    if len(image_bytes) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 20MB.")

    # Validate search mode
    if search_mode not in ("person", "image", "both"):
        raise HTTPException(status_code=400, detail=f"Invalid search_mode: {search_mode}. Use person, image, or both.")

    # Build threshold override if any custom values provided
    threshold_override = None
    if any(v is not None for v in [high_confidence, possible_match, low_confidence]):
        from app.config import get_config
        cfg = get_config()
        threshold_override = {
            "high_confidence": high_confidence if high_confidence is not None else cfg.thresholds.high_confidence,
            "possible_match": possible_match if possible_match is not None else cfg.thresholds.possible_match,
            "low_confidence": low_confidence if low_confidence is not None else cfg.thresholds.low_confidence,
        }

    # Run search
    service = FaceSearchService()
    result = service.search_image_bytes(
        image_bytes=image_bytes,
        top_k=top_k,
        threshold_override=threshold_override,
        search_mode=search_mode,
    )

    if result.faces_detected == 0:
        logger.info("Search request %s: no faces detected in uploaded image.", result.request_id)

    return result


@router.post("/download-matches")
async def download_matches(
    paths: str = Form(..., description="Comma-separated list of absolute image paths to download"),
    folder_name: str = Form("matched_faces", description="Name for the folder inside the ZIP"),
):
    """
    Download matched images as a ZIP file.

    Takes a comma-separated list of absolute image paths and packages them
    into a ZIP archive with a user-specified folder name.
    """
    path_list = [p.strip() for p in paths.split(",") if p.strip()]

    if not path_list:
        raise HTTPException(status_code=400, detail="No image paths provided.")

    if len(path_list) > 500:
        raise HTTPException(status_code=400, detail="Too many images. Maximum is 500.")

    # Sanitize folder name
    safe_folder = "".join(c for c in folder_name if c.isalnum() or c in " _-").strip()
    if not safe_folder:
        safe_folder = "matched_faces"

    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}

    zip_buffer = io.BytesIO()
    added = 0

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path_str in path_list:
            file_path = Path(file_path_str)
            if not file_path.exists() or not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed_extensions:
                continue

            # Add to zip under the folder name
            arcname = f"{safe_folder}/{file_path.name}"
            # Handle duplicate filenames
            base = file_path.stem
            ext = file_path.suffix
            counter = 1
            while arcname in [info.filename for info in zf.filelist]:
                arcname = f"{safe_folder}/{base}_{counter}{ext}"
                counter += 1

            zf.write(str(file_path), arcname)
            added += 1

    if added == 0:
        raise HTTPException(status_code=404, detail="No valid images found at the provided paths.")

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_folder}.zip"'},
    )
