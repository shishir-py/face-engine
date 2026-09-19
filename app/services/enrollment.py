"""Enrollment pipeline — process image directories and build the face index.

Supports two modes:
  1. Structured: person-per-folder (data/people/person_001/*.jpg)
  2. Flat/unstructured: all images in one directory, each image indexed
     independently so you can search "find all images containing this face"
"""

import datetime
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import get_config
from app.models.database import (
    Face, Image, Person, ProcessingLog, SourceDirectory, get_session,
)
from app.services.face_detector import get_detector
from app.services.face_index import get_face_index

logger = logging.getLogger(__name__)


class EnrollmentPipeline:
    """Processes image directories and enrolls faces into the search index."""

    def __init__(self):
        self.cfg = get_config()
        self.detector = get_detector()
        self.index = get_face_index()
        self.supported_ext = set(self.cfg.enrollment.supported_extensions)

    # ─── Scan a single directory (auto-detect layout) ──────────

    def scan_directory(self, dir_path: str | Path, layout: str = "auto", alias: str | None = None) -> dict:
        """
        Scan a directory and enroll all faces found.

        Args:
            dir_path: Path to the image directory.
            layout: "structured" (person-per-folder), "flat" (all images loose),
                    or "auto" (detect based on directory structure).
            alias: Optional friendly name for this source.

        Returns:
            Enrollment statistics dict.
        """
        dir_path = Path(dir_path).resolve()
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        # Auto-detect layout
        if layout == "auto":
            layout = self._detect_layout(dir_path)
            logger.info("Auto-detected layout for %s: %s", dir_path, layout)

        # Register source directory
        session = get_session()
        source = session.query(SourceDirectory).filter_by(path=str(dir_path)).first()
        if source is None:
            source = SourceDirectory(
                path=str(dir_path),
                alias=alias or dir_path.name,
                layout=layout,
            )
            session.add(source)
            session.commit()

        session.close()

        if layout == "structured":
            return self._scan_structured(dir_path)
        else:
            return self._scan_flat(dir_path)

    def _detect_layout(self, dir_path: Path) -> str:
        """
        Detect whether a directory is structured (person-per-folder) or flat.

        Heuristic: if the directory contains subdirectories that themselves
        contain image files, it's structured. Otherwise it's flat.
        """
        has_image_subdirs = False
        has_loose_images = False

        for item in dir_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                # Check if this subdir has images
                for f in item.iterdir():
                    if f.is_file() and f.suffix.lower() in self.supported_ext:
                        has_image_subdirs = True
                        break
            elif item.is_file() and item.suffix.lower() in self.supported_ext:
                has_loose_images = True

        # If there are subdirs with images, treat as structured
        # (even if there are also loose images at root — subdirs take priority)
        if has_image_subdirs:
            return "structured"
        return "flat"

    # ─── Structured scan (person-per-folder) ───────────────────

    def _scan_structured(self, repo_path: Path) -> dict:
        """Scan a structured repository where each subfolder = one person."""
        return self.scan_repository(repo_path)

    def scan_repository(self, repo_path: str | Path | None = None) -> dict:
        """
        Scan the image repository with person-per-folder structure.

        Directory structure:
            repo_path/
            ├── person_001/
            │   ├── img1.jpg
            │   └── img2.jpg
            └── person_002/
                └── img1.jpg
        """
        if repo_path is None:
            repo_path = Path(self.cfg.paths.image_repository)
        else:
            repo_path = Path(repo_path)

        if not repo_path.exists():
            raise FileNotFoundError(f"Image repository not found: {repo_path}")

        start_time = time.time()
        session = get_session()
        log_entry = ProcessingLog(
            run_type="enrollment",
            model_version=self.cfg.face_detection.model_pack,
            source_directory=str(repo_path),
        )
        session.add(log_entry)
        session.commit()

        stats = {
            "total_images": 0,
            "images_processed": 0,
            "faces_detected": 0,
            "images_skipped": 0,
            "images_failed": 0,
            "people_enrolled": 0,
        }
        failed_images = []

        person_dirs = sorted([
            d for d in repo_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        logger.info("Found %d person directories in %s", len(person_dirs), repo_path)

        all_embeddings = []
        all_face_ids = []

        for person_dir in person_dirs:
            person_id = person_dir.name

            person = session.query(Person).filter_by(person_id=person_id).first()
            if person is None:
                person = Person(person_id=person_id, name=person_id, is_labeled=True)
                session.add(person)
                session.flush()
                stats["people_enrolled"] += 1

            image_files = sorted([
                f for f in person_dir.iterdir()
                if f.is_file() and f.suffix.lower() in self.supported_ext
            ])

            for img_path in image_files:
                stats["total_images"] += 1
                rel_path = str(img_path.relative_to(repo_path))

                if self.cfg.enrollment.skip_existing:
                    existing = session.query(Image).filter_by(file_path=rel_path).first()
                    if existing and existing.processed:
                        stats["images_skipped"] += 1
                        continue

                embeddings, face_ids, error = self._process_single_image(
                    session, person, img_path, rel_path, source_dir=str(repo_path)
                )

                if error:
                    stats["images_failed"] += 1
                    failed_images.append({"file_path": rel_path, "reason": error})
                else:
                    stats["images_processed"] += 1
                    stats["faces_detected"] += len(embeddings)
                    all_embeddings.extend(embeddings)
                    all_face_ids.extend(face_ids)

            person.image_count = session.query(Image).filter_by(person_id_fk=person.id, processed=True).count()
            person.face_count = session.query(Face).filter_by(person_id=person_id).count()

        if all_embeddings:
            emb_array = np.vstack(all_embeddings).astype(np.float32)
            self.index.add_embeddings(emb_array, all_face_ids)
            self.index.save()

        elapsed = time.time() - start_time
        log_entry.finished_at = datetime.datetime.utcnow()
        log_entry.total_images = stats["total_images"]
        log_entry.images_processed = stats["images_processed"]
        log_entry.faces_detected = stats["faces_detected"]
        log_entry.images_skipped = stats["images_skipped"]
        log_entry.images_failed = stats["images_failed"]
        session.commit()
        session.close()

        stats["processing_time_seconds"] = round(elapsed, 2)

        logger.info(
            "Enrollment complete: %d images, %d faces, %d failed, %.1fs",
            stats["images_processed"], stats["faces_detected"],
            stats["images_failed"], elapsed,
        )

        return {"stats": stats, "failed_images": failed_images}

    # ─── Flat/unstructured scan ────────────────────────────────

    def _scan_flat(self, dir_path: Path) -> dict:
        """
        Scan a flat directory (no person subfolders).

        Each image gets its own auto-generated person_id based on filename.
        This allows searching "find all images containing this face" without
        pre-labeling who is in each image.

        Also walks subdirectories (for mixed/nested structures).
        """
        start_time = time.time()
        session = get_session()
        log_entry = ProcessingLog(
            run_type="scan_directory",
            model_version=self.cfg.face_detection.model_pack,
            source_directory=str(dir_path),
        )
        session.add(log_entry)
        session.commit()

        stats = {
            "total_images": 0,
            "images_processed": 0,
            "faces_detected": 0,
            "images_skipped": 0,
            "images_failed": 0,
            "people_enrolled": 0,
        }
        failed_images = []
        all_embeddings = []
        all_face_ids = []

        # Walk all images in this directory and subdirectories
        image_files = sorted(self._find_all_images(dir_path))
        logger.info("Found %d images in %s (flat scan)", len(image_files), dir_path)

        for img_path in image_files:
            stats["total_images"] += 1

            # Use the full path relative to the source dir as a unique key
            rel_path = str(img_path.relative_to(dir_path))
            # Unique file key uses absolute path to avoid collisions between sources
            file_key = f"{dir_path.name}/{rel_path}"

            if self.cfg.enrollment.skip_existing:
                existing = session.query(Image).filter_by(file_path=file_key).first()
                if existing and existing.processed:
                    stats["images_skipped"] += 1
                    continue

            # For flat directories, each image file becomes its own "person"
            # (the person_id is the filename stem — images with the same name
            #  in different subdirs get separate entries)
            person_id = f"img_{dir_path.name}_{img_path.stem}"

            # If the image is inside a subfolder, use the subfolder name as person_id
            # (heuristic: maybe the user partially organized by person)
            parent_rel = img_path.parent.relative_to(dir_path)
            if str(parent_rel) != ".":
                # Image is inside a subfolder — use folder name as person identity
                person_id = str(parent_rel).replace("/", "_").replace("\\", "_")

            person = session.query(Person).filter_by(person_id=person_id).first()
            if person is None:
                person = Person(
                    person_id=person_id,
                    name=person_id,
                    is_labeled=False,  # auto-generated, not user-labeled
                )
                session.add(person)
                session.flush()
                stats["people_enrolled"] += 1

            embeddings, face_ids, error = self._process_single_image(
                session, person, img_path, file_key, source_dir=str(dir_path)
            )

            if error:
                stats["images_failed"] += 1
                failed_images.append({"file_path": file_key, "reason": error})
            else:
                stats["images_processed"] += 1
                stats["faces_detected"] += len(embeddings)
                all_embeddings.extend(embeddings)
                all_face_ids.extend(face_ids)

            # Update person counts periodically
            person.image_count = session.query(Image).filter_by(person_id_fk=person.id, processed=True).count()
            person.face_count = session.query(Face).filter_by(person_id=person_id).count()

        if all_embeddings:
            emb_array = np.vstack(all_embeddings).astype(np.float32)
            self.index.add_embeddings(emb_array, all_face_ids)
            self.index.save()

        elapsed = time.time() - start_time
        log_entry.finished_at = datetime.datetime.utcnow()
        log_entry.total_images = stats["total_images"]
        log_entry.images_processed = stats["images_processed"]
        log_entry.faces_detected = stats["faces_detected"]
        log_entry.images_skipped = stats["images_skipped"]
        log_entry.images_failed = stats["images_failed"]

        # Update source directory record
        source = session.query(SourceDirectory).filter_by(path=str(dir_path)).first()
        if source:
            source.last_scanned = datetime.datetime.utcnow()
            source.image_count = stats["images_processed"] + stats["images_skipped"]
            source.face_count = stats["faces_detected"]

        session.commit()
        session.close()

        stats["processing_time_seconds"] = round(elapsed, 2)
        logger.info(
            "Flat scan complete: %d images, %d faces, %d failed, %.1fs",
            stats["images_processed"], stats["faces_detected"],
            stats["images_failed"], elapsed,
        )

        return {"stats": stats, "failed_images": failed_images}

    def _find_all_images(self, dir_path: Path) -> List[Path]:
        """Recursively find all supported image files in a directory."""
        images = []
        for item in sorted(dir_path.rglob("*")):
            if item.is_file() and item.suffix.lower() in self.supported_ext:
                if not any(part.startswith(".") for part in item.parts):
                    images.append(item)
        return images

    # ─── Single person enrollment ──────────────────────────────

    def enroll_person(
        self,
        person_id: str,
        image_paths: List[str | Path],
        name: Optional[str] = None,
    ) -> dict:
        """Enroll a single person with their images."""
        session = get_session()

        person = session.query(Person).filter_by(person_id=person_id).first()
        if person is None:
            person = Person(person_id=person_id, name=name or person_id, is_labeled=True)
            session.add(person)
            session.flush()

        all_embeddings = []
        all_face_ids = []
        failed = []

        for img_path in image_paths:
            img_path = Path(img_path)
            rel_path = str(img_path)

            embeddings, face_ids, error = self._process_single_image(
                session, person, img_path, rel_path
            )

            if error:
                failed.append({"file_path": str(img_path), "reason": error})
            else:
                all_embeddings.extend(embeddings)
                all_face_ids.extend(face_ids)

        if all_embeddings:
            emb_array = np.vstack(all_embeddings).astype(np.float32)
            self.index.add_embeddings(emb_array, all_face_ids)
            self.index.save()

        person.image_count = session.query(Image).filter_by(person_id_fk=person.id, processed=True).count()
        person.face_count = session.query(Face).filter_by(person_id=person_id).count()
        session.commit()
        session.close()

        return {
            "person_id": person_id,
            "images_processed": len(image_paths) - len(failed),
            "faces_detected": len(all_embeddings),
            "failed_images": failed,
        }

    # ─── Internal: process a single image ──────────────────────

    def _process_single_image(
        self, session, person: Person, img_path: Path, rel_path: str,
        source_dir: str | None = None,
    ) -> Tuple[List[np.ndarray], List[int], Optional[str]]:
        """
        Process a single image: detect faces, extract embeddings, store in DB.
        Returns: (list_of_embeddings, list_of_face_db_ids, error_or_None)
        """
        img_array, faces, error = self.detector.process_image_file(img_path)

        if error:
            img_record = session.query(Image).filter_by(file_path=rel_path).first()
            if img_record is None:
                img_record = Image(
                    person_id_fk=person.id,
                    file_path=rel_path,
                    file_name=img_path.name,
                    absolute_path=str(img_path.resolve()),
                    source_dir=source_dir,
                    file_size=img_path.stat().st_size if img_path.exists() else None,
                    processed=False,
                    error=error,
                )
                session.add(img_record)
            else:
                img_record.error = error
                img_record.processed = False
            session.flush()
            return [], [], error

        if len(faces) == 0:
            error_msg = "No face detected"
            img_record = session.query(Image).filter_by(file_path=rel_path).first()
            if img_record is None:
                img_record = Image(
                    person_id_fk=person.id,
                    file_path=rel_path,
                    file_name=img_path.name,
                    absolute_path=str(img_path.resolve()),
                    source_dir=source_dir,
                    file_size=img_path.stat().st_size,
                    width=img_array.shape[1] if img_array is not None else None,
                    height=img_array.shape[0] if img_array is not None else None,
                    processed=False,
                    error=error_msg,
                )
                session.add(img_record)
            else:
                img_record.error = error_msg
                img_record.processed = False
            session.flush()
            return [], [], error_msg

        img_record = session.query(Image).filter_by(file_path=rel_path).first()
        if img_record is None:
            img_record = Image(
                person_id_fk=person.id,
                file_path=rel_path,
                file_name=img_path.name,
                absolute_path=str(img_path.resolve()),
                source_dir=source_dir,
                file_size=img_path.stat().st_size,
                width=img_array.shape[1],
                height=img_array.shape[0],
                processed=True,
                processed_at=datetime.datetime.utcnow(),
                face_count=len(faces),
                error=None,
            )
            session.add(img_record)
        else:
            img_record.processed = True
            img_record.processed_at = datetime.datetime.utcnow()
            img_record.face_count = len(faces)
            img_record.error = None
            img_record.absolute_path = str(img_path.resolve())
            img_record.source_dir = source_dir

        session.flush()

        embeddings = []
        face_ids = []
        current_max = session.query(Face).count()

        for i, face in enumerate(faces):
            embedding = self.detector.extract_embedding(face)
            face_info = self.detector.get_face_info(face)
            faiss_idx = current_max + len(embeddings)

            face_record = Face(
                image_id=img_record.id,
                person_id=person.person_id,
                faiss_idx=faiss_idx,
                bbox_x1=face_info["bbox"]["x1"],
                bbox_y1=face_info["bbox"]["y1"],
                bbox_x2=face_info["bbox"]["x2"],
                bbox_y2=face_info["bbox"]["y2"],
                detection_score=face_info["detection_score"],
                embedding_norm=face_info["embedding_norm"],
            )
            session.add(face_record)
            session.flush()

            embeddings.append(embedding)
            face_ids.append(face_record.id)

        return embeddings, face_ids, None

    # ─── Source management ─────────────────────────────────────

    def list_sources(self) -> List[dict]:
        """List all registered source directories."""
        session = get_session()
        sources = session.query(SourceDirectory).order_by(SourceDirectory.added_at).all()
        result = [
            {
                "id": s.id,
                "path": s.path,
                "alias": s.alias,
                "layout": s.layout,
                "image_count": s.image_count,
                "face_count": s.face_count,
                "last_scanned": s.last_scanned.isoformat() if s.last_scanned else None,
            }
            for s in sources
        ]
        session.close()
        return result

    def reprocess_all(self) -> dict:
        """Reprocess all images and rebuild the index."""
        session = get_session()
        session.query(Face).delete()
        session.query(Image).update({Image.processed: False, Image.error: None})
        session.commit()

        # Get all source directories to re-scan
        sources = session.query(SourceDirectory).all()
        session.close()

        self.index.initialize(rebuild=True)

        all_stats = {
            "total_images": 0, "images_processed": 0, "faces_detected": 0,
            "images_skipped": 0, "images_failed": 0, "people_enrolled": 0,
        }
        all_failed = []

        # Re-scan each registered source
        for source in sources:
            result = self.scan_directory(source.path, layout=source.layout)
            for k in all_stats:
                all_stats[k] += result["stats"].get(k, 0)
            all_failed.extend(result["failed_images"])

        # Also re-scan the default structured repo if it exists
        default_repo = Path(self.cfg.paths.image_repository)
        if default_repo.exists() and any(default_repo.iterdir()):
            already_scanned = {s.path for s in sources}
            if str(default_repo.resolve()) not in already_scanned:
                result = self.scan_repository(default_repo)
                for k in all_stats:
                    all_stats[k] += result["stats"].get(k, 0)
                all_failed.extend(result["failed_images"])

        return {"stats": all_stats, "failed_images": all_failed}
