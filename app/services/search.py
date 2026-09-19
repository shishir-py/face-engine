"""Face search service with person-level aggregation and image-level matching.

Supports three search modes:
  - "person": Aggregate results by person identity (default)
  - "image": Return individual matching images with file paths
  - "both": Return both person-level and image-level results
"""

import logging
import time
import uuid
from collections import defaultdict
from typing import List, Optional

import numpy as np

from app.config import get_config
from app.models.database import Face, Image, Person, SearchLog, get_session
from app.models.schemas import (
    BoundingBox,
    FaceSearchResult,
    ImageMatch,
    PersonMatch,
    SearchResponse,
)
from app.services.face_detector import get_detector
from app.services.face_index import get_face_index

logger = logging.getLogger(__name__)


class FaceSearchService:
    """
    Searches for matching people/images given a query image.

    Flow:
      1. Detect faces in query image
      2. For each face, search FAISS index for top-K similar embeddings
      3. Map results to people and/or images depending on search mode
      4. Rank candidates and apply confidence thresholds
    """

    def __init__(self):
        self.cfg = get_config()
        self.detector = get_detector()
        self.index = get_face_index()
        self.thresholds = self.cfg.thresholds

    def search_image_bytes(
        self,
        image_bytes: bytes,
        top_k: Optional[int] = None,
        threshold_override: Optional[dict] = None,
        search_mode: str = "person",
    ) -> SearchResponse:
        """
        Search for matching people/images from uploaded image bytes.

        Args:
            image_bytes: Raw image file bytes.
            top_k: Number of candidates per face (default from config).
            threshold_override: Custom thresholds for this request.
            search_mode: "person" (default), "image", or "both".

        Returns:
            SearchResponse with ranked results for each detected face.
        """
        request_id = str(uuid.uuid4())[:12]
        start = time.time()

        if search_mode not in ("person", "image", "both"):
            search_mode = "person"

        if top_k is None:
            top_k = self.cfg.search.default_top_k
        top_k = min(top_k, self.cfg.search.max_top_k)

        thresholds = threshold_override or {
            "high_confidence": self.thresholds.high_confidence,
            "possible_match": self.thresholds.possible_match,
            "low_confidence": self.thresholds.low_confidence,
        }

        # Detect faces
        img_array, faces, error = self.detector.process_image_bytes(image_bytes)

        if error:
            elapsed = (time.time() - start) * 1000
            return SearchResponse(
                request_id=request_id,
                search_mode=search_mode,
                faces_detected=0,
                processing_time_ms=round(elapsed, 2),
                threshold_config=thresholds,
                results=[],
            )

        # Search for each detected face
        results: List[FaceSearchResult] = []

        for i, face in enumerate(faces):
            embedding = self.detector.extract_embedding(face)
            face_info = self.detector.get_face_info(face)

            bbox = BoundingBox(**face_info["bbox"])

            # FAISS search — over-fetch for aggregation
            raw_matches = self.index.search(embedding, top_k=top_k * 3)

            candidates = []
            best_match = None
            image_matches = []

            # Person-level results
            if search_mode in ("person", "both"):
                candidates = self._aggregate_by_person(raw_matches, thresholds, top_k)
                best_match = (
                    candidates[0]
                    if candidates and candidates[0].match_status != "no_match"
                    else None
                )

            # Image-level results
            if search_mode in ("image", "both"):
                image_matches = self._get_image_matches(raw_matches, thresholds, top_k)

            results.append(FaceSearchResult(
                face_index=i,
                bounding_box=bbox,
                detection_confidence=face_info["detection_score"],
                candidates=candidates,
                best_match=best_match,
                image_matches=image_matches,
            ))

        elapsed = (time.time() - start) * 1000

        # Log the search
        self._log_search(
            request_id, len(faces), top_k,
            sum(len(r.candidates) + len(r.image_matches) for r in results),
            elapsed, search_mode,
        )

        return SearchResponse(
            request_id=request_id,
            search_mode=search_mode,
            faces_detected=len(faces),
            processing_time_ms=round(elapsed, 2),
            threshold_config=thresholds,
            results=results,
        )

    # ─── Image-level matching ─────────────────────────────────

    def _get_image_matches(
        self,
        raw_matches: list,
        thresholds: dict,
        top_k: int,
    ) -> List[ImageMatch]:
        """
        Return individual matching images ranked by similarity.

        Each result includes the file path on disk so the user can
        locate the matching image. Multiple faces from the same image
        are deduplicated — only the best score is kept.
        """
        if not raw_matches:
            return []

        session = get_session()

        face_ids = [face_id for face_id, _ in raw_matches]
        face_records = session.query(Face).filter(Face.id.in_(face_ids)).all()
        face_map = {f.id: f for f in face_records}

        # Deduplicate by image — keep best similarity per image
        image_best: dict = {}  # image_id -> (similarity, face_record)

        for face_id, similarity in raw_matches:
            face = face_map.get(face_id)
            if face is None:
                continue
            img_id = face.image_id
            if img_id not in image_best or similarity > image_best[img_id][0]:
                image_best[img_id] = (similarity, face)

        # Build image match list
        image_ids = list(image_best.keys())
        images = session.query(Image).filter(Image.id.in_(image_ids)).all()
        image_map = {img.id: img for img in images}

        # Preload person info for labeled identities
        person_ids = {img.person_id_fk for img in images}
        persons = session.query(Person).filter(Person.id.in_(person_ids)).all()
        person_map = {p.id: p for p in persons}

        matches = []
        for img_id, (similarity, face_rec) in image_best.items():
            image = image_map.get(img_id)
            if image is None:
                continue

            person = person_map.get(image.person_id_fk)

            matches.append(ImageMatch(
                image_path=image.file_path,
                absolute_path=image.absolute_path,
                similarity=round(similarity, 4),
                source_dir=image.source_dir,
                person_id=person.person_id if person else None,
                person_name=person.name if person else None,
                is_labeled=person.is_labeled if person else False,
            ))

        session.close()

        # Sort by similarity descending
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:top_k]

    # ─── Person-level aggregation ─────────────────────────────

    def _aggregate_by_person(
        self,
        raw_matches: list,
        thresholds: dict,
        top_k: int,
    ) -> List[PersonMatch]:
        """
        Aggregate face-level matches into person-level ranked candidates.

        Strategy:
          - Group matching faces by person_id.
          - For each person, compute an aggregated score:
              score = max_similarity * 0.7 + mean_of_top3 * 0.3
            This weights the best single match heavily while rewarding
            consistency across multiple images.
          - Classify match confidence using thresholds.
          - Return top_k ranked results.
        """
        if not raw_matches:
            return []

        session = get_session()

        # Map face DB IDs to person info
        face_ids = [face_id for face_id, _ in raw_matches]
        face_records = session.query(Face).filter(Face.id.in_(face_ids)).all()
        face_map = {f.id: f for f in face_records}

        # Group by person
        person_scores: dict = defaultdict(list)
        person_images: dict = defaultdict(list)

        for face_id, similarity in raw_matches:
            face = face_map.get(face_id)
            if face is None:
                continue
            person_scores[face.person_id].append(similarity)

            # Get image path for this face
            image = session.query(Image).filter_by(id=face.image_id).first()
            if image:
                person_images[face.person_id].append(
                    ImageMatch(
                        image_path=image.file_path,
                        absolute_path=image.absolute_path,
                        similarity=round(similarity, 4),
                        source_dir=image.source_dir,
                    )
                )

        # Calculate person-level scores
        candidates = []
        for person_id, scores in person_scores.items():
            scores_sorted = sorted(scores, reverse=True)
            max_score = scores_sorted[0]
            top3_mean = float(np.mean(scores_sorted[:3]))

            # Weighted aggregation
            agg_score = max_score * 0.7 + top3_mean * 0.3

            # Classify
            match_status = self._classify_match(agg_score, thresholds)

            # Get person name and labeled status
            person = session.query(Person).filter_by(person_id=person_id).first()
            name = person.name if person else None
            is_labeled = person.is_labeled if person else False

            # Top image matches for this person
            top_images = sorted(
                person_images[person_id], key=lambda m: m.similarity, reverse=True
            )[:5]

            candidates.append(PersonMatch(
                rank=0,  # set below
                person_id=person_id,
                name=name,
                similarity=round(agg_score, 4),
                match_status=match_status,
                matching_images=len(scores),
                is_labeled=is_labeled,
                top_image_matches=top_images,
            ))

        session.close()

        # Sort by score and assign ranks
        candidates.sort(key=lambda c: c.similarity, reverse=True)
        for i, c in enumerate(candidates):
            c.rank = i + 1

        return candidates[:top_k]

    def _classify_match(self, score: float, thresholds: dict) -> str:
        """Classify a similarity score into a match status."""
        if score >= thresholds["high_confidence"]:
            return "high_confidence"
        elif score >= thresholds["possible_match"]:
            return "possible_match"
        elif score >= thresholds["low_confidence"]:
            return "low_confidence"
        else:
            return "no_match"

    def _log_search(
        self,
        request_id: str,
        faces_count: int,
        top_k: int,
        total_candidates: int,
        elapsed_ms: float,
        search_mode: str = "person",
    ):
        """Record search in the search log."""
        try:
            session = get_session()
            log = SearchLog(
                request_id=request_id,
                faces_in_query=faces_count,
                top_k=top_k,
                search_mode=search_mode,
                total_candidates=total_candidates,
                processing_time_ms=round(elapsed_ms, 2),
            )
            session.add(log)
            session.commit()
            session.close()
        except Exception as e:
            logger.warning("Failed to log search: %s", e)
