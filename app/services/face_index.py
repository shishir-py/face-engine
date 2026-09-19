"""FAISS vector index manager for face embeddings."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from app.config import get_config

logger = logging.getLogger(__name__)


class FaceIndex:
    """
    Manages a FAISS index for face embedding storage and similarity search.

    Supports two index types:
    - "flat": Exact search (IndexFlatIP). Best accuracy, O(n) search.
    - "ivf": Approximate search (IndexIVFFlat). Faster for large repos.

    Uses inner product (cosine similarity) since embeddings are L2-normalized.
    """

    INDEX_FILE = "face_index.faiss"
    IDS_FILE = "face_ids.npy"

    def __init__(self):
        cfg = get_config()
        self.embedding_dim = cfg.face_recognition.embedding_dim
        self.index_type = cfg.search.index_type
        self.ivf_clusters = cfg.search.ivf_clusters
        self.ivf_nprobe = cfg.search.ivf_nprobe
        self.index_dir = Path(cfg.paths.faiss_index)
        self._index: Optional[faiss.Index] = None
        # Maps FAISS internal index → our face DB ID
        self._id_map: List[int] = []

    @property
    def size(self) -> int:
        """Number of embeddings in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def initialize(self, rebuild: bool = False):
        """Load existing index from disk or create a new one."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir / self.INDEX_FILE
        ids_path = self.index_dir / self.IDS_FILE

        if not rebuild and index_path.exists() and ids_path.exists():
            try:
                self._index = faiss.read_index(str(index_path))
                self._id_map = np.load(str(ids_path)).tolist()
                logger.info("Loaded FAISS index with %d embeddings.", self._index.ntotal)
                return
            except Exception as e:
                logger.warning("Failed to load existing index: %s. Rebuilding.", e)

        self._create_empty_index()
        logger.info("Created new empty FAISS index (type=%s, dim=%d).", self.index_type, self.embedding_dim)

    def _create_empty_index(self):
        """Create an empty FAISS index."""
        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self._index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, self.ivf_clusters, faiss.METRIC_INNER_PRODUCT
            )
            self._index.nprobe = self.ivf_nprobe
        else:
            # Flat index — exact search with inner product (cosine similarity for normalized vectors)
            self._index = faiss.IndexFlatIP(self.embedding_dim)

        self._id_map = []

    def add_embeddings(self, embeddings: np.ndarray, face_db_ids: List[int]):
        """
        Add embeddings to the index.

        Args:
            embeddings: (N, 512) float32 array of L2-normalized embeddings.
            face_db_ids: List of Face.id values from the database.
        """
        if self._index is None:
            raise RuntimeError("Index not initialized.")

        if len(embeddings) == 0:
            return

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # For IVF index, must be trained before adding
        if isinstance(self._index, faiss.IndexIVFFlat) and not self._index.is_trained:
            # Need at least ivf_clusters vectors to train
            if len(embeddings) >= self.ivf_clusters:
                logger.info("Training IVF index with %d vectors...", len(embeddings))
                self._index.train(embeddings)
            else:
                # Fall back to flat index if not enough data
                logger.warning(
                    "Not enough vectors (%d) to train IVF index (need %d). Using flat index.",
                    len(embeddings),
                    self.ivf_clusters,
                )
                self._index = faiss.IndexFlatIP(self.embedding_dim)

        self._index.add(embeddings)
        self._id_map.extend(face_db_ids)

        logger.info("Added %d embeddings to index (total: %d).", len(embeddings), self._index.ntotal)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for the most similar faces.

        Args:
            query_embedding: (512,) float32 L2-normalized embedding.
            top_k: Number of results to return.

        Returns:
            List of (face_db_id, similarity_score) tuples, sorted by score descending.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        actual_k = min(top_k, self._index.ntotal)

        scores, indices = self._index.search(query, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            if idx < len(self._id_map):
                face_db_id = self._id_map[idx]
                results.append((face_db_id, float(score)))

        return results

    def save(self):
        """Persist the index to disk."""
        if self._index is None:
            return

        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir / self.INDEX_FILE
        ids_path = self.index_dir / self.IDS_FILE

        faiss.write_index(self._index, str(index_path))
        np.save(str(ids_path), np.array(self._id_map, dtype=np.int64))

        logger.info("Saved FAISS index (%d embeddings) to %s.", self._index.ntotal, self.index_dir)

    def rebuild(self, embeddings: np.ndarray, face_db_ids: List[int]):
        """
        Rebuild the entire index from scratch.
        Used when the model changes or a full reindex is needed.
        """
        self._create_empty_index()
        if len(embeddings) > 0:
            self.add_embeddings(embeddings, face_db_ids)
        self.save()
        logger.info("Index rebuilt with %d embeddings.", len(embeddings))

    def remove_by_face_ids(self, face_db_ids_to_remove: set):
        """
        Remove specific face IDs from the index.
        Since FAISS flat index doesn't support removal efficiently,
        we rebuild from the remaining vectors.

        Note: For large-scale production, use IndexIDMap2 for O(1) removal.
        """
        if self._index is None or self._index.ntotal == 0:
            return

        # Reconstruct all vectors and filter
        keep_indices = []
        keep_ids = []
        for i, fid in enumerate(self._id_map):
            if fid not in face_db_ids_to_remove:
                keep_indices.append(i)
                keep_ids.append(fid)

        if len(keep_indices) == len(self._id_map):
            return  # Nothing to remove

        # Reconstruct kept embeddings
        all_embeddings = self._reconstruct_all()
        kept_embeddings = all_embeddings[keep_indices] if len(keep_indices) > 0 else np.empty((0, self.embedding_dim), dtype=np.float32)

        self.rebuild(kept_embeddings, keep_ids)

    def _reconstruct_all(self) -> np.ndarray:
        """Reconstruct all embeddings from the index."""
        n = self._index.ntotal
        if n == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        embeddings = np.empty((n, self.embedding_dim), dtype=np.float32)
        for i in range(n):
            embeddings[i] = self._index.reconstruct(i)
        return embeddings


# Singleton
_face_index: Optional[FaceIndex] = None


def get_face_index() -> FaceIndex:
    """Get the global FaceIndex instance."""
    global _face_index
    if _face_index is None:
        _face_index = FaceIndex()
    return _face_index
