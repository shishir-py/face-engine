"""Face detection and embedding extraction using InsightFace (ArcFace)."""

import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# insightface.utils.face_align calls skimage's SimilarityTransform.estimate(), which is
# deprecated in scikit-image 0.26. It fires on every aligned face; nothing we can fix
# from here, so keep it out of the logs.
warnings.filterwarnings(
    "ignore",
    message=r"`estimate` is deprecated",
    category=FutureWarning,
)

from insightface.app import FaceAnalysis

from app.config import get_config

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detects faces and extracts 512-dim ArcFace embeddings using InsightFace."""

    def __init__(self):
        cfg = get_config()
        self.model_pack = cfg.face_detection.model_pack
        self.det_threshold = cfg.face_detection.detection_threshold
        self.min_face_size = cfg.face_detection.min_face_size
        self.max_faces = cfg.face_detection.max_faces_per_image
        self.embedding_dim = cfg.face_recognition.embedding_dim
        self._app: Optional[FaceAnalysis] = None

    def initialize(self):
        """Load InsightFace models. Call once at startup."""
        if self._app is not None:
            return

        logger.info("Loading InsightFace model pack: %s", self.model_pack)
        self._app = FaceAnalysis(
            name=self.model_pack,
            providers=["CPUExecutionProvider"],
        )
        # det_size controls detection resolution — (640, 640) is a good balance
        self._app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=self.det_threshold)
        logger.info("InsightFace models loaded successfully.")

    @property
    def app(self) -> FaceAnalysis:
        if self._app is None:
            self.initialize()
        return self._app

    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect faces in an image.

        Args:
            image: BGR numpy array (OpenCV format).

        Returns:
            List of InsightFace Face objects, each with:
              - .bbox: [x1, y1, x2, y2]
              - .det_score: detection confidence
              - .embedding: 512-dim numpy array (L2-normalized by ArcFace)
        """
        faces = self.app.get(image)

        # Filter by minimum face size
        filtered = []
        for face in faces:
            bbox = face.bbox.astype(int)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w >= self.min_face_size and h >= self.min_face_size:
                filtered.append(face)

        # Sort by detection score (highest first)
        filtered.sort(key=lambda f: f.det_score, reverse=True)

        # Limit if configured
        if self.max_faces > 0:
            filtered = filtered[: self.max_faces]

        return filtered

    def process_image_file(self, image_path: str | Path) -> Tuple[Optional[np.ndarray], list, Optional[str]]:
        """
        Load an image file and detect faces.

        Returns:
            (image_array, faces_list, error_message)
            If error_message is not None, the image could not be processed.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return None, [], f"File not found: {image_path}"

        try:
            # Read image with OpenCV
            img = cv2.imread(str(image_path))
            if img is None:
                return None, [], f"Could not read image (corrupted or unsupported format): {image_path.name}"

            h, w = img.shape[:2]
            if h < 10 or w < 10:
                return None, [], f"Image too small ({w}x{h}): {image_path.name}"

        except Exception as e:
            return None, [], f"Error reading image: {str(e)}"

        try:
            faces = self.detect_faces(img)
        except Exception as e:
            return img, [], f"Face detection error: {str(e)}"

        return img, faces, None

    def process_image_bytes(self, image_bytes: bytes) -> Tuple[Optional[np.ndarray], list, Optional[str]]:
        """
        Process image from raw bytes (for API uploads).

        Returns:
            (image_array, faces_list, error_message)
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None, [], "Could not decode image (corrupted or unsupported format)"

            h, w = img.shape[:2]
            if h < 10 or w < 10:
                return None, [], f"Image too small ({w}x{h})"

        except Exception as e:
            return None, [], f"Error decoding image: {str(e)}"

        try:
            faces = self.detect_faces(img)
        except Exception as e:
            return img, [], f"Face detection error: {str(e)}"

        return img, faces, None

    def extract_embedding(self, face) -> np.ndarray:
        """
        Get the embedding vector from a detected face.
        InsightFace ArcFace produces 512-dim L2-normalized embeddings.
        """
        embedding = face.embedding
        # Ensure it's normalized (ArcFace should already do this)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def get_face_info(self, face) -> dict:
        """Extract structured info from an InsightFace face object."""
        bbox = face.bbox.astype(float)
        return {
            "bbox": {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
                "width": float(bbox[2] - bbox[0]),
                "height": float(bbox[3] - bbox[1]),
            },
            "detection_score": float(face.det_score),
            "embedding_norm": float(np.linalg.norm(face.embedding)),
        }


# Singleton instance
_detector: Optional[FaceDetector] = None


def get_detector() -> FaceDetector:
    """Get the global FaceDetector instance."""
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector
