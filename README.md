# FaceMatch AI — Intelligent Face Search & Recognition Platform

A production-ready face recognition system that detects faces, generates ArcFace embeddings, and searches against a repository of enrolled identities. Returns ranked, person-level matches with confidence scores.

## Architecture

```
Image Repository  →  InsightFace (RetinaFace + ArcFace)  →  512-dim Embeddings
                                                                    ↓
                                                              FAISS Index
                                                                    ↓
Query Image  →  Face Detection  →  Embedding  →  Similarity Search  →  Person-Level Ranking  →  API Response
```

**Stack:** Python 3.10+ · FastAPI · InsightFace (buffalo_l) · FAISS · SQLite · vanilla HTML/JS frontend

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Organize your images
#    Place images in data/people/<person_id>/
#    Each subdirectory = one person, can have multiple images
mkdir -p data/people/john_doe
cp ~/photos/john/*.jpg data/people/john_doe/

# 3. Start the server
python run.py

# 4. Open the web UI
#    http://localhost:8000

# 5. Enroll faces (click "Scan Repository" in the Repository tab)
#    Or via API:
curl -X POST http://localhost:8000/enrollment/scan
```

## API Endpoints

### Face Search

```bash
# Upload an image and find matching people
curl -X POST http://localhost:8000/face/search \
  -F "image=@query_photo.jpg" \
  -F "top_k=10"
```

**Response:**
```json
{
  "request_id": "a1b2c3d4",
  "faces_detected": 1,
  "processing_time_ms": 245.3,
  "results": [
    {
      "face_index": 0,
      "bounding_box": {"x1": 120, "y1": 80, "x2": 280, "y2": 300, "width": 160, "height": 220},
      "detection_confidence": 0.98,
      "candidates": [
        {
          "rank": 1,
          "person_id": "john_doe",
          "similarity": 0.72,
          "match_status": "high_confidence",
          "matching_images": 3,
          "top_image_matches": [...]
        }
      ]
    }
  ]
}
```

### Enrollment

```bash
# Scan entire repository
curl -X POST http://localhost:8000/enrollment/scan

# Enroll a single person with images
curl -X POST http://localhost:8000/enrollment/person \
  -F "person_id=jane_smith" \
  -F "name=Jane Smith" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg"

# Reprocess all images (after model change)
curl -X POST http://localhost:8000/enrollment/reprocess
```

### Admin & Monitoring

```bash
# Health check & stats
curl http://localhost:8000/admin/health

# List enrolled people
curl http://localhost:8000/admin/people

# View failed images
curl http://localhost:8000/admin/failed-images

# Processing and search logs
curl http://localhost:8000/admin/logs/processing
curl http://localhost:8000/admin/logs/searches
```

Full API docs: `http://localhost:8000/docs` (Swagger UI)

## Image Repository Structure

```
data/people/
├── person_001/
│   ├── front.jpg
│   ├── side.jpg
│   └── casual.png
├── person_002/
│   └── headshot.jpg
└── person_003/
    ├── img1.jpg
    ├── img2.jpg
    └── img3.jpg
```

Each directory name becomes the `person_id`. Multiple images per person improve matching accuracy.

## How Matching Works

1. **Detection** — RetinaFace locates faces with bounding boxes and confidence scores.
2. **Embedding** — ArcFace produces a 512-dimensional L2-normalized vector per face.
3. **Search** — FAISS finds the top-K most similar embeddings by cosine similarity.
4. **Aggregation** — Results are grouped by person. Each person's score combines their best single-image match (70% weight) with the mean of their top-3 matches (30% weight), rewarding consistency across multiple enrolled images.
5. **Classification** — Scores are mapped to confidence levels via configurable thresholds.

### Default Thresholds (configurable in `config.yaml`)

| Score Range    | Status             |
|----------------|--------------------|
| ≥ 0.55         | `high_confidence`  |
| 0.40 – 0.54    | `possible_match`   |
| 0.30 – 0.39    | `low_confidence`   |
| < 0.30         | `no_match`         |

> **Note:** Similarity scores are cosine similarities of ArcFace embeddings, not calibrated probabilities. Tune thresholds for your dataset using the validation tools.

## Configuration

All settings are in `config.yaml`. Key options:

- **`face_detection.model_pack`** — `buffalo_l` (best accuracy) or `buffalo_s` (faster)
- **`search.index_type`** — `flat` (exact, best for < 50K faces) or `ivf` (approximate, faster for large repos)
- **`thresholds.*`** — Match confidence cutoffs
- **`enrollment.skip_existing`** — Skip already-processed images on re-scan

## Project Structure

```
facematch-ai/
├── app/
│   ├── main.py                  # FastAPI app with lifespan
│   ├── config.py                # YAML config loader
│   ├── api/
│   │   ├── search.py            # POST /face/search
│   │   ├── enrollment.py        # POST /enrollment/*
│   │   └── admin.py             # GET /admin/*
│   ├── models/
│   │   ├── database.py          # SQLite models (Person, Image, Face, logs)
│   │   └── schemas.py           # Pydantic request/response schemas
│   └── services/
│       ├── face_detector.py     # InsightFace wrapper
│       ├── face_index.py        # FAISS index manager
│       ├── enrollment.py        # Repository scanning pipeline
│       └── search.py            # Search with person-level aggregation
├── static/
│   └── index.html               # Web UI
├── data/
│   ├── people/                  # Image repository
│   ├── index/                   # FAISS index files
│   └── db/                      # SQLite database
├── config.yaml
├── requirements.txt
├── run.py
└── tests/
    └── test_e2e.py
```
