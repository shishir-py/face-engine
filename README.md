<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/InsightFace-ArcFace-FF6F00?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-4285F4?style=for-the-badge&logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">FaceMatch AI</h1>
<h3 align="center">Intelligent Face Search & Recognition Platform</h3>

<p align="center">
  Upload a photo, detect every face, instantly search thousands of enrolled images,<br/>
  and get ranked matches with confidence scores — all from a beautiful web UI.
</p>

---

## Overview

FaceMatch AI is a self-hosted face recognition platform that detects faces in uploaded images, converts them into 512-dimensional ArcFace embeddings, and searches them against a FAISS-powered vector index of enrolled identities. It provides a polished web UI with real-time webcam scanning, image preview, configurable match thresholds, and batch download of results.

### Key Features

- **High-Accuracy Face Detection** — Powered by InsightFace's `buffalo_l` model pack with configurable detection thresholds
- **ArcFace Embeddings** — 512-dimensional face embeddings with cosine similarity search
- **FAISS Vector Search** — Sub-millisecond search across thousands of enrolled faces
- **Live Camera Scan** — Real-time webcam face matching with auto-capture every 3 seconds
- **Directory Browser** — Visual folder picker to enroll image directories from your filesystem
- **Image Preview** — Click any match result to see a full-size preview with metadata
- **Match Threshold Filter** — Set a minimum similarity percentage to show only strong matches
- **Batch Download** — Download all matching images above your threshold as a ZIP file
- **Flexible Enrollment** — Structured (person-per-folder) or flat (all images loose) layouts
- **Source Directory Management** — Add, track, and re-scan multiple image directories
- **REST API** — Full OpenAPI/Swagger documentation at `/docs`
- **Docker Ready** — One-command deployment with Docker Compose
- **Cross-Platform** — Runs on macOS, Linux, and Windows

---

## Architecture

```
+---------------------------------------------------------+
|                    Web UI (localhost:8000)                |
|  Upload · Search · Camera · Directory Browser · Download |
+----------------+----------------------------+-----------+
                 |                            |
         +-------v--------+          +--------v--------+
         |  /face/search   |          |  /enrollment/*  |
         |  Search API     |          |  Enrollment API |
         +-------+--------+          +--------+--------+
                 |                            |
         +-------v----------------------------v-------+
         |          InsightFace Detector               |
         |     (buffalo_l / ArcFace / SCRFD)           |
         +-------+----------------------------+-------+
                 |                            |
         +-------v--------+          +--------v--------+
         |  FAISS Index    |          |  SQLite Database |
         |  (IndexFlatIP)  |          |  (SQLAlchemy)    |
         +----------------+          +-----------------+
```

---

## Quick Start

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **pip** package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/facematch-ai.git
cd facematch-ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
python run.py
```

Open **http://localhost:8000** in your browser.

### Windows Quick Start

1. Double-click **`setup.bat`** — creates a virtual environment and installs all dependencies
2. Double-click **`start.bat`** — launches the server
3. Open **http://localhost:8000** in your browser

---

## Docker Deployment

```bash
# Build and run with Docker Compose
docker compose up -d
```

To mount your local image directory, edit `docker-compose.yml` and uncomment the volume line:

```yaml
volumes:
  - /path/to/your/images:/app/data/people
```

The server will be available at **http://localhost:8000**.

To stop:

```bash
docker compose down
```

---

## How to Add Image Directories

FaceMatch AI supports three ways to enroll face images:

### Method 1: Browse & Add via Web UI (Recommended)

1. Open **http://localhost:8000** in your browser
2. Scroll to the **"Repository"** section
3. Click the **"Browse"** button to open the directory picker
4. Navigate to the folder containing your images, or type a path directly in the quick-path input
5. Select the directory and choose a layout:
   - **Auto** — Automatically detects the folder structure
   - **Structured** — Each subfolder = one person (best for organized collections)
   - **Flat** — All images at the same level, indexed individually
6. Click **"Select"** — the system scans all images and enrolls faces automatically

### Method 2: Structured Directory (Person-per-Folder)

Organize your images so each person has their own subfolder:

```
my_images/
├── john_doe/
│   ├── front.jpg
│   ├── side.jpg
│   └── casual.png
├── jane_smith/
│   └── headshot.jpg
└── bob_wilson/
    ├── img1.jpg
    ├── img2.jpg
    └── img3.jpg
```

Each directory name becomes the `person_id`. Multiple images per person improve matching accuracy.

Then add the directory via the Web UI Browse button or via API:

```bash
curl -X POST http://localhost:8000/enrollment/sources/add \
  -F "path=/path/to/my_images" \
  -F "layout=structured"
```

### Method 3: Flat Directory (All Images Loose)

If all images are in a single folder without person-based subfolders:

```
photos/
├── photo001.jpg
├── photo002.jpg
├── photo003.png
└── photo004.jpg
```

```bash
curl -X POST http://localhost:8000/enrollment/sources/add \
  -F "path=/path/to/photos" \
  -F "layout=flat"
```

### Method 4: Upload Individual Person Photos

Enroll a specific person by uploading their images directly:

```bash
curl -X POST http://localhost:8000/enrollment/person \
  -F "person_id=john_doe" \
  -F "name=John Doe" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg"
```

### Adding Multiple Source Directories

You can add as many source directories as you want. Each is tracked separately and can be re-scanned independently for new images:

```bash
# Add first directory
curl -X POST http://localhost:8000/enrollment/sources/add \
  -F "path=/Users/me/photos/family" -F "alias=Family Photos"

# Add second directory
curl -X POST http://localhost:8000/enrollment/sources/add \
  -F "path=/Users/me/photos/work" -F "alias=Work ID Photos"

# List all registered sources
curl http://localhost:8000/enrollment/sources

# Re-scan a specific source for new images
curl -X POST http://localhost:8000/enrollment/sources/1/scan
```

### Supported Image Formats

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

---

## Using Face Search

### Upload Search

1. Go to the **Face Search** section in the Web UI
2. Upload an image containing one or more faces
3. Adjust settings:
   - **Top K** — Number of results to return (1–100)
   - **Match Threshold** — Minimum similarity percentage (e.g., 50% to show only strong matches)
4. Click **Search**
5. View ranked results — click any match card for a full-size image preview
6. Click **"Download Matches"** to save all results above your threshold as a ZIP file

### Live Camera Scan

1. Click **"Scan Now"** in the Web UI
2. Allow camera access when prompted
3. The system auto-captures frames every 3 seconds and searches for matching faces in real-time
4. Matching images appear as thumbnails with similarity scores
5. Click **"Capture"** to take a manual snapshot, or **"Stop"** to end the session

> **Important:** Camera access requires `http://localhost:8000` — it won't work from `http://0.0.0.0:8000` or a non-HTTPS remote address.

### Search via API

```bash
curl -X POST http://localhost:8000/face/search \
  -F "image=@query_photo.jpg" \
  -F "top_k=20" \
  -F "mode=image"
```

---

## How Matching Works

1. **Detection** — SCRFD locates faces with bounding boxes and confidence scores
2. **Embedding** — ArcFace produces a 512-dimensional L2-normalized vector per face
3. **Search** — FAISS finds the top-K most similar embeddings by cosine similarity
4. **Aggregation** — Results are grouped by person; each person's score combines their best single-image match (70% weight) with the mean of their top-3 matches (30% weight)
5. **Classification** — Scores are mapped to confidence levels via configurable thresholds

### Default Thresholds

| Score Range | Status |
|-------------|--------|
| >= 0.55 | `high_confidence` |
| 0.40 - 0.54 | `possible_match` |
| 0.30 - 0.39 | `low_confidence` |
| < 0.30 | `no_match` |

> Similarity scores are cosine similarities of ArcFace embeddings, not calibrated probabilities. Tune thresholds for your dataset in `config.yaml`.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/face/search` | Search for faces in an uploaded image |
| `POST` | `/face/download-matches` | Download matched images as ZIP |
| `POST` | `/enrollment/sources/add` | Add and scan an image directory |
| `GET` | `/enrollment/sources` | List registered source directories |
| `POST` | `/enrollment/sources/{id}/scan` | Re-scan a source directory |
| `POST` | `/enrollment/scan` | Scan the default repository |
| `POST` | `/enrollment/person` | Enroll a person with uploaded images |
| `POST` | `/enrollment/reprocess` | Reprocess all images and rebuild index |
| `GET` | `/enrollment/browse?path=` | Browse filesystem directories |
| `GET` | `/admin/health` | System health check with stats |
| `GET` | `/admin/people` | List enrolled people |
| `GET` | `/admin/people/{id}` | Get person details and images |
| `GET` | `/admin/config` | View current configuration |
| `GET` | `/admin/image?path=` | Serve an enrolled image for preview |
| `GET` | `/admin/failed-images` | List images that failed processing |
| `GET` | `/admin/logs/processing` | View enrollment logs |
| `GET` | `/admin/logs/searches` | View search request logs |

Full interactive docs at **http://localhost:8000/docs** (Swagger UI).

---

## Configuration

All settings live in `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000

face_detection:
  model_pack: "buffalo_l"          # buffalo_l (accurate) or buffalo_s (fast)
  detection_threshold: 0.5         # Face detection confidence (0.0-1.0)
  min_face_size: 20                # Minimum face size in pixels

search:
  default_top_k: 10                # Default results to return
  max_top_k: 100                   # Maximum allowed
  index_type: "flat"               # "flat" (exact) or "ivf" (approximate)

thresholds:
  high_confidence: 0.55
  possible_match: 0.40
  low_confidence: 0.30
```

---

## Project Structure

```
facematch-ai/
├── app/
│   ├── __init__.py                # Version
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Configuration loader
│   ├── api/
│   │   ├── search.py              # Face search endpoints
│   │   ├── enrollment.py          # Enrollment & directory browser
│   │   └── admin.py               # Admin, health, image serving
│   ├── models/
│   │   ├── database.py            # SQLAlchemy models
│   │   └── schemas.py             # Pydantic schemas
│   └── services/
│       ├── face_detector.py       # InsightFace wrapper
│       ├── face_index.py          # FAISS index manager
│       ├── enrollment.py          # Enrollment pipeline
│       └── search.py              # Search orchestrator
├── static/
│   └── index.html                 # Web UI (single-file)
├── data/                          # Runtime data (gitignored)
│   ├── people/                    # Enrolled images
│   ├── index/                     # FAISS index files
│   ├── db/                        # SQLite database
│   └── uploads/                   # Temporary uploads
├── tests/
│   └── test_e2e.py
├── config.yaml                    # System configuration
├── requirements.txt               # Python dependencies
├── run.py                         # Server entry point
├── Dockerfile                     # Multi-stage Docker build
├── docker-compose.yml             # Docker Compose config
├── setup.bat                      # Windows setup script
├── start.bat                      # Windows launch script
├── .gitignore
└── .dockerignore
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI + Uvicorn |
| Face Detection | InsightFace (SCRFD + ArcFace) |
| Vector Search | FAISS (IndexFlatIP, cosine similarity) |
| Embeddings | ArcFace 512-dimensional |
| Database | SQLite + SQLAlchemy |
| Image Processing | OpenCV + Pillow |
| Runtime | ONNX Runtime |
| Frontend | Vanilla HTML/CSS/JS |
| Containerization | Docker + Docker Compose |

---

## Troubleshooting

**Camera not working in "Scan Now"?**
Use `http://localhost:8000` instead of `http://0.0.0.0:8000`. Browsers require a secure origin for camera access.

**"Permission denied" when browsing directories?**
Some OS directories are restricted. Use the quick-path input to type the exact path, or navigate from an accessible location like `/Users` on macOS.

**No faces detected in my images?**
Ensure faces are at least 20px in width/height. Lower `detection_threshold` in `config.yaml` for more sensitivity. AI-generated or heavily stylized images may not be detected.

**How do I re-scan a directory for new images?**
Go to `/enrollment/sources` to see all registered sources, then call `/enrollment/sources/{id}/scan` or use the Web UI's re-scan button.

---

## Accuracy & Limitations

- Face recognition accuracy depends on image quality, lighting, angle, and occlusion
- Works best with clear, front-facing photos
- InsightFace `buffalo_l` achieves 99.83% on the LFW benchmark
- AI-generated or heavily stylized images may not be detected
- No face-recognition system guarantees 100% accuracy — always verify critical matches

---

## License

This project is licensed under the MIT License.

---

<p align="center">
  Built with InsightFace &middot; FAISS &middot; FastAPI
</p>
