"""
Face Authentication Microservice
=================================
Reusable face-based authentication API you can call from ANY project
(Django, another FastAPI app, React frontend, mobile app, etc).

Flow:
  1. POST /enroll   -> user uploads a clear face photo once, embedding is stored
  2. POST /verify    -> user uploads a photo to log in, we compare embeddings
  3. On successful verify, a JWT is issued that other services can trust

Run:
  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import io
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse  # Naya import add kiya gaya hai
from pydantic import BaseModel
from PIL import Image
import jwt

from deepface import DeepFace

from .liveness import check_liveness

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent.parent / "faces.db"
JWT_SECRET = "CHANGE_THIS_SECRET_IN_PRODUCTION"  # move to env var in real deployment
JWT_ALGO = "HS256"
JWT_EXPIRY_MINUTES = 60

MODEL_NAME = "ArcFace"          # accurate, good balance of speed/accuracy
DETECTOR_BACKEND = "opencv"     # fast, no extra downloads; swap to "retinaface" for higher accuracy
MATCH_THRESHOLD = 0.68          # cosine distance threshold for ArcFace (lower = stricter)

app = FastAPI(title="Face Authentication Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            user_id TEXT PRIMARY KEY,
            embedding TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)


def get_embedding(img_array: np.ndarray) -> list:
    """Extract a face embedding vector using DeepFace/ArcFace."""
    try:
        result = DeepFace.represent(
            img_path=img_array,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"No face detected clearly: {e}")

    if not result:
        raise HTTPException(status_code=400, detail="No face detected")

    return result[0]["embedding"]


def cosine_distance(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def create_jwt(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Dependency other endpoints (or other services) can use to require auth."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing token")
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class VerifyResponse(BaseModel):
    match: bool
    user_id: Optional[str] = None
    distance: Optional[float] = None
    token: Optional[str] = None
    liveness_passed: Optional[bool] = None


class EnrollResponse(BaseModel):
    status: str
    user_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# Naya Homepage Endpoint
@app.get("/", response_class=FileResponse)
def home():
    """Serve the index.html page at the root URL."""
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    return str(html_path)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(user_id: str = Form(...), file: UploadFile = File(...)):
    """
    Register a user's face. Call this once during signup / profile setup.
    Overwrites any previously stored face for that user_id.
    """
    img_bytes = await file.read()
    img_array = read_image(img_bytes)
    embedding = get_embedding(img_array)

    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO faces (user_id, embedding, created_at) VALUES (?, ?, ?)",
        (user_id, json.dumps(embedding), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()

    return EnrollResponse(status="enrolled", user_id=user_id)


@app.post("/verify", response_model=VerifyResponse)
async def verify(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    check_live: bool = Form(False),
    frames: Optional[str] = Form(None),  # optional: JSON list of base64 frames for liveness
):
    """
    Verify a face against enrolled users.

    - If user_id is given: 1-to-1 match (typical login: "is this really user X?")
    - If user_id is omitted: 1-to-many search across all enrolled faces
    - Set check_live=true and pass `frames` (a few sequential webcam frames,
      base64-encoded JSON array) to run a basic blink-based liveness check
      before trusting the match.
    """
    img_bytes = await file.read()
    img_array = read_image(img_bytes)
    probe_embedding = get_embedding(img_array)

    liveness_passed = None
    if check_live:
        if not frames:
            raise HTTPException(status_code=400, detail="frames required when check_live=true")
        liveness_passed = check_liveness(json.loads(frames))
        if not liveness_passed:
            return VerifyResponse(match=False, liveness_passed=False)

    conn = get_db()
    if user_id:
        row = conn.execute("SELECT * FROM faces WHERE user_id = ?", (user_id,)).fetchall()
    else:
        row = conn.execute("SELECT * FROM faces").fetchall()
    conn.close()

    best_match = None
    best_distance = 999.0
    for r in row:
        stored_embedding = json.loads(r["embedding"])
        dist = cosine_distance(probe_embedding, stored_embedding)
        if dist < best_distance:
            best_distance = dist
            best_match = r["user_id"]

    if best_match and best_distance < MATCH_THRESHOLD:
        token = create_jwt(best_match)
        return VerifyResponse(
            match=True,
            user_id=best_match,
            distance=round(best_distance, 4),
            token=token,
            liveness_passed=liveness_passed,
        )

    return VerifyResponse(
        match=False,
        distance=round(best_distance, 4) if best_match else None,
        liveness_passed=liveness_passed,
    )


@app.delete("/enroll/{user_id}")
def delete_face(user_id: str, current_user: str = Depends(verify_jwt)):
    """Remove an enrolled face. Requires a valid token."""
    conn = get_db()
    conn.execute("DELETE FROM faces WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "user_id": user_id}


@app.get("/whoami")
def whoami(current_user: str = Depends(verify_jwt)):
    """Any other project can call this with the issued JWT to confirm identity."""
    return {"user_id": current_user}


# Serve the demo webcam page
static_dir = Path(__file__).parent.parent / "static"
app.mount("/demo", StaticFiles(directory=static_dir, html=True), name="demo")