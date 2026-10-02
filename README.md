# Face Authentication Microservice

A standalone REST API for face-based authentication that you can drop into
any other project (Django, React, Flutter, another FastAPI service, etc.)
without duplicating ML code everywhere.

## Why this design

- **Standalone service** — enroll once, then any of your apps calls this API
  over HTTP to verify. CancerAI, Royal Pay, your exam proctoring system —
  all could call the same face-auth service instead of each reimplementing it.
- **DeepFace + ArcFace** instead of `face_recognition`/dlib — avoids the
  dlib C++ compile step that's caused you Windows PATH/build-tool pain before.
  DeepFace pulls prebuilt model weights on first run.
- **MediaPipe for liveness** (blink detection) — also pip-installs cleanly on
  Windows, no compiler needed.
- **JWT tokens** — on successful verify, you get a token any of your other
  backends can validate independently (they just need the same `JWT_SECRET`,
  or you route all `/whoami` checks through this service).

## Setup (Windows-friendly)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

First request will download ArcFace weights automatically (~50MB, cached
after that in `~/.deepface`).

Open the demo UI: **http://localhost:8000/demo**

## Or run with Docker (recommended — sidesteps all Windows dependency issues)

```bash
docker compose up --build
```

## API Reference

### `POST /enroll`
Register a face. Call once at signup.
```
form-data: user_id=<str>, file=<image>
```
```json
{ "status": "enrolled", "user_id": "prakash" }
```

### `POST /verify`
Verify a face for login.
```
form-data: file=<image>, user_id=<optional str>, check_live=<bool>
```
- Omit `user_id` to search across all enrolled faces (1:N).
- Pass `user_id` to check against just that one person (1:1, faster & more secure).

```json
{
  "match": true,
  "user_id": "prakash",
  "distance": 0.31,
  "token": "eyJhbGciOi...",
  "liveness_passed": null
}
```

### `GET /whoami`
Any other service calls this with `Authorization: Bearer <token>` to confirm
who the token belongs to.

### `DELETE /enroll/{user_id}`
Remove an enrolled face. Requires a valid token.

## Integrating into your other projects

**From Django (Royal Pay, CancerAI, etc.):**
```python
import requests

def verify_face(image_file):
    resp = requests.post(
        "http://face-auth-service:8000/verify",
        files={"file": image_file},
    )
    data = resp.json()
    if data["match"]:
        # trust data["token"] as their session, or look up data["user_id"]
        ...
```

**From React frontend:**
```js
const formData = new FormData();
formData.append("file", capturedBlob, "face.jpg");
const res = await fetch("http://localhost:8000/verify", { method: "POST", body: formData });
const { match, user_id, token } = await res.json();
```

Store the returned JWT the same way you'd store any auth token (httpOnly
cookie or localStorage depending on your app's existing pattern).

## Tuning accuracy

- `MATCH_THRESHOLD` in `app/main.py` (default `0.68`) — lower = stricter
  (fewer false accepts, more false rejects). Test with your own face at a
  few thresholds and pick what works.
- `DETECTOR_BACKEND = "opencv"` is fast; switch to `"retinaface"` for better
  accuracy on angled/poorly lit faces (slower, downloads an extra model).

## Security notes before using this for anything real

- Change `JWT_SECRET` to a real secret from an environment variable, not the
  hardcoded placeholder.
- The blink-liveness check is a basic deterrent, not real anti-spoofing —
  a video replay of someone's face could still pass it. Fine for a portfolio
  project; not sufficient for a banking-grade product (see how Royal Pay's
  payment providers likely require dedicated KYC/liveness vendors for that).
- Add rate limiting to `/verify` to prevent brute-force probing.
- CORS is wide open (`allow_origins=["*"]`) for demo purposes — restrict it
  to your real frontend domains in production.
