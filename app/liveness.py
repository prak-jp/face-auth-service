"""
Very lightweight liveness check: detects a blink across a short sequence of
frames using MediaPipe FaceMesh eye-aspect-ratio (EAR). This stops someone
from holding up a static printed photo or a phone screenshot to /verify.

This is NOT bulletproof anti-spoofing (a video replay could still pass) -
for production-grade liveness, look at commercial SDKs or 3D depth sensors.
This is meant as a reasonable free/open-source baseline.
"""

import base64
import io
from typing import List

import numpy as np
from PIL import Image
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices (MediaPipe FaceMesh, 468-point model)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_BLINK_THRESHOLD = 0.21


def _eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices])
    vertical1 = np.linalg.norm(pts[1] - pts[5])
    vertical2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal == 0:
        return 0.3
    return (vertical1 + vertical2) / (2.0 * horizontal)


def _decode_base64_frame(b64_str: str) -> np.ndarray:
    if "," in b64_str:  # strip data URL prefix like "data:image/jpeg;base64,"
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def check_liveness(base64_frames: List[str]) -> bool:
    """
    Pass 5-15 sequential webcam frames (roughly 1-2 seconds of video at low fps).
    Returns True if a blink (EAR dip then recovery) was detected across frames,
    which a static photo cannot produce.
    """
    if len(base64_frames) < 3:
        return False

    ear_values = []
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as face_mesh:
        for b64 in base64_frames:
            frame = _decode_base64_frame(b64)
            h, w, _ = frame.shape
            results = face_mesh.process(frame)
            if not results.multi_face_landmarks:
                continue
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear_values.append((left_ear + right_ear) / 2.0)

    if len(ear_values) < 3:
        return False

    # A blink = EAR dropping below threshold then rising back above it
    below = [e < EAR_BLINK_THRESHOLD for e in ear_values]
    for i in range(1, len(below) - 1):
        if below[i] and not below[i - 1]:
            # dipped, check it recovers later
            if any(not b for b in below[i + 1:]):
                return True
    return False
