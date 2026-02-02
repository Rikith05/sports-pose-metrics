from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


JOINTS: List[str] = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _pose_landmark_index(name: str) -> int:
    mapping = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }
    return int(mapping[name])


CONNECTIONS: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    num_frames: Optional[int]


class PoseSmoother:
    def __init__(self, alpha: float):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)
        self._state: Dict[str, np.ndarray] = {}

    def update(self, joint: str, x: float, y: float, z: float) -> Tuple[float, float, float]:
        v = np.array([x, y, z], dtype=float)
        if joint not in self._state:
            self._state[joint] = v
            return float(v[0]), float(v[1]), float(v[2])

        prev = self._state[joint]
        if self.alpha >= 1.0:
            self._state[joint] = v
        elif self.alpha <= 0.0:
            self._state[joint] = prev
        else:
            self._state[joint] = self.alpha * v + (1.0 - self.alpha) * prev

        out = self._state[joint]
        return float(out[0]), float(out[1]), float(out[2])


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _video_meta(cap: cv2.VideoCapture) -> VideoMeta:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames = int(n) if n and n > 0 else None

    return VideoMeta(fps=fps, width=width, height=height, num_frames=num_frames)


def _draw_skeleton(
    frame_bgr: np.ndarray,
    joint_xy_px: Dict[str, Tuple[int, int]],
) -> np.ndarray:
    for a, b in CONNECTIONS:
        if a in joint_xy_px and b in joint_xy_px:
            cv2.line(frame_bgr, joint_xy_px[a], joint_xy_px[b], (0, 255, 0), 2)

    for name, (x, y) in joint_xy_px.items():
        if name.startswith("left_"):
            color = (0, 0, 255)
        elif name.startswith("right_"):
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)
        cv2.circle(frame_bgr, (x, y), 4, color, -1)

    return frame_bgr


def run_pose_estimation(
    input_video_path: Union[str, Path],
    overlay_video_path: Union[str, Path],
    *,
    smoothing_alpha: float = 0.3,
    visibility_threshold: float = 0.5,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    max_frames: Optional[int] = None,
) -> Tuple[pd.DataFrame, VideoMeta]:
    input_video_path = Path(input_video_path)
    overlay_video_path = Path(overlay_video_path)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video_path}")

    meta = _video_meta(cap)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _ensure_parent(overlay_video_path)
    writer = cv2.VideoWriter(str(overlay_video_path), fourcc, meta.fps, (meta.width, meta.height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {overlay_video_path}")

    smoother = PoseSmoother(alpha=smoothing_alpha)

    import mediapipe as mp
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    Image = mp.Image
    ImageFormat = mp.ImageFormat

    base_options = BaseOptions(
        model_asset_path="pose_landmarker.task"
    )
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=float(min_detection_confidence),
        min_pose_presence_confidence=float(min_tracking_confidence),
    )
    landmarker = PoseLandmarker.create_from_options(options)

    records: List[Dict[str, float]] = []

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if max_frames is not None and frame_idx >= int(max_frames):
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * 1000 / meta.fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row: Dict[str, float] = {
                "frame": float(frame_idx),
                "time_s": float(frame_idx) / float(meta.fps),
            }

            joint_xy_px: Dict[str, Tuple[int, int]] = {}

            if not result.pose_landmarks:
                row["detected"] = 0.0
                row["mean_visibility"] = float("nan")
                row["num_visible"] = 0.0

                for j in JOINTS:
                    row[f"{j}_x"] = float("nan")
                    row[f"{j}_y"] = float("nan")
                    row[f"{j}_z"] = float("nan")
                    row[f"{j}_visibility"] = float("nan")

                writer.write(frame_bgr)
                records.append(row)
                frame_idx += 1
                continue

            landmarks = result.pose_landmarks[0]
            row["detected"] = 1.0

            visibilities: List[float] = []
            num_visible = 0

            for j in JOINTS:
                idx = _pose_landmark_index(j)
                lm = landmarks[idx]

                vis = float(lm.visibility) if hasattr(lm, "visibility") else 1.0
                visibilities.append(vis)
                row[f"{j}_visibility"] = vis

                if vis < float(visibility_threshold):
                    row[f"{j}_x"] = float("nan")
                    row[f"{j}_y"] = float("nan")
                    row[f"{j}_z"] = float("nan")
                    continue

                num_visible += 1

                x_s, y_s, z_s = smoother.update(j, float(lm.x), float(lm.y), float(lm.z))

                row[f"{j}_x"] = x_s
                row[f"{j}_y"] = y_s
                row[f"{j}_z"] = z_s

                x_px = int(round(x_s * float(meta.width)))
                y_px = int(round(y_s * float(meta.height)))
                if 0 <= x_px < meta.width and 0 <= y_px < meta.height:
                    joint_xy_px[j] = (x_px, y_px)

            row["mean_visibility"] = float(np.mean(visibilities)) if visibilities else float("nan")
            row["num_visible"] = float(num_visible)

            frame_out = _draw_skeleton(frame_bgr, joint_xy_px)
            writer.write(frame_out)

            records.append(row)
            frame_idx += 1

    finally:
        landmarker.close()
        cap.release()
        writer.release()

    df = pd.DataFrame.from_records(records)

    df["frame"] = df["frame"].astype(int)
    df["detected"] = df["detected"].astype(int)
    df["num_visible"] = df["num_visible"].astype(int)

    return df, meta
