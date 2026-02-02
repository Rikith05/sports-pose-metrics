#!/usr/bin/env python3
"""
Live camera pose estimation with real-time metrics overlay.
Press SPACE to toggle metrics overlay, 'r' to start/stop recording, 's' to save a snapshot, ESC to quit.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from metrics import compute_metrics
from pose_estimation import JOINTS, PoseSmoother, _pose_landmark_index, _draw_skeleton

# MediaPipe Tasks setup
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
Image = mp.Image
ImageFormat = mp.ImageFormat


def init_pose_landmarker(model_path: str = "pose_landmarker.task"):
    base_options = BaseOptions(model_asset_path=model_path)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.LIVE_STREAM,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        result_callback=pose_callback,
    )
    return PoseLandmarker.create_from_options(options)


# Global to store latest result
_latest_result = None


def pose_callback(result, timestamp_ms: int, image):
    global _latest_result
    _latest_result = result


def draw_metrics_overlay(
    frame: np.ndarray,
    metrics: Dict[str, float],
    fps: float,
    frame_idx: int,
) -> np.ndarray:
    """Draw simple metrics overlay on the frame."""
    y0, dy = 30, 25
    for i, (k, v) in enumerate(metrics.items()):
        if isinstance(v, float):
            txt = f"{k}: {v:.1f}"
        else:
            txt = f"{k}: {v}"
        cv2.putText(frame, txt, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Frame: {frame_idx} | FPS: {fps:.1f}", (10, y0 + len(metrics) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Live camera pose estimation with metrics")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--model", type=str, default="pose_landmarker.task", help="Pose model path")
    parser.add_argument("--smoothing", type=float, default=0.3, help="EMA smoothing alpha")
    parser.add_argument("--record-dir", type=str, default="outputs/live_record", help="Directory to save recordings")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    landmarker = init_pose_landmarker(args.model)
    smoother = PoseSmoother(alpha=args.smoothing)

    record_dir = Path(args.record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    video_writer: Optional[cv2.VideoWriter] = None
    recording = False

    show_metrics = True
    frame_idx = 0
    last_time = time.time()
    fps_history: List[float] = []

    # Storage for post-run metrics export
    all_records: List[Dict[str, float]] = []

    print("Controls: SPACE=toggle metrics, r=record toggle, s=snapshot, ESC=quit")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        # Get latest result from callback
        global _latest_result
        result = _latest_result
        landmarks = result.pose_landmarks[0] if result and result.pose_landmarks else None

        # Prepare row for metrics
        row: Dict[str, float] = {
            "frame": float(frame_idx),
            "time_s": float(frame_idx) / 30.0,  # assume 30 fps for timing
        }
        joint_xy_px: Dict[str, tuple[int, int]] = {}

        if landmarks:
            row["detected"] = 1.0
            vis_vals = []
            for j in JOINTS:
                idx = _pose_landmark_index(j)
                lm = landmarks[idx]
                vis = float(lm.visibility) if hasattr(lm, "visibility") else 1.0
                row[f"{j}_visibility"] = vis
                vis_vals.append(vis)
                if vis >= 0.5:
                    x_s, y_s, z_s = smoother.update(j, float(lm.x), float(lm.y), float(lm.z))
                    row[f"{j}_x"] = x_s
                    row[f"{j}_y"] = y_s
                    row[f"{j}_z"] = z_s
                    x_px = int(round(x_s * args.width))
                    y_px = int(round(y_s * args.height))
                    if 0 <= x_px < args.width and 0 <= y_px < args.height:
                        joint_xy_px[j] = (x_px, y_px)
                else:
                    row[f"{j}_x"] = float("nan")
                    row[f"{j}_y"] = float("nan")
                    row[f"{j}_z"] = float("nan")
            row["mean_visibility"] = np.mean(vis_vals)
            row["num_visible"] = sum(1 for v in vis_vals if v >= 0.5)
        else:
            row["detected"] = 0.0
            row["mean_visibility"] = 0.0
            row["num_visible"] = 0
            for j in JOINTS:
                row[f"{j}_visibility"] = 0.0
                row[f"{j}_x"] = float("nan")
                row[f"{j}_y"] = float("nan")
                row[f"{j}_z"] = float("nan")

        all_records.append(row)

        # Draw skeleton
        frame_out = _draw_skeleton(frame_bgr, joint_xy_px)

        # Compute metrics if we have enough data
        if len(all_records) >= 2:
            df = pd.DataFrame.from_records(all_records)
            df["frame"] = df["frame"].astype(int)
            df["detected"] = df["detected"].astype(int)
            metrics_df, _ = compute_metrics(df, fps=30.0, width=args.width, height=args.height)
            if not metrics_df.empty:
                latest_metrics = metrics_df.iloc[-1].to_dict()
                # Show only a few key metrics
                display_metrics = {
                    "torso_lean_deg": latest_metrics.get("torso_lean_deg", np.nan),
                    "left_elbow_deg": latest_metrics.get("left_elbow_angle_deg", np.nan),
                    "right_elbow_deg": latest_metrics.get("right_elbow_angle_deg", np.nan),
                    "left_wrist_speed": latest_metrics.get("left_wrist_speed_norm", np.nan),
                    "right_wrist_speed": latest_metrics.get("right_wrist_speed_norm", np.nan),
                }
                if show_metrics:
                    frame_out = draw_metrics_overlay(frame_out, display_metrics, fps=0.0, frame_idx=frame_idx)

        # FPS calculation
        now = time.time()
        fps = 1.0 / (now - last_time) if now != last_time else 0.0
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        last_time = now

        # Recording
        if recording:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = record_dir / f"recording_{int(time.time())}.mp4"
                video_writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (args.width, args.height))
            video_writer.write(frame_out)

        cv2.imshow("Live Pose Estimation", frame_out)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(" "):
            show_metrics = not show_metrics
        elif key == ord("r"):
            recording = not recording
            if not recording and video_writer is not None:
                video_writer.release()
                video_writer = None
            print(f"Recording {'started' if recording else 'stopped'}")
        elif key == ord("s"):
            snap_path = record_dir / f"snapshot_{int(time.time())}.png"
            cv2.imwrite(str(snap_path), frame_out)
            print(f"Snapshot saved to {snap_path}")

        frame_idx += 1

    # Cleanup
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # Export metrics and keypoints
    if all_records:
        df = pd.DataFrame.from_records(all_records)
        df["frame"] = df["frame"].astype(int)
        df["detected"] = df["detected"].astype(int)
        df.to_csv(record_dir / "keypoints.csv", index=False)
        df.to_json(record_dir / "keypoints.json", orient="records")
        metrics_df, _ = compute_metrics(df, fps=30.0, width=args.width, height=args.height)
        metrics_df.to_csv(record_dir / "metrics.csv", index=False)
        # Simple summary (since summarize_metrics is internal)
        summary = {
            "fps": 30.0,
            "num_frames": len(df),
            "torso_lean_deg": {"min": metrics_df["torso_lean_deg"].min(), "max": metrics_df["torso_lean_deg"].max()},
            "left_elbow_angle_deg": {"min": metrics_df["left_elbow_angle_deg"].min(), "max": metrics_df["left_elbow_angle_deg"].max()},
            "right_elbow_angle_deg": {"min": metrics_df["right_elbow_angle_deg"].min(), "max": metrics_df["right_elbow_angle_deg"].max()},
        }
        (record_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Saved keypoints/metrics to {record_dir}")


if __name__ == "__main__":
    main()
