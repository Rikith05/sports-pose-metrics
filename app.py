#!/usr/bin/env python3
"""
Streamlit app for sports pose estimation.
Upload a video, use live camera, and view/download outputs.
"""
import sys
import tempfile
import time
from pathlib import Path
import json

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from metrics import compute_metrics, _angle_deg
from pose_estimation import JOINTS, PoseSmoother, _pose_landmark_index, _draw_skeleton
from run_pipeline import run as run_pipeline

# MediaPipe Tasks setup
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
Image = mp.Image
ImageFormat = mp.ImageFormat

# Globals
_landmarker = None
_latest_result = None
_recording_frames = []
_is_recording = False
_last_frame = None
_auto_save_writer = None
_auto_save_path = None
_last_angles = {"left_elbow": 0.0, "right_elbow": 0.0, "left_knee": 0.0, "right_knee": 0.0}


def get_landmarker():
    global _landmarker
    if _landmarker is None:
        base_options = BaseOptions(model_asset_path="pose_landmarker.task")
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.VIDEO,
        )
        _landmarker = PoseLandmarker.create_from_options(options)
    return _landmarker


def pose_callback(result, timestamp_ms: int, image):
    global _latest_result
    _latest_result = result


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Process each frame from WebRTC for real pose estimation and overlay angles."""
    img_bgr = frame.to_ndarray(format="bgr24")
    h, w = img_bgr.shape[:2]

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=img_rgb)
    timestamp_ms = int(time.time() * 1000)

    # Send frame to landmarker (synchronous for accuracy)
    landmarker = get_landmarker()
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    landmarks = result.pose_landmarks[0] if result and result.pose_landmarks else None

    # Default values if no pose detected
    left_elbow = right_elbow = left_knee = right_knee = 0.0
    joint_xy_px = {}

    if landmarks:
        # Map joints to pixel coordinates
        for j in JOINTS:
            idx = _pose_landmark_index(j)
            lm = landmarks[idx]
            vis = float(lm.visibility) if hasattr(lm, "visibility") else 1.0
            if vis >= 0.5:
                x_px = int(round(float(lm.x) * w))
                y_px = int(round(float(lm.y) * h))
                if 0 <= x_px < w and 0 <= y_px < h:
                    joint_xy_px[j] = (x_px, y_px)

        # Compute angles if we have the required joints
        if all(j in joint_xy_px for j in ["left_shoulder", "left_elbow", "left_wrist"]):
            left_elbow = float(_angle_deg(
                np.array([joint_xy_px["left_shoulder"][0], joint_xy_px["left_shoulder"][1]]),
                np.array([joint_xy_px["left_elbow"][0], joint_xy_px["left_elbow"][1]]),
                np.array([joint_xy_px["left_wrist"][0], joint_xy_px["left_wrist"][1]]),
            ))
        if all(j in joint_xy_px for j in ["right_shoulder", "right_elbow", "right_wrist"]):
            right_elbow = float(_angle_deg(
                np.array([joint_xy_px["right_shoulder"][0], joint_xy_px["right_shoulder"][1]]),
                np.array([joint_xy_px["right_elbow"][0], joint_xy_px["right_elbow"][1]]),
                np.array([joint_xy_px["right_wrist"][0], joint_xy_px["right_wrist"][1]]),
            ))
        if all(j in joint_xy_px for j in ["left_hip", "left_knee", "left_ankle"]):
            left_knee = float(_angle_deg(
                np.array([joint_xy_px["left_hip"][0], joint_xy_px["left_hip"][1]]),
                np.array([joint_xy_px["left_knee"][0], joint_xy_px["left_knee"][1]]),
                np.array([joint_xy_px["left_ankle"][0], joint_xy_px["left_ankle"][1]]),
            ))
        if all(j in joint_xy_px for j in ["right_hip", "right_knee", "right_ankle"]):
            right_knee = float(_angle_deg(
                np.array([joint_xy_px["right_hip"][0], joint_xy_px["right_hip"][1]]),
                np.array([joint_xy_px["right_knee"][0], joint_xy_px["right_knee"][1]]),
                np.array([joint_xy_px["right_ankle"][0], joint_xy_px["right_ankle"][1]]),
            ))

        # Draw skeleton
        img_bgr = _draw_skeleton(img_bgr, joint_xy_px)

    # Determine if form is "perfect" (simple thresholds)
    perfect = (
        70 <= left_elbow <= 110 and
        70 <= right_elbow <= 110 and
        80 <= left_knee <= 120 and
        80 <= right_knee <= 120
    )
    status = "PERFECT" if perfect else "NOT PERFECT"
    color = (0, 255, 0) if perfect else (0, 0, 255)

    # Draw angles and status on frame
    y = 30
    for name, angle in [("L Elbow", left_elbow), ("R Elbow", right_elbow), ("L Knee", left_knee), ("R Knee", right_knee)]:
        txt = f"{name}: {angle:.0f}°"
        cv2.putText(img_bgr, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 35
    cv2.putText(img_bgr, status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Store frame if recording
    global _is_recording, _recording_frames, _last_frame, _auto_save_writer, _auto_save_path, _last_angles
    _last_frame = img_bgr.copy()
    if _is_recording:
        _recording_frames.append(img_bgr.copy())
    # Auto-save to file continuously
    if _auto_save_writer is not None:
        _auto_save_writer.write(img_bgr)

    # Update last known angles for display
    _last_angles = {
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_knee": left_knee,
        "right_knee": right_knee,
    }

    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


def main():
    st.set_page_config(page_title="Sports Pose Metrics", layout="wide")
    st.title("🏃 Sports Pose Estimation & Metrics")

    # Sidebar navigation
    mode = st.sidebar.selectbox("Mode", ["Upload Video", "Live Camera", "View Outputs"])
    st.sidebar.markdown("---")

    if mode == "Upload Video":
        st.header("Upload a video for pose analysis")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
        if uploaded_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            st.video(uploaded_file)

            if st.button("Run Pose Estimation"):
                with st.spinner("Processing video..."):
                    output_dir = Path("outputs/streamlit_upload")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    run_pipeline(
                        input_video=Path(tmp_path),
                        output_dir=output_dir,
                        smoothing_alpha=0.3,
                        visibility_threshold=0.5,
                        model_complexity=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                        max_frames=None,
                    )
                st.success("Done! Check outputs below.")

    elif mode == "Live Camera":
        st.header("Live Camera Pose Estimation")
        st.write("Enable your camera and allow browser permissions.")
        ctx = webrtc_streamer(
            key="live-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx.state.playing:
            st.success("Camera is active. Pose estimation overlay will appear here.")
            # Recording status indicator
            global _is_recording, _recording_frames, _auto_save_writer, _auto_save_path
            if _is_recording:
                st.error("🔴 Recording...")
            else:
                st.info("⏸️ Not recording")
            if _auto_save_writer is not None:
                st.warning("💾 Auto-saving to file...")
            # Recording controls
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                if st.button("Start Recording"):
                    _is_recording = True
                    _recording_frames = []
            with col2:
                if st.button("Stop Recording"):
                    _is_recording = False
            with col3:
                st.write(f"Frames captured: {len(_recording_frames)}")
                if st.button("Save Recording") and _recording_frames:
                    # Save frames as MP4
                    h, w = _recording_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_path = Path("outputs/live_recording.mp4")
                    out_path.parent.mkdir(exist_ok=True)
                    out = cv2.VideoWriter(str(out_path), fourcc, 20.0, (w, h))
                    for frame in _recording_frames:
                        out.write(frame)
                    out.release()
                    st.success(f"Recording saved to {out_path}")
                    with open(out_path, "rb") as f:
                        st.download_button("Download MP4", data=f.read(), file_name="live_recording.mp4", mime="video/mp4")
            # Auto-save controls
            st.subheader("Auto-Save")
            col_auto1, col_auto2, col_auto3 = st.columns([1,1,2])
            with col_auto1:
                if st.button("Start Auto-Save"):
                    if _auto_save_writer is None:
                        # Initialize writer with a timestamped filename
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        _auto_save_path = Path(f"outputs/auto_save_{timestamp}.mp4")
                        _auto_save_path.parent.mkdir(exist_ok=True)
                        # Assume 640x480 at 20 fps; adjust if needed
                        h, w = 480, 640
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        _auto_save_writer = cv2.VideoWriter(str(_auto_save_path), fourcc, 20.0, (w, h))
                        st.success(f"Auto-saving to {_auto_save_path}")
            with col_auto2:
                if st.button("Stop Auto-Save"):
                    if _auto_save_writer is not None:
                        _auto_save_writer.release()
                        _auto_save_writer = None
                        st.success(f"Auto-save stopped. File saved to {_auto_save_path}")
                        # Provide download
                        if _auto_save_path and _auto_save_path.exists():
                            with open(_auto_save_path, "rb") as f:
                                st.download_button("Download Auto-Save", data=f.read(), file_name=_auto_save_path.name, mime="video/mp4")
            with col_auto3:
                if _auto_save_path and _auto_save_path.exists():
                    st.write(f"File: {_auto_save_path.name}")
        else:
            st.info("Click 'Start' to enable the camera.")
        # Display latest angles in sidebar
        st.sidebar.markdown("### Live Angles")
        st.sidebar.json(_last_angles)

    elif mode == "View Outputs":
        st.header("Browse and download outputs")
        base_dir = Path("outputs")
        if not base_dir.exists():
            st.warning("No outputs directory found.")
            return

        # List runs and uploads
        runs = [d for d in base_dir.iterdir() if d.is_dir() and (d.name.startswith("run") or d.name == "streamlit_upload")]
        if not runs:
            st.warning("No run directories found.")
            return

        selected_run = st.selectbox("Select run", sorted(runs, key=lambda p: p.name, reverse=True))
        run_dir = base_dir / selected_run
        st.write(f"Debug: Selected directory = {run_dir}")

        # Show files
        files = list(run_dir.glob("*"))
        st.write(f"Debug: Found {len(files)} files")
        if not files:
            st.warning("No files found in this directory.")
            return
        for idx, f in enumerate(files):
            st.write(f"Debug file {idx}: {f.name} ({f.stat().st_size} bytes)")
            st.subheader(f.name)
            if f.suffix.lower() in [".mp4", ".mov", ".avi"]:
                st.video(str(f))
                st.download_button(f"Download {f.name}", data=open(f, "rb").read(), file_name=f.name, mime="video/mp4")
            elif f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                st.image(str(f))
                st.download_button(f"Download {f.name}", data=open(f, "rb").read(), file_name=f.name, mime="image/png")
            elif f.suffix.lower() in [".csv"]:
                st.write("CSV file (too large to preview)")
                st.caption(f"Size: {f.stat().st_size / (1024*1024):.2f} MB")
                st.download_button(f"Download {f.name}", data=open(f, "rb").read(), file_name=f.name, mime="text/csv")
            elif f.suffix.lower() in [".json"]:
                st.json(json.loads(open(f, "r").read()))
                st.download_button(f"Download {f.name}", data=open(f, "rb").read(), file_name=f.name, mime="application/json")
            else:
                st.write("Unsupported file type for preview.")
                st.download_button(f"Download {f.name}", data=open(f, "rb").read(), file_name=f.name)

        # Wrong-form highlights if present
        wrong_videos = list(base_dir.glob("wrong_form*.mp4"))
        if wrong_videos:
            st.header("❌ Wrong Form Highlights")
            for vid in wrong_videos:
                st.subheader(str(vid.name))
                st.video(str(vid))
                st.download_button(f"Download {vid.name}", data=open(vid, "rb").read(), file_name=vid.name, mime="video/mp4")


if __name__ == "__main__":
    main()
