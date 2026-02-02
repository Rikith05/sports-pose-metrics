# Sports Pose Metrics

## Approach
This project provides real-time and batch pose estimation for sports videos using MediaPipe. It includes:
- A **Streamlit web app** for uploading videos, live camera pose estimation, and viewing/downloading results
- A **command-line pipeline** for batch processing
- Real-time angle computation and “perfect/not perfect” form feedback
- Recording and auto-save capabilities for live sessions

## Model Used & Why
- **MediaPipe Pose Landmarker (Lite)** – CPU-efficient, runs in real-time on consumer hardware
- Chosen for:
  - Speed (real-time performance on webcam)
  - Stability for full-body landmarks
  - No GPU requirement
  - Consistent joint naming and visibility scores

## Metrics Defined
- **Torso lean angle**: Angle of torso relative to vertical (balance/posture indicator)
- **Elbow angles (left/right)**: Shoulder–elbow–wrist angle (arm mechanics)
- **Knee angles (left/right)**: Hip–knee–ankle angle (lower-body engagement)
- **Wrist speed (normalized)**: Proxy for swing/throw speed, scaled by torso length
- **Form status**: “PERFECT” if angles are within predefined thresholds, otherwise “NOT PERFECT”

## Observations & Limitations
- **Occlusion handling**: Lower body (knees/ankles) often occluded by equipment or framing
- **Jitter**: Minor wrist jitter during fast motion; mitigated with EMA smoothing
- **Lighting dependency**: Requires adequate lighting for reliable detection
- **Side-view bias**: Metrics optimized for side/semi-side views
- **Real-time vs accuracy trade-off**: Live camera uses lighter processing for speed

## Improvement Plan
- **Temporal smoothing**: Implement Kalman filter for more stable tracking
- **Multi-view support**: Combine front/side views for better occlusion handling
- **Sport-specific tuning**: Adjust angle thresholds per sport (e.g., golf vs squat)
- **Advanced models**: Evaluate RTMPose/YOLO-Pose for higher accuracy
- **Mobile optimization**: Reduce model size for mobile deployment
- **Data collection**: Gather sport-specific annotated datasets for fine-tuning

## Outputs
### Batch Processing (`run_pipeline.py`)
- `overlay.mp4`: Video with skeleton overlay and angle text
- `keypoints.csv/.json`: Frame-wise joint coordinates and visibility
- `metrics.csv/.json`: Computed angles and speeds per frame
- `metrics_summary.json`: Statistical summary and event frames

### Live Camera (`app.py`)
- Real-time video overlay with angles and form status
- Optional recording to MP4
- Auto-save functionality with timestamped files
- Live angle display in sidebar

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Streamlit Web App
```bash
streamlit run app.py
```
- Upload videos for batch processing
- Use live camera for real-time feedback
- Browse and download outputs

### Command-Line Pipeline
```bash
python src/run_pipeline.py --input data/squat_side_view.mp4 --output outputs/run1
```

### Live Camera (CLI)
```bash
python src/live_camera.py
```

## Project Structure
```
sports-pose-metrics/
├── app.py                 # Streamlit web app
├── src/
│   ├── run_pipeline.py    # Batch processing
│   ├── live_camera.py     # CLI live camera
│   ├── pose_estimation.py # MediaPipe wrapper & drawing
│   └── metrics.py         # Angle/speed computation
├── data/
│   └── squat_side_view.mp4
├── outputs/               # Generated results
├── pose_landmarker.task   # MediaPipe model file
└── requirements.txt
```

## Key Features
- **Real-time feedback**: Angles and form status update live
- **Recording**: Save annotated sessions for review
- **Auto-save**: Continuous recording to timestamped files
- **Clean UI**: Simple Streamlit interface with no HTML/Flask
- **Cross-platform**: Runs on Windows/Mac/Linux with webcam
