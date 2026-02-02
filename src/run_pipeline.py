import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from metrics import compute_metrics
from pose_estimation import run_pose_estimation


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_df_csv_json(df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records")


def _json_dump(path: Path, obj: Dict[str, Any]) -> None:
    def _default(o: Any) -> Any:
        try:
            import numpy as np

            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
        except Exception:
            pass
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def run(
    input_video: Path,
    output_dir: Path,
    *,
    smoothing_alpha: float,
    visibility_threshold: float,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    max_frames: Optional[int],
) -> Tuple[Path, Path, Path, Path, Path]:
    _ensure_dir(output_dir)

    overlay_path = output_dir / "overlay.mp4"
    keypoints_csv = output_dir / "keypoints.csv"
    keypoints_json = output_dir / "keypoints.json"

    keypoints_df, meta = run_pose_estimation(
        input_video,
        overlay_path,
        smoothing_alpha=smoothing_alpha,
        visibility_threshold=visibility_threshold,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        max_frames=max_frames,
    )

    _save_df_csv_json(keypoints_df, keypoints_csv, keypoints_json)

    metrics_df, summary = compute_metrics(keypoints_df, fps=meta.fps, width=meta.width, height=meta.height)

    metrics_csv = output_dir / "metrics.csv"
    metrics_summary_json = output_dir / "metrics_summary.json"

    metrics_df.to_csv(metrics_csv, index=False)
    _json_dump(metrics_summary_json, summary)

    return overlay_path, keypoints_csv, keypoints_json, metrics_csv, metrics_summary_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to an input video (mp4/mov/avi)")
    parser.add_argument("--output", default="outputs/run1", help="Output folder")

    parser.add_argument("--smoothing-alpha", type=float, default=0.3)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)

    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)

    parser.add_argument("--max-frames", type=int, default=None)

    args = parser.parse_args()

    input_video = Path(args.input)
    output_dir = Path(args.output)

    overlay_path, keypoints_csv, keypoints_json, metrics_csv, metrics_summary_json = run(
        input_video,
        output_dir,
        smoothing_alpha=float(args.smoothing_alpha),
        visibility_threshold=float(args.visibility_threshold),
        model_complexity=int(args.model_complexity),
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
        max_frames=int(args.max_frames) if args.max_frames is not None else None,
    )

    print(f"Overlay video: {overlay_path}")
    print(f"Keypoints CSV: {keypoints_csv}")
    print(f"Keypoints JSON: {keypoints_json}")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Metrics summary JSON: {metrics_summary_json}")


if __name__ == "__main__":
    main()
