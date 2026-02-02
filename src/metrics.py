import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _to_px(df: pd.DataFrame, name: str, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    x = df[f"{name}_x"].to_numpy(dtype=float) * float(width)
    y = df[f"{name}_y"].to_numpy(dtype=float) * float(height)
    return x, y


def _angle_deg(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> np.ndarray:
    a = np.stack([ax, ay], axis=1)
    b = np.stack([bx, by], axis=1)
    c = np.stack([cx, cy], axis=1)

    ba = a - b
    bc = c - b

    ba_norm = np.linalg.norm(ba, axis=1)
    bc_norm = np.linalg.norm(bc, axis=1)

    dot = np.einsum("ij,ij->i", ba, bc)

    with np.errstate(invalid="ignore", divide="ignore"):
        cos = dot / (ba_norm * bc_norm)

    cos = np.clip(cos, -1.0, 1.0)

    invalid = (ba_norm == 0.0) | (bc_norm == 0.0) | np.isnan(cos)
    angle = np.degrees(np.arccos(cos))
    angle[invalid] = np.nan

    return angle


def _speed_norm(
    x: np.ndarray,
    y: np.ndarray,
    fps: float,
    scale: float,
) -> np.ndarray:
    dx = np.diff(x)
    dy = np.diff(y)

    with np.errstate(invalid="ignore"):
        speed_px_s = np.sqrt(dx * dx + dy * dy) * float(fps)

    speed = np.concatenate([[np.nan], speed_px_s])

    if not np.isfinite(scale) or scale <= 0.0:
        return speed

    return speed / float(scale)


def _summarize(series: pd.Series) -> Optional[Dict[str, float]]:
    s = series.dropna()
    if s.empty:
        return None
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "range": float(s.max() - s.min()),
    }


def _idxmax_frame(metrics_df: pd.DataFrame, col: str) -> Optional[int]:
    s = metrics_df[col].dropna()
    if s.empty:
        return None
    return int(metrics_df.loc[s.idxmax(), "frame"])


def compute_metrics(
    keypoints_df: pd.DataFrame,
    fps: float,
    width: int,
    height: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metrics_df = keypoints_df[["frame", "time_s", "detected", "mean_visibility", "num_visible"]].copy()

    l_sh_x, l_sh_y = _to_px(keypoints_df, "left_shoulder", width, height)
    r_sh_x, r_sh_y = _to_px(keypoints_df, "right_shoulder", width, height)
    l_hp_x, l_hp_y = _to_px(keypoints_df, "left_hip", width, height)
    r_hp_x, r_hp_y = _to_px(keypoints_df, "right_hip", width, height)

    mid_sh_x = (l_sh_x + r_sh_x) / 2.0
    mid_sh_y = (l_sh_y + r_sh_y) / 2.0
    mid_hp_x = (l_hp_x + r_hp_x) / 2.0
    mid_hp_y = (l_hp_y + r_hp_y) / 2.0

    torso_len = np.sqrt((mid_sh_x - mid_hp_x) ** 2 + (mid_sh_y - mid_hp_y) ** 2)
    torso_scale = float(np.nanmedian(torso_len)) if np.isfinite(np.nanmedian(torso_len)) else float("nan")

    v_x = mid_sh_x - mid_hp_x
    v_y = mid_sh_y - mid_hp_y
    with np.errstate(invalid="ignore"):
        torso_lean_deg = np.degrees(np.arctan2(v_x, -v_y))

    metrics_df["torso_lean_deg"] = torso_lean_deg

    r_el_x, r_el_y = _to_px(keypoints_df, "right_elbow", width, height)
    r_wr_x, r_wr_y = _to_px(keypoints_df, "right_wrist", width, height)
    l_el_x, l_el_y = _to_px(keypoints_df, "left_elbow", width, height)
    l_wr_x, l_wr_y = _to_px(keypoints_df, "left_wrist", width, height)

    metrics_df["right_elbow_angle_deg"] = _angle_deg(r_sh_x, r_sh_y, r_el_x, r_el_y, r_wr_x, r_wr_y)
    metrics_df["left_elbow_angle_deg"] = _angle_deg(l_sh_x, l_sh_y, l_el_x, l_el_y, l_wr_x, l_wr_y)

    r_kn_x, r_kn_y = _to_px(keypoints_df, "right_knee", width, height)
    r_an_x, r_an_y = _to_px(keypoints_df, "right_ankle", width, height)
    l_kn_x, l_kn_y = _to_px(keypoints_df, "left_knee", width, height)
    l_an_x, l_an_y = _to_px(keypoints_df, "left_ankle", width, height)

    metrics_df["right_knee_angle_deg"] = _angle_deg(r_hp_x, r_hp_y, r_kn_x, r_kn_y, r_an_x, r_an_y)
    metrics_df["left_knee_angle_deg"] = _angle_deg(l_hp_x, l_hp_y, l_kn_x, l_kn_y, l_an_x, l_an_y)

    metrics_df["right_wrist_speed_norm"] = _speed_norm(r_wr_x, r_wr_y, fps, torso_scale)
    metrics_df["left_wrist_speed_norm"] = _speed_norm(l_wr_x, l_wr_y, fps, torso_scale)

    pose_detected_ratio = float(metrics_df["detected"].mean()) if "detected" in metrics_df.columns else float("nan")
    mean_visibility_detected = float(metrics_df.loc[metrics_df["detected"] == 1, "mean_visibility"].mean())

    missing_by_joint: Dict[str, float] = {}
    for joint in [
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
    ]:
        col = f"{joint}_x"
        if col in keypoints_df.columns:
            missing_by_joint[joint] = float(keypoints_df[col].isna().mean())

    right_visible = 1.0 - np.nanmean(
        [missing_by_joint.get("right_shoulder", math.nan), missing_by_joint.get("right_elbow", math.nan), missing_by_joint.get("right_wrist", math.nan)]
    )
    left_visible = 1.0 - np.nanmean(
        [missing_by_joint.get("left_shoulder", math.nan), missing_by_joint.get("left_elbow", math.nan), missing_by_joint.get("left_wrist", math.nan)]
    )
    dominant_side = "right" if right_visible >= left_visible else "left"

    hip_sway_norm = float(np.nanstd(mid_hp_x) / torso_scale) if np.isfinite(torso_scale) and torso_scale > 0 else float("nan")

    summary: Dict[str, Any] = {
        "fps": float(fps),
        "num_frames": int(len(metrics_df)),
        "pose_detected_ratio": pose_detected_ratio,
        "mean_visibility_detected": mean_visibility_detected,
        "dominant_side": dominant_side,
        "torso_scale_px_median": torso_scale,
        "hip_sway_norm": hip_sway_norm,
        "missing_by_joint": missing_by_joint,
        "metrics": {
            "torso_lean_deg": _summarize(metrics_df["torso_lean_deg"]),
            "right_elbow_angle_deg": _summarize(metrics_df["right_elbow_angle_deg"]),
            "left_elbow_angle_deg": _summarize(metrics_df["left_elbow_angle_deg"]),
            "right_knee_angle_deg": _summarize(metrics_df["right_knee_angle_deg"]),
            "left_knee_angle_deg": _summarize(metrics_df["left_knee_angle_deg"]),
            "right_wrist_speed_norm": _summarize(metrics_df["right_wrist_speed_norm"]),
            "left_wrist_speed_norm": _summarize(metrics_df["left_wrist_speed_norm"]),
        },
        "events": {
            "frame_peak_right_wrist_speed": _idxmax_frame(metrics_df, "right_wrist_speed_norm"),
            "frame_peak_left_wrist_speed": _idxmax_frame(metrics_df, "left_wrist_speed_norm"),
        },
    }

    return metrics_df, summary
