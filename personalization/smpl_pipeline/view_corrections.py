"""View-specific diagnostic/canonical alignment corrections."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_view_corrections(path: str | Path | None) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "views" in payload:
        payload = payload["views"]
    return {str(k): dict(v) for k, v in payload.items()}


def _rot_y_np(deg: float) -> np.ndarray:
    t = math.radians(float(deg))
    c = math.cos(t)
    s = math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def correction_pivot_np(points: np.ndarray, corr: dict[str, float]) -> np.ndarray:
    if "pivot_xyz" in corr:
        return np.asarray(corr["pivot_xyz"], dtype=np.float64)
    return np.asarray(points, dtype=np.float64).mean(axis=0)


def apply_camera_correction_np(points: np.ndarray, view: str, corrections: dict[str, dict[str, float]]) -> np.ndarray:
    corr = corrections.get(view, {})
    if not corr:
        return np.asarray(points, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    pivot = correction_pivot_np(pts, corr)
    r = _rot_y_np(float(corr.get("yaw_delta_deg", 0.0)))
    out = pivot + (r @ (pts - pivot).T).T
    out = out + np.array(
        [
            float(corr.get("x_shift_m", 0.0)),
            float(corr.get("y_shift_m", 0.0)),
            float(corr.get("z_shift_m", 0.0)),
        ],
        dtype=np.float64,
    )
    return out


def invert_camera_correction_np(points: np.ndarray, view: str, corrections: dict[str, dict[str, float]]) -> np.ndarray:
    corr = corrections.get(view, {})
    if not corr:
        return np.asarray(points, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    shift = np.array(
        [
            float(corr.get("x_shift_m", 0.0)),
            float(corr.get("y_shift_m", 0.0)),
            float(corr.get("z_shift_m", 0.0)),
        ],
        dtype=np.float64,
    )
    pivot = correction_pivot_np(pts - shift, corr)
    r_inv = _rot_y_np(-float(corr.get("yaw_delta_deg", 0.0)))
    return pivot + (r_inv @ (pts - shift - pivot).T).T


def apply_camera_correction_torch(points: Any, view: str, corrections: dict[str, dict[str, float]]) -> Any:
    corr = corrections.get(view, {})
    if not corr:
        return points
    import torch

    pivot = (
        torch.tensor(corr["pivot_xyz"], device=points.device, dtype=points.dtype)
        if "pivot_xyz" in corr
        else points.mean(dim=0)
    )
    theta = torch.tensor(math.radians(float(corr.get("yaw_delta_deg", 0.0))), device=points.device, dtype=points.dtype)
    c = torch.cos(theta)
    s = torch.sin(theta)
    r = torch.stack(
        [
            torch.stack([c, torch.zeros_like(c), s]),
            torch.stack([torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c)]),
            torch.stack([-s, torch.zeros_like(c), c]),
        ],
        dim=0,
    )
    shift = torch.tensor(
        [
            float(corr.get("x_shift_m", 0.0)),
            float(corr.get("y_shift_m", 0.0)),
            float(corr.get("z_shift_m", 0.0)),
        ],
        device=points.device,
        dtype=points.dtype,
    )
    return pivot + (points - pivot) @ r.T + shift
