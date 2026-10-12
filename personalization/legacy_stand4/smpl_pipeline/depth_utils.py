"""Depth loading and normal computation for II-10."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .ii8_init import load_observed_xyz


def compute_normals_from_xyz(xyz: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Compute unit normals from local finite differences."""
    h, w, _ = xyz.shape
    normals = np.zeros_like(xyz, dtype=np.float32)
    dx = np.zeros_like(xyz, dtype=np.float32)
    dy = np.zeros_like(xyz, dtype=np.float32)
    dx[:, 1:-1] = xyz[:, 2:] - xyz[:, :-2]
    dy[1:-1, :] = xyz[2:, :] - xyz[:-2, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1)
    ok = valid & np.isfinite(n).all(axis=-1) & (norm > 1e-8)
    n[ok] = n[ok] / norm[ok, None]
    flip = ok & (n[..., 2] > 0.0)
    n[flip] *= -1.0
    normals[ok] = n[ok]
    return normals, {
        "valid_normals": int(ok.sum()),
        "invalid_normals": int((valid & ~ok).sum()),
        "flipped_normals": int(flip.sum()),
    }


def load_depth_observation(
    depth_path: Path,
    k: np.ndarray,
    depth_y_axis: str,
) -> dict:
    xyz, valid = load_observed_xyz(depth_path, k, depth_y_axis)
    normals, diag = compute_normals_from_xyz(xyz, valid)
    return {
        "xyz_obs": xyz.astype(np.float32),
        "z_obs": xyz[..., 2].astype(np.float32),
        "valid_mask": valid.astype(bool),
        "normal_obs": normals.astype(np.float32),
        "normal_diagnostics": diag,
    }

