"""SMPL surface sampling helpers."""
from __future__ import annotations

from typing import Any


def surface_points(vertices: Any, faces: Any, max_points: int) -> Any:
    """Return deterministic vertices + face centroids bounded by max_points."""
    import torch

    pts = [vertices]
    if faces is not None:
        face_v = vertices[faces]
        pts.append(face_v.mean(dim=1))
    all_pts = torch.cat(pts, dim=0)
    if max_points > 0 and all_pts.shape[0] > max_points:
        idx = torch.linspace(0, all_pts.shape[0] - 1, steps=max_points, device=vertices.device).long()
        all_pts = all_pts[idx]
    return all_pts

