"""Approximate projection visibility for II-10."""
from __future__ import annotations

from typing import Any


def project_y_up(points: Any, k: Any) -> tuple[Any, Any, Any]:
    z = points[:, 2].clamp(min=1e-6)
    u = k[0, 0] * (points[:, 0] / z) + k[0, 2]
    v = k[1, 2] - k[1, 1] * (points[:, 1] / z)
    return u, v, z


def sample_map(map_t: Any, u: Any, v: Any, image_hw: tuple[int, int], mode: str = "bilinear") -> Any:
    import torch
    import torch.nn.functional as F

    h, w = image_hw
    gx = (u / max(w - 1, 1)) * 2.0 - 1.0
    gy = (v / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    return F.grid_sample(map_t, grid, mode=mode, padding_mode="zeros", align_corners=True).view(map_t.shape[1], -1).T


def gather_projected_observation(
    points: Any,
    k: Any,
    obs_t: dict[str, Any],
    image_hw: tuple[int, int],
    *,
    visibility_margin_m: float = 0.20,
    use_foreground_for_body: bool = False,
    use_symmetric_depth_window: bool = False,
) -> dict[str, Any]:
    import torch

    u, v, z_model = project_y_up(points, k)
    h, w = image_hw
    inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    p_obs = sample_map(obs_t["xyz"], u, v, image_hw)
    n_obs = sample_map(obs_t["normal"], u, v, image_hw)
    valid = sample_map(obs_t["valid"], u, v, image_hw)[:, 0] > 0.5
    skin = sample_map(obs_t["skin"], u, v, image_hw)[:, 0] > 0.5
    cloth = sample_map(obs_t["cloth"], u, v, image_hw)[:, 0] > 0.5
    shoe = sample_map(obs_t["shoe"], u, v, image_hw)[:, 0] > 0.5
    fg = sample_map(obs_t["fg"], u, v, image_hw)[:, 0] > 0.5
    fg_dist = sample_map(obs_t["fg_dist"], u, v, image_hw)[:, 0]
    if use_symmetric_depth_window:
        near_surface = (z_model - p_obs[:, 2]).abs() <= visibility_margin_m
    else:
        near_surface = z_model >= (p_obs[:, 2] - visibility_margin_m)
    body_region = (fg if use_foreground_for_body else (skin | cloth))
    usable = inside & valid & near_surface & body_region & (~shoe)
    return {
        "u": u,
        "v": v,
        "z_model": z_model,
        "p_model": points,
        "p_obs": p_obs,
        "n_obs": n_obs,
        "inside_mask": inside,
        "valid_depth_mask": valid,
        "fg_mask": fg,
        "skin_mask": usable & skin,
        "cloth_mask": usable & cloth,
        "foreground_body_mask": usable,
        "shoe_mask": inside & valid & shoe,
        "body_mask": usable,
        "silhouette_distance": torch.where(inside, fg_dist, fg_dist.new_full(fg_dist.shape, 100.0)),
    }
