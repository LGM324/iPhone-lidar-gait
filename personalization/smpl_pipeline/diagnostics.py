"""Diagnostics and export helpers for II-10."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .ii8_init import project_smpl_y_up_to_image


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for tri in faces.astype(np.int64) + 1:
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")


def height_report(vertices_per_view: list[np.ndarray], real_height_m: float = 1.810) -> dict:
    heights = [float(v[:, 1].max() - v[:, 1].min()) for v in vertices_per_view]
    mean_h = float(np.mean(heights))
    return {
        "height_per_view_m": heights,
        "mean_height_m": mean_h,
        "std_height_m": float(np.std(heights)),
        "real_height_m": real_height_m,
        "mean_error_m": mean_h - real_height_m,
        "acceptable_first_pass_range_m": [1.79, 1.84],
    }


def leg_cross_section_report(vertices: np.ndarray) -> dict:
    """Approximate leg widths from lower-body vertex slices."""
    y_min, y_max = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    h = y_max - y_min
    x_mid = float(np.median(vertices[:, 0]))

    def width_at(frac: float, band: float = 0.025) -> dict:
        y0 = y_min + frac * h
        pts = vertices[np.abs(vertices[:, 1] - y0) < band]
        if len(pts) < 10:
            return {"left_width_m": None, "right_width_m": None}
        left = pts[pts[:, 0] < x_mid]
        right = pts[pts[:, 0] >= x_mid]
        def width(group: np.ndarray) -> float | None:
            if len(group) < 5:
                return None
            return float(group[:, 0].max() - group[:, 0].min())
        return {"left_width_m": width(left), "right_width_m": width(right)}

    return {
        "ankle_above_floor": width_at(0.08),
        "calf_maximum_region": width_at(0.22),
        "below_knee": width_at(0.34),
        "knee_center": width_at(0.40),
        "exposed_thigh_under_shorts": width_at(0.55),
        "notes": "approximate; for debugging only",
    }


def save_overlay(
    rgb_path: Path | None,
    out_path: Path,
    vertices: np.ndarray,
    k: np.ndarray,
    label: str,
    skin_uv: np.ndarray | None = None,
    cloth_uv: np.ndarray | None = None,
    shoe_uv: np.ndarray | None = None,
) -> None:
    from PIL import Image, ImageDraw

    if rgb_path is not None and rgb_path.exists():
        image = Image.open(rgb_path).convert("RGB")
    else:
        image = Image.new("RGB", (int(k[0, 2] * 2), int(k[1, 2] * 2)), "white")
    draw = ImageDraw.Draw(image)
    idx = np.linspace(0, len(vertices) - 1, min(2000, len(vertices))).astype(np.int64)
    uv = project_smpl_y_up_to_image(vertices[idx], k)
    for u, v in uv:
        if 0 <= u < image.width and 0 <= v < image.height:
            draw.point((float(u), float(v)), fill=(255, 0, 0))
    for pts, color in ((skin_uv, (0, 255, 0)), (cloth_uv, (0, 80, 255)), (shoe_uv, (255, 150, 0))):
        if pts is None:
            continue
        for u, v in pts[:2000]:
            if 0 <= u < image.width and 0 <= v < image.height:
                draw.point((float(u), float(v)), fill=color)
    draw.text((18, 18), label, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

