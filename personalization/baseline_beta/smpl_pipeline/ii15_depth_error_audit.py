"""II-15 depth-error audit for canonical SMPL / armor surfaces.

Projects a canonical mesh back into each source camera view, keeps the visible
model point per pixel with a simple z-buffer, and measures residuals against the
metric depth observation. The primary report focuses on lower-body labels while
excluding shoes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .depth_utils import load_depth_observation
from .ii8_init import find_depth_path, find_rgb_path
from .ii12_canonical_fusion import YAW_PRESETS, apply_y_alignment, load_obj, rot_y, view_key
from .ii13_single_refine import canonicalize_view_mesh
from .region_map import load_partseg
from .view_corrections import (
    apply_camera_correction_np,
    invert_camera_correction_np,
    load_view_corrections,
)


LOWER_BODY_LABELS = {7, 11, 12, 16, 20}
LEG_SKIN_LABELS = {7, 11, 16, 20}
LOWER_CLOTH_LABELS = {12}
SHOE_LABELS = {4, 8, 9, 13, 17, 18}


def load_ascii_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="ascii", errors="replace") as f:
        vertex_count = None
        face_count = None
        for line in f:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            elif line.startswith("element face"):
                face_count = int(line.split()[2])
            elif line.strip() == "end_header":
                break
        if vertex_count is None or face_count is None:
            raise ValueError(f"Missing vertex/face count in {path}")
        verts = []
        for _ in range(vertex_count):
            parts = f.readline().split()
            verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        faces = []
        for _ in range(face_count):
            parts = f.readline().split()
            if int(parts[0]) == 3:
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def project_y_up_np(points: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.maximum(points[:, 2], 1e-6)
    u = k[0, 0] * (points[:, 0] / z) + k[0, 2]
    v = k[1, 2] - k[1, 1] * (points[:, 1] / z)
    return u, v, z


def simple_zbuffer_indices(points_cam: np.ndarray, k: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    u, v, z = project_y_up_np(points_cam, k)
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h) & np.isfinite(z) & (z > 1e-4)
    idx = np.nonzero(inside)[0]
    if len(idx) == 0:
        return idx
    pix = y[idx] * w + x[idx]
    order = np.lexsort((z[idx], pix))
    idx_sorted = idx[order]
    pix_sorted = pix[order]
    keep = np.ones(len(idx_sorted), dtype=bool)
    keep[1:] = pix_sorted[1:] != pix_sorted[:-1]
    return idx_sorted[keep]


def stats(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "count": 0,
            "mean_m": None,
            "median_m": None,
            "p90_m": None,
            "p95_m": None,
            "rmse_m": None,
            "signed_mean_m": None,
        }
    return {
        "count": int(len(values)),
        "mean_m": float(np.mean(np.abs(values))),
        "median_m": float(np.median(np.abs(values))),
        "p90_m": float(np.percentile(np.abs(values), 90)),
        "p95_m": float(np.percentile(np.abs(values), 95)),
        "rmse_m": float(np.sqrt(np.mean(values**2))),
        "signed_mean_m": float(np.mean(values)),
    }


def distance_stats(values: np.ndarray, raw_count: int | None = None) -> dict:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "raw_count": int(raw_count or 0),
            "sampled_count": 0,
            "mean_m": None,
            "median_m": None,
            "p90_m": None,
            "p95_m": None,
            "p99_m": None,
            "rmse_m": None,
        }
    return {
        "raw_count": int(raw_count if raw_count is not None else len(values)),
        "sampled_count": int(len(values)),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p90_m": float(np.percentile(values, 90)),
        "p95_m": float(np.percentile(values, 95)),
        "p99_m": float(np.percentile(values, 99)),
        "rmse_m": float(np.sqrt(np.mean(values**2))),
    }


def sample_indices(count: int, max_count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_count <= 0 or count <= max_count:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_count, dtype=np.int64)


def make_distance_debug_png(
    path: Path,
    image_hw: tuple[int, int],
    pixels_xy: np.ndarray,
    distances: np.ndarray,
    rgb_path: Path | None,
    max_vis_m: float,
) -> None:
    from PIL import Image, ImageDraw

    h, w = image_hw
    if rgb_path is not None and rgb_path.exists():
        img = Image.open(rgb_path).convert("RGB").resize((w, h))
    else:
        img = Image.new("RGB", (w, h), (18, 18, 18))
    draw = ImageDraw.Draw(img, "RGBA")
    clipped = np.clip(distances, 0.0, max_vis_m)
    denom = max(max_vis_m, 1e-6)
    for (x, y), dd in zip(pixels_xy, clipped):
        # cyan = close, yellow/red = far
        t = float(dd / denom)
        r = int(40 + 215 * t)
        g = int(220 - 120 * max(t - 0.35, 0.0) / 0.65)
        b = int(255 * (1.0 - t))
        alpha = int(80 + 175 * t)
        draw.ellipse((int(x) - 1, int(y) - 1, int(x) + 1, int(y) + 1), fill=(r, g, b, alpha))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def make_debug_png(
    path: Path,
    image_hw: tuple[int, int],
    u: np.ndarray,
    v: np.ndarray,
    residual: np.ndarray,
    rgb_path: Path | None,
) -> None:
    from PIL import Image, ImageDraw

    h, w = image_hw
    if rgb_path is not None and rgb_path.exists():
        img = Image.open(rgb_path).convert("RGB").resize((w, h))
    else:
        img = Image.new("RGB", (w, h), (18, 18, 18))
    draw = ImageDraw.Draw(img, "RGBA")
    clipped = np.clip(residual, -0.20, 0.20)
    for uu, vv, rr in zip(u, v, clipped):
        x = int(round(float(uu)))
        y = int(round(float(vv)))
        if rr >= 0:
            alpha = int(70 + 185 * min(abs(rr) / 0.20, 1.0))
            color = (255, 40, 40, alpha)
        else:
            alpha = int(70 + 185 * min(abs(rr) / 0.20, 1.0))
            color = (40, 120, 255, alpha)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def load_final_params(path: Path):
    return np.load(path, allow_pickle=True)


def canonical_to_view_camera(
    verts_canonical: np.ndarray,
    view: str,
    final_params,
    view_to_index: dict[str, int],
    yaw_deg: dict[str, float],
    y_shifts: dict[str, float],
    view_corrections: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    idx = view_to_index[view]
    pivot = final_params["transl_final"][idx].astype(np.float64)
    shift = np.array([0.0, y_shifts.get(view, 0.0), 0.0], dtype=np.float64)
    r = rot_y(math.radians(yaw_deg[view]))
    pts = pivot + (r.T @ (verts_canonical - shift - pivot).T).T
    return apply_camera_correction_np(pts, view, view_corrections or {})


def view_camera_to_canonical(
    pts_cam: np.ndarray,
    view: str,
    final_params,
    view_to_index: dict[str, int],
    yaw_deg: dict[str, float],
    y_shifts: dict[str, float],
    view_corrections: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    pts_cam = invert_camera_correction_np(pts_cam, view, view_corrections or {})
    idx = view_to_index[view]
    pivot = final_params["transl_final"][idx].astype(np.float64)
    shift = np.array([0.0, y_shifts.get(view, 0.0), 0.0], dtype=np.float64)
    r = rot_y(math.radians(yaw_deg[view]))
    return pivot + (r @ (pts_cam - pivot).T).T + shift


def build_o3d_distance_scene(verts: np.ndarray, faces: np.ndarray):
    import open3d as o3d

    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex["positions"] = o3d.core.Tensor(verts.astype(np.float32))
    mesh.triangle["indices"] = o3d.core.Tensor(faces.astype(np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def scene_distances(scene, pts: np.ndarray) -> np.ndarray:
    import open3d as o3d

    if len(pts) == 0:
        return np.zeros((0,), dtype=np.float64)
    query = o3d.core.Tensor(pts.astype(np.float32))
    return scene.compute_distance(query).numpy().astype(np.float64)


def compute_y_alignment(
    input_dir: Path,
    final_params,
    view_to_index: dict[str, int],
    yaw_deg: dict[str, float],
    y_align: str,
    rotation_pivot: str,
    view_corrections: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    targets = {}
    for view in ("front", "right", "back", "left"):
        verts, _ = load_obj(input_dir / "meshes" / f"final_{view}.obj")
        verts = invert_camera_correction_np(verts, view, view_corrections or {})
        idx = view_to_index[view]
        targets[view] = canonicalize_view_mesh(
            verts,
            final_params["transl_final"][idx].astype(np.float64),
            view,
            rotation_pivot,
            yaw_deg,
        )
    _, diag = apply_y_alignment(targets, y_align)
    return diag.get("shift_y_by_view_m", {}) if isinstance(diag, dict) else {}


def measure_mesh(args: argparse.Namespace, mesh_path: Path, out_dir: Path) -> dict:
    final_params = load_final_params(Path(args.final_params))
    view_names = [str(v) for v in final_params["view_names"]]
    view_to_index = {view_key(name): i for i, name in enumerate(view_names)}
    k_all = final_params["K"].astype(np.float64)
    yaw_deg = YAW_PRESETS[args.yaw_preset]
    view_corrections = load_view_corrections(args.view_corrections_json)
    y_shifts = compute_y_alignment(
        Path(args.input_dir),
        final_params,
        view_to_index,
        yaw_deg,
        args.y_align,
        args.rotation_pivot,
        view_corrections,
    )
    verts_canonical, faces = load_ascii_ply_mesh(mesh_path)
    distance_scene = build_o3d_distance_scene(verts_canonical, faces)
    report = {
        "mesh": str(mesh_path),
        "view_reports": {},
        "aggregate": {},
        "point_to_mesh_distance": {
            "description": (
                "Unsigned nearest distance from observed depth points to the canonical mesh surface, in meters. "
                "Depth points are canonicalized with the same yaw preset and y alignment used by final fitting."
            ),
            "max_points_per_region": args.max_pointmesh_points_per_region,
            "views": {},
            "aggregate": {},
        },
    }
    agg = {"lower_body_no_shoe": [], "leg_skin_only": [], "lower_cloth": []}
    pointmesh_agg = {"lower_body_no_shoe": [], "leg_skin_only": [], "lower_cloth": []}
    pointmesh_raw_counts = {"lower_body_no_shoe": 0, "leg_skin_only": 0, "lower_cloth": 0}
    for view in ("front", "right", "back", "left"):
        idx = view_to_index[view]
        view_name = view_names[idx]
        k = k_all[idx]
        depth_path = find_depth_path(Path(args.depth_dir), view_name)
        obs = load_depth_observation(depth_path, k.astype(np.float32), args.depth_y_axis)
        image_hw = obs["valid_mask"].shape
        regions = load_partseg(
            Path(args.partseg_dir),
            view_name,
            obs["valid_mask"],
            allow_png_label_fallback=args.allow_png_label_fallback,
        )
        label_map = regions["label_map"]
        pts_cam = canonical_to_view_camera(
            verts_canonical, view, final_params, view_to_index, yaw_deg, y_shifts, view_corrections
        )
        keep_idx = simple_zbuffer_indices(pts_cam, k, image_hw)
        pts = pts_cam[keep_idx]
        u, v, z_model = project_y_up_np(pts, k)
        x = np.rint(u).astype(np.int64)
        y = np.rint(v).astype(np.int64)
        h, w = image_hw
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x = x[inside]
        y = y[inside]
        u = u[inside]
        v = v[inside]
        z_model = z_model[inside]
        valid = obs["valid_mask"][y, x]
        labels = label_map[y, x]
        z_obs = obs["z_obs"][y, x]
        residual = z_model - z_obs
        plausible = valid & np.isfinite(residual) & (np.abs(residual) <= args.max_abs_residual_m)
        masks = {
            "lower_body_no_shoe": plausible & np.isin(labels, list(LOWER_BODY_LABELS)) & ~np.isin(labels, list(SHOE_LABELS)),
            "leg_skin_only": plausible & np.isin(labels, list(LEG_SKIN_LABELS)),
            "lower_cloth": plausible & np.isin(labels, list(LOWER_CLOTH_LABELS)),
        }
        view_report = {
            "view_name": view_name,
            "depth_path": str(depth_path),
            "visible_projected_points": int(len(keep_idx)),
            "valid_projected_points": int((valid & np.isfinite(residual)).sum()),
            "metrics": {},
        }
        for name, mask in masks.items():
            vals = residual[mask]
            view_report["metrics"][name] = stats(vals)
            agg[name].append(vals)
            if name == "lower_body_no_shoe":
                rgb = find_rgb_path(Path(args.rgb_dir), view_name, f"{view_name}.png") if args.rgb_dir else None
                make_debug_png(
                    out_dir / "debug" / f"{view}_{name}_residual.png",
                    image_hw,
                    u[mask],
                    v[mask],
                    vals,
                    rgb,
                )
        report["view_reports"][view] = view_report

        full_labels = label_map
        full_valid = obs["valid_mask"] & np.isfinite(obs["xyz_obs"]).all(axis=-1)
        full_masks = {
            "lower_body_no_shoe": full_valid
            & np.isin(full_labels, list(LOWER_BODY_LABELS))
            & ~np.isin(full_labels, list(SHOE_LABELS)),
            "leg_skin_only": full_valid & np.isin(full_labels, list(LEG_SKIN_LABELS)),
            "lower_cloth": full_valid & np.isin(full_labels, list(LOWER_CLOTH_LABELS)),
        }
        pointmesh_view = {
            "view_name": view_name,
            "depth_path": str(depth_path),
            "metrics": {},
        }
        rgb = find_rgb_path(Path(args.rgb_dir), view_name, f"{view_name}.png") if args.rgb_dir else None
        for name, mask2d in full_masks.items():
            yy, xx = np.nonzero(mask2d)
            raw_count = int(len(xx))
            take = sample_indices(raw_count, args.max_pointmesh_points_per_region)
            pts_cam_depth = obs["xyz_obs"][yy[take], xx[take]].astype(np.float64)
            pts_canonical = view_camera_to_canonical(
                pts_cam_depth, view, final_params, view_to_index, yaw_deg, y_shifts, view_corrections
            )
            distances = scene_distances(distance_scene, pts_canonical)
            keep = np.isfinite(distances) & (distances <= args.max_pointmesh_distance_m)
            distances_kept = distances[keep]
            pointmesh_view["metrics"][name] = distance_stats(distances_kept, raw_count=raw_count)
            pointmesh_view["metrics"][name]["discarded_over_max_distance"] = int((~keep).sum())
            pointmesh_agg[name].append(distances_kept)
            pointmesh_raw_counts[name] += raw_count
            if name == "lower_body_no_shoe":
                pixels = np.stack([xx[take][keep], yy[take][keep]], axis=1) if len(take) else np.zeros((0, 2))
                make_distance_debug_png(
                    out_dir / "debug" / f"{view}_{name}_point_to_mesh_distance.png",
                    image_hw,
                    pixels,
                    distances_kept,
                    rgb,
                    args.distance_debug_max_m,
                )
        report["point_to_mesh_distance"]["views"][view] = pointmesh_view
    for name, chunks in agg.items():
        values = np.concatenate([c for c in chunks if len(c) > 0]) if any(len(c) > 0 for c in chunks) else np.array([])
        report["aggregate"][name] = stats(values)
    for name, chunks in pointmesh_agg.items():
        values = np.concatenate([c for c in chunks if len(c) > 0]) if any(len(c) > 0 for c in chunks) else np.array([])
        report["point_to_mesh_distance"]["aggregate"][name] = distance_stats(
            values, raw_count=pointmesh_raw_counts[name]
        )
    return report


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_paths = {
        "armor": Path(args.armor_mesh),
        "final_single": Path(args.final_single_mesh),
    }
    full_report = {
        "schema_version": "ii15_depth_error_audit_v1",
        "description": (
            "Projected z residuals in meters: z_model - z_observed. Positive means model is farther from camera. "
            "Also includes point-to-mesh distance from observed depth points to the canonical mesh surface."
        ),
        "lower_body_labels": sorted(LOWER_BODY_LABELS),
        "leg_skin_labels": sorted(LEG_SKIN_LABELS),
        "shoe_labels_excluded": sorted(SHOE_LABELS),
        "max_abs_residual_m": args.max_abs_residual_m,
        "yaw_preset": args.yaw_preset,
        "y_align": args.y_align,
        "view_corrections": load_view_corrections(args.view_corrections_json),
        "meshes": {},
    }
    for name, path in mesh_paths.items():
        full_report["meshes"][name] = measure_mesh(args, path, out_dir / name)
    with (out_dir / "depth_error_report.json").open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"[done] wrote {out_dir}")
    for mesh_name, mesh_report in full_report["meshes"].items():
        agg = mesh_report["aggregate"]["lower_body_no_shoe"]
        print(
            f"  {mesh_name:12s} lower_body_no_shoe n={agg['count']} "
            f"mean={agg['mean_m']} median={agg['median_m']} p95={agg['p95_m']} rmse={agg['rmse_m']}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="outputs/ii10_pr7_strict_full_final")
    p.add_argument("--final-params", default="outputs/ii10_pr7_strict_full_final/final_smpl_params.npz")
    p.add_argument("--armor-mesh", default="outputs/ii14_armor_overlay_pr7_yawonly/armor_4view_surface_on_smpl.ply")
    p.add_argument("--final-single-mesh", default="outputs/ii13_single_refine_pr7_yawonly/final_single_smpl_canonical.ply")
    p.add_argument("--output-dir", default="outputs/ii15_depth_error_armor_pr7_yawonly")
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--partseg-dir", required=True)
    p.add_argument("--rgb-dir", default=None)
    p.add_argument("--depth-y-axis", choices=("up", "down"), default="down")
    p.add_argument("--allow-png-label-fallback", action="store_true")
    p.add_argument("--rotation-pivot", choices=("transl", "bbox_center", "origin"), default="transl")
    p.add_argument("--yaw-preset", choices=tuple(YAW_PRESETS.keys()), default="inverse_sides")
    p.add_argument("--y-align", choices=("none", "floor", "center"), default="floor")
    p.add_argument("--view-corrections-json", default=None)
    p.add_argument("--max-abs-residual-m", type=float, default=0.50)
    p.add_argument("--max-pointmesh-points-per-region", type=int, default=50000)
    p.add_argument("--max-pointmesh-distance-m", type=float, default=0.50)
    p.add_argument("--distance-debug-max-m", type=float, default=0.30)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
