"""II-16 depth point cloud and SMPL mesh overlay exports.

Exports:
1. one canonical-frame overlay: all four human depth point clouds + final single SMPL
2. four camera-frame overlays: each view human depth + that view's II-10 SMPL mesh
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .depth_utils import load_depth_observation
from .ii8_init import find_depth_path
from .ii12_canonical_fusion import YAW_PRESETS, apply_y_alignment, load_obj, rot_y, view_key, write_ply
from .ii13_single_refine import canonicalize_view_mesh
from .region_map import load_partseg
from .view_corrections import invert_camera_correction_np, load_view_corrections


VIEW_ORDER = ("front", "right", "back", "left")
DEPTH_COLORS = {
    "front": (255, 110, 110),
    "right": (90, 230, 100),
    "back": (90, 130, 255),
    "left": (245, 100, 245),
}
VIEW_SMPL_COLORS = {
    "front": (190, 35, 35),
    "right": (30, 165, 55),
    "back": (35, 70, 190),
    "left": (180, 45, 180),
}
FINAL_SMPL_COLOR = (255, 230, 45)


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


def write_overlay_ply(
    path: Path,
    point_sets: list[tuple[np.ndarray, tuple[int, int, int]]],
    mesh_sets: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
) -> None:
    verts_all = []
    faces_all = []
    colors_all = []
    offset = 0
    for pts, color in point_sets:
        pts = np.asarray(pts, dtype=np.float64)
        verts_all.append(pts)
        colors_all.append(np.tile(np.asarray(color, dtype=np.uint8), (len(pts), 1)))
        offset += len(pts)
    for verts, faces, color in mesh_sets:
        verts = np.asarray(verts, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        verts_all.append(verts)
        faces_all.append(faces + offset)
        colors_all.append(np.tile(np.asarray(color, dtype=np.uint8), (len(verts), 1)))
        offset += len(verts)
    all_verts = np.concatenate(verts_all, axis=0) if verts_all else np.zeros((0, 3), dtype=np.float64)
    all_colors = np.concatenate(colors_all, axis=0) if colors_all else None
    all_faces = np.concatenate(faces_all, axis=0) if faces_all else None
    write_ply(path, all_verts, all_faces, all_colors)


def points_to_surfels(
    pts: np.ndarray,
    color: tuple[int, int, int],
    *,
    size_m: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = sample_points(np.asarray(pts, dtype=np.float64), max_points)
    if len(pts) == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
        )
    half = size_m * 0.5
    offsets = np.asarray(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )
    verts = (pts[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    base = np.arange(len(pts), dtype=np.int32) * 4
    faces = np.stack(
        [
            np.stack([base, base + 1, base + 2], axis=1),
            np.stack([base, base + 2, base + 3], axis=1),
        ],
        axis=1,
    ).reshape(-1, 3)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(verts), 1))
    return verts, faces, colors


def write_surfel_overlay_ply(
    path: Path,
    point_sets: list[tuple[np.ndarray, tuple[int, int, int]]],
    mesh_sets: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
    *,
    surfel_size_m: float,
    max_surfel_points_per_set: int,
) -> None:
    verts_all = []
    faces_all = []
    colors_all = []
    offset = 0
    for pts, color in point_sets:
        sv, sf, sc = points_to_surfels(
            pts,
            color,
            size_m=surfel_size_m,
            max_points=max_surfel_points_per_set,
        )
        verts_all.append(sv)
        faces_all.append(sf + offset)
        colors_all.append(sc)
        offset += len(sv)
    for verts, faces, color in mesh_sets:
        verts = np.asarray(verts, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        verts_all.append(verts)
        faces_all.append(faces + offset)
        colors_all.append(np.tile(np.asarray(color, dtype=np.uint8), (len(verts), 1)))
        offset += len(verts)
    write_ply(
        path,
        np.concatenate(verts_all, axis=0),
        np.concatenate(faces_all, axis=0),
        np.concatenate(colors_all, axis=0),
    )


def sample_points(pts: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_points, dtype=np.int64)
    return pts[idx]


def body_depth_points(
    view: str,
    view_name: str,
    k: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    depth_path = find_depth_path(Path(args.depth_dir), view_name)
    obs = load_depth_observation(depth_path, k.astype(np.float32), args.depth_y_axis)
    regions = load_partseg(
        Path(args.partseg_dir),
        view_name,
        obs["valid_mask"],
        allow_png_label_fallback=args.allow_png_label_fallback,
    )
    if args.depth_region == "body_no_shoe":
        mask = (regions["skin_mask"] | regions["cloth_mask"]) & ~regions["shoe_mask"]
    elif args.depth_region == "foreground_no_shoe":
        mask = regions["fgmask"] & obs["valid_mask"] & ~regions["shoe_mask"]
    elif args.depth_region == "lower_body_no_shoe":
        labels = regions["label_map"]
        lower = np.isin(labels, [7, 11, 12, 16, 20])
        mask = lower & obs["valid_mask"] & ~regions["shoe_mask"]
    else:
        raise ValueError(f"Unsupported depth_region: {args.depth_region}")
    pts = obs["xyz_obs"][mask]
    pts = pts[np.isfinite(pts).all(axis=1) & (pts[:, 2] > 1e-4)]
    pts = sample_points(pts.astype(np.float64), args.max_depth_points_per_view)
    return pts, {
        "view": view,
        "view_name": view_name,
        "depth_path": str(depth_path),
        "region": args.depth_region,
        "points": int(len(pts)),
    }


def compute_y_alignment(
    ii10_dir: Path,
    final_params,
    view_to_index: dict[str, int],
    yaw_deg: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    targets = {}
    view_corrections = load_view_corrections(args.view_corrections_json)
    for view in VIEW_ORDER:
        verts, _ = load_obj(ii10_dir / "meshes" / f"final_{view}.obj")
        verts = invert_camera_correction_np(verts, view, view_corrections)
        idx = view_to_index[view]
        targets[view] = canonicalize_view_mesh(
            verts,
            final_params["transl_final"][idx].astype(np.float64),
            view,
            args.rotation_pivot,
            yaw_deg,
        )
    _, diag = apply_y_alignment(targets, args.y_align)
    return diag.get("shift_y_by_view_m", {}) if isinstance(diag, dict) else {}


def depth_cam_to_canonical(
    pts_cam: np.ndarray,
    view: str,
    final_params,
    view_to_index: dict[str, int],
    yaw_deg: dict[str, float],
    y_shifts: dict[str, float],
    args: argparse.Namespace,
) -> np.ndarray:
    idx = view_to_index[view]
    pts_cam = invert_camera_correction_np(pts_cam, view, load_view_corrections(args.view_corrections_json))
    if args.rotation_pivot == "transl":
        pivot = final_params["transl_final"][idx].astype(np.float64)
    elif args.rotation_pivot == "origin":
        pivot = np.zeros(3, dtype=np.float64)
    elif args.rotation_pivot == "bbox_center":
        # Depth does not have the SMPL bbox; transl is the safer approximate pivot.
        pivot = final_params["transl_final"][idx].astype(np.float64)
    else:
        raise ValueError(f"Unsupported rotation_pivot: {args.rotation_pivot}")
    r = rot_y(math.radians(yaw_deg[view]))
    pts = pivot + (r @ (pts_cam - pivot).T).T
    pts = pts + np.array([0.0, y_shifts.get(view, 0.0), 0.0], dtype=np.float64)
    return pts


def export_obj_pair(path: Path, depth_pts: np.ndarray, mesh_verts: np.ndarray, mesh_faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# depth points followed by mesh vertices/faces\n")
        for p in depth_pts:
            f.write(f"v {p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")
        mesh_offset = len(depth_pts)
        for p in mesh_verts:
            f.write(f"v {p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")
        for tri in mesh_faces:
            a, b, c = tri + 1 + mesh_offset
            f.write(f"f {int(a)} {int(b)} {int(c)}\n")


def run(args: argparse.Namespace) -> None:
    ii10_dir = Path(args.ii10_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_params = np.load(Path(args.final_params), allow_pickle=True)
    view_names = [str(v) for v in final_params["view_names"]]
    view_to_index = {view_key(name): i for i, name in enumerate(view_names)}
    yaw_deg = YAW_PRESETS[args.yaw_preset]
    y_shifts = compute_y_alignment(ii10_dir, final_params, view_to_index, yaw_deg, args)

    final_verts, final_faces = load_ascii_ply_mesh(Path(args.final_single_ply))
    point_sets_canonical = []
    report = {
        "schema_version": "ii16_depth_mesh_overlays_v1",
        "depth_region": args.depth_region,
        "yaw_preset": args.yaw_preset,
        "y_align": args.y_align,
        "view_corrections": load_view_corrections(args.view_corrections_json),
        "outputs": {},
        "views": {},
    }

    per_view_dir = out_dir / "per_view_depth_vs_view_smpl"
    for view in VIEW_ORDER:
        idx = view_to_index[view]
        view_name = view_names[idx]
        k = final_params["K"][idx].astype(np.float64)
        depth_pts_cam, depth_diag = body_depth_points(view, view_name, k, args)
        view_smpl_verts, view_smpl_faces = load_obj(ii10_dir / "meshes" / f"final_{view}.obj")

        ply_path = per_view_dir / f"{view}_depth_vs_view_smpl_overlay.ply"
        write_overlay_ply(
            ply_path,
            [(depth_pts_cam, DEPTH_COLORS[view])],
            [(view_smpl_verts, view_smpl_faces, VIEW_SMPL_COLORS[view])],
        )
        if args.write_obj:
            export_obj_pair(
                per_view_dir / f"{view}_depth_vs_view_smpl_overlay.obj",
                depth_pts_cam,
                view_smpl_verts,
                view_smpl_faces,
            )
        if args.write_visible_surfels:
            write_surfel_overlay_ply(
                per_view_dir / f"{view}_depth_surfels_vs_view_smpl_overlay.ply",
                [(depth_pts_cam, DEPTH_COLORS[view])],
                [(view_smpl_verts, view_smpl_faces, VIEW_SMPL_COLORS[view])],
                surfel_size_m=args.surfel_size_m,
                max_surfel_points_per_set=args.max_surfel_points_per_view,
            )

        depth_pts_can = depth_cam_to_canonical(
            depth_pts_cam, view, final_params, view_to_index, yaw_deg, y_shifts, args
        )
        point_sets_canonical.append((depth_pts_can, DEPTH_COLORS[view]))
        report["views"][view] = {
            **depth_diag,
            "per_view_overlay_ply": str(ply_path.resolve()),
        }

    final_overlay_path = out_dir / "all_view_depth_vs_final_single_smpl_overlay.ply"
    write_overlay_ply(
        final_overlay_path,
        point_sets_canonical,
        [(final_verts, final_faces, FINAL_SMPL_COLOR)],
    )
    if args.write_obj:
        all_depth = np.concatenate([p for p, _ in point_sets_canonical], axis=0)
        export_obj_pair(
            out_dir / "all_view_depth_vs_final_single_smpl_overlay.obj",
            all_depth,
            final_verts,
            final_faces,
        )
    if args.write_visible_surfels:
        write_surfel_overlay_ply(
            out_dir / "all_view_depth_surfels_vs_final_single_smpl_overlay.ply",
            point_sets_canonical,
            [(final_verts, final_faces, FINAL_SMPL_COLOR)],
            surfel_size_m=args.surfel_size_m,
            max_surfel_points_per_set=args.max_surfel_points_per_view,
        )

    report["outputs"]["all_view_depth_vs_final_single_smpl_overlay_ply"] = str(final_overlay_path.resolve())
    report["outputs"]["per_view_dir"] = str(per_view_dir.resolve())
    with (out_dir / "depth_mesh_overlay_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[done] wrote {out_dir}")
    print(f"  final overlay: {final_overlay_path}")
    for view in VIEW_ORDER:
        print(f"  {view}: {report['views'][view]['points']} depth pts")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ii10-dir", default="outputs/ii10_pr7_strict_full_final")
    p.add_argument("--final-params", default="outputs/ii10_pr7_strict_full_final/final_smpl_params.npz")
    p.add_argument("--final-single-ply", default="outputs/ii13_single_refine_pr7_yawonly/final_single_smpl_canonical.ply")
    p.add_argument("--output-dir", default="outputs/ii16_depth_mesh_overlays_pr7_yawonly")
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--partseg-dir", required=True)
    p.add_argument("--depth-y-axis", choices=("up", "down"), default="down")
    p.add_argument("--allow-png-label-fallback", action="store_true")
    p.add_argument("--depth-region", choices=("body_no_shoe", "foreground_no_shoe", "lower_body_no_shoe"), default="body_no_shoe")
    p.add_argument("--max-depth-points-per-view", type=int, default=90000)
    p.add_argument("--rotation-pivot", choices=("transl", "bbox_center", "origin"), default="transl")
    p.add_argument("--yaw-preset", choices=tuple(YAW_PRESETS.keys()), default="inverse_sides")
    p.add_argument("--y-align", choices=("none", "floor", "center"), default="floor")
    p.add_argument("--view-corrections-json", default=None)
    p.add_argument("--write-obj", action="store_true")
    p.add_argument("--write-visible-surfels", action="store_true")
    p.add_argument("--surfel-size-m", type=float, default=0.008)
    p.add_argument("--max-surfel-points-per-view", type=int, default=25000)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
