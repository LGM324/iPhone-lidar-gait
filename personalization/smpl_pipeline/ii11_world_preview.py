"""II-11 world-frame preview for STAND 4-view SMPL outputs.

This script does not fuse SMPL parameters. It exports diagnostic world-frame
PLYs from the final II-10 view meshes so camera/world coordinate conventions can
be checked before choosing a single canonical SMPL.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np


VIEWS = ("front", "right", "back", "left")
VIEW_COLORS = {
    "front": (255, 70, 70),
    "right": (70, 220, 80),
    "back": (70, 110, 255),
    "left": (240, 80, 240),
}
DEPTH_COLORS = {
    "front": (150, 150, 150),
    "right": (170, 170, 170),
    "back": (130, 130, 130),
    "left": (190, 190, 190),
}
NOMINAL_YAW_DEG = {"front": 0.0, "right": -90.0, "back": 180.0, "left": 90.0}


def rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def cam_to_world(pts: np.ndarray, t_cw: np.ndarray, yaw_deg: float) -> np.ndarray:
    r, t = t_cw[:3, :3], t_cw[:3, 3]
    p0 = (r @ pts.T).T + t
    return (rot_y(math.radians(yaw_deg)) @ p0.T).T


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                raw = line.split()[1:]
                idx = []
                for item in raw:
                    idx.append(int(item.split("/")[0]) - 1)
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    if not verts or not faces:
        raise ValueError(f"OBJ has no vertices/faces: {path}")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def load_ply_vertices(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load x/y/z and optional RGB from simple Open3D-style PLY files."""
    with path.open("rb") as f:
        header = b""
        while b"end_header\n" not in header:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header += line
        text = header.decode("ascii", errors="replace")
        lines = text.splitlines()
        fmt = next((line.split()[1] for line in lines if line.startswith("format ")), None)
        n = next((int(line.split()[2]) for line in lines if line.startswith("element vertex")), None)
        if n is None:
            raise ValueError(f"No vertex count in PLY: {path}")
        vertex_props: list[tuple[str, str]] = []
        in_vertex = False
        for line in lines:
            if line.startswith("element vertex"):
                in_vertex = True
                continue
            if line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex = False
            if in_vertex and line.startswith("property "):
                _, typ, name = line.split()[:3]
                vertex_props.append((typ, name))

        if fmt == "ascii":
            pts = []
            cols = []
            has_rgb = {"red", "green", "blue"}.issubset({name for _, name in vertex_props})
            names = [name for _, name in vertex_props]
            for _ in range(n):
                parts = f.readline().decode("ascii", errors="replace").split()
                row = {name: parts[i] for i, name in enumerate(names)}
                pts.append([float(row["x"]), float(row["y"]), float(row["z"])])
                if has_rgb:
                    cols.append([int(row["red"]), int(row["green"]), int(row["blue"])])
            return np.asarray(pts, dtype=np.float64), np.asarray(cols, dtype=np.uint8) if has_rgb else None

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format {fmt}: {path}")

        type_map = {
            "char": ("i1", 1),
            "uchar": ("u1", 1),
            "short": ("<i2", 2),
            "ushort": ("<u2", 2),
            "int": ("<i4", 4),
            "uint": ("<u4", 4),
            "float": ("<f4", 4),
            "double": ("<f8", 8),
        }
        dtype = np.dtype([(name, type_map[typ][0]) for typ, name in vertex_props])
        arr = np.fromfile(f, dtype=dtype, count=n)
        pts = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
        if {"red", "green", "blue"}.issubset(arr.dtype.names or ()):
            cols = np.column_stack([arr["red"], arr["green"], arr["blue"]]).astype(np.uint8)
            return pts, cols
        return pts, None


def write_ply(
    path: Path,
    verts: np.ndarray,
    faces: np.ndarray | None = None,
    colors: np.ndarray | tuple[int, int, int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(verts, dtype=np.float64)
    has_color = colors is not None
    if colors is None:
        color_arr = None
    elif isinstance(colors, tuple):
        color_arr = np.tile(np.asarray(colors, dtype=np.uint8), (len(verts), 1))
    else:
        color_arr = np.asarray(colors, dtype=np.uint8)
        if len(color_arr) != len(verts):
            raise ValueError("color count does not match vertex count")

    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if faces is not None:
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        if has_color:
            for p, c in zip(verts, color_arr):
                f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for p in verts:
                f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")
        if faces is not None:
            for tri in faces:
                f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def subsample_points(pts: np.ndarray, cols: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(pts) <= max_points:
        return pts, cols
    idx = np.linspace(0, len(pts) - 1, num=max_points, dtype=np.int64)
    return pts[idx], cols[idx]


def bbox_report(verts: np.ndarray) -> dict:
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = (lo + hi) * 0.5
    size = hi - lo
    return {
        "min_xyz": lo.tolist(),
        "max_xyz": hi.tolist(),
        "center_xyz": center.tolist(),
        "size_xyz": size.tolist(),
        "height_y_m": float(size[1]),
    }


def center_distance_report(items: dict[str, np.ndarray]) -> dict:
    centers = {k: ((v.min(axis=0) + v.max(axis=0)) * 0.5) for k, v in items.items()}
    out: dict[str, float] = {}
    for i, a in enumerate(VIEWS):
        for b in VIEWS[i + 1 :]:
            d = centers[a] - centers[b]
            out[f"{a}_{b}_3d_m"] = float(np.linalg.norm(d))
            out[f"{a}_{b}_xz_m"] = float(np.linalg.norm(d[[0, 2]]))
            out[f"{a}_{b}_y_m"] = float(abs(d[1]))
    return out


def make_variant_fns() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "as_is": lambda p: p.copy(),
        "smpl_yflip_to_opencv": lambda p: p * np.array([1.0, -1.0, 1.0], dtype=np.float64),
        "viewer_yflip_zsame": lambda p: p * np.array([1.0, -1.0, 1.0], dtype=np.float64),
        "smpl_yflip_zflip": lambda p: p * np.array([1.0, -1.0, -1.0], dtype=np.float64),
    }


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    t_cw = np.load(Path(args.cam2world)).astype(np.float64)
    if t_cw.shape != (4, 4):
        raise ValueError(f"cam2world must be 4x4, got {t_cw.shape}: {args.cam2world}")

    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for view in VIEWS:
        mesh_path = input_dir / "meshes" / f"final_{view}.obj"
        meshes[view] = load_obj(mesh_path)

    depth_world: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if args.world_pcd_dir:
        world_pcd_dir = Path(args.world_pcd_dir)
        for view in VIEWS:
            pcd_path = world_pcd_dir / f"{view}_depth_filt_world.ply"
            if pcd_path.exists():
                pts, cols = load_ply_vertices(pcd_path)
                if cols is None:
                    cols = np.tile(np.asarray(DEPTH_COLORS[view], dtype=np.uint8), (len(pts), 1))
                depth_world[view] = (pts, cols)

    report = {
        "schema_version": "ii11_world_preview_v1",
        "input_dir": str(input_dir),
        "cam2world": str(Path(args.cam2world)),
        "nominal_yaw_deg": NOMINAL_YAW_DEG,
        "world_pcd_dir": str(Path(args.world_pcd_dir)) if args.world_pcd_dir else None,
        "note": (
            "Diagnostic only. These PLYs validate world alignment before any "
            "single-SMPL parameter fusion."
        ),
        "variants": {},
    }

    for variant_name, pre_fn in make_variant_fns().items():
        variant_dir = out_dir / variant_name
        variant_items: dict[str, np.ndarray] = {}
        overlay_depth_v: list[np.ndarray] = []
        overlay_depth_c: list[np.ndarray] = []
        combined_v: list[np.ndarray] = []
        combined_f: list[np.ndarray] = []
        combined_c: list[np.ndarray] = []
        vertex_offset = 0
        variant_report = {"views": {}}

        for view in VIEWS:
            verts, faces = meshes[view]
            pts_cam = pre_fn(verts)
            pts_world = cam_to_world(pts_cam, t_cw, NOMINAL_YAW_DEG[view])
            variant_items[view] = pts_world
            color = VIEW_COLORS[view]
            write_ply(variant_dir / f"smpl_world_{view}.ply", pts_world, faces, color)
            variant_report["views"][view] = bbox_report(pts_world)
            if view in depth_world:
                depth_pts, depth_cols = depth_world[view]
                depth_pts_sub, depth_cols_sub = subsample_points(
                    depth_pts, depth_cols, max(1, args.max_depth_points_per_view)
                )
                write_ply(
                    variant_dir / f"{view}_depth_vs_smpl_overlay.ply",
                    np.concatenate([depth_pts_sub, pts_world], axis=0),
                    faces + len(depth_pts_sub),
                    np.concatenate(
                        [
                            depth_cols_sub,
                            np.tile(np.asarray(color, dtype=np.uint8), (len(pts_world), 1)),
                        ],
                        axis=0,
                    ),
                )
                depth_rep = bbox_report(depth_pts)
                smpl_center = np.asarray(variant_report["views"][view]["center_xyz"], dtype=np.float64)
                depth_center = np.asarray(depth_rep["center_xyz"], dtype=np.float64)
                variant_report["views"][view]["depth_world_bbox"] = depth_rep
                variant_report["views"][view]["smpl_to_depth_center_delta_xyz_m"] = (
                    smpl_center - depth_center
                ).tolist()
                variant_report["views"][view]["smpl_to_depth_center_dist_3d_m"] = float(
                    np.linalg.norm(smpl_center - depth_center)
                )
                variant_report["views"][view]["smpl_to_depth_center_dist_xz_m"] = float(
                    np.linalg.norm((smpl_center - depth_center)[[0, 2]])
                )
                overlay_depth_v.append(depth_pts_sub)
                overlay_depth_c.append(depth_cols_sub)

            combined_v.append(pts_world)
            combined_f.append(faces + vertex_offset)
            combined_c.append(np.tile(np.asarray(color, dtype=np.uint8), (len(pts_world), 1)))
            vertex_offset += len(pts_world)

        all_v = np.concatenate(combined_v, axis=0)
        all_f = np.concatenate(combined_f, axis=0)
        all_c = np.concatenate(combined_c, axis=0)
        write_ply(variant_dir / "smpl_4view_world_preview_colored.ply", all_v, all_f, all_c)
        if overlay_depth_v:
            depth_v = np.concatenate(overlay_depth_v, axis=0)
            depth_c = np.concatenate(overlay_depth_c, axis=0)
            write_ply(
                variant_dir / "depth_4view_vs_smpl_4view_overlay.ply",
                np.concatenate([depth_v, all_v], axis=0),
                all_f + len(depth_v),
                np.concatenate([depth_c, all_c], axis=0),
            )
        variant_report["combined"] = bbox_report(all_v)
        variant_report["pairwise_center_distances"] = center_distance_report(variant_items)
        variant_report["mean_pairwise_center_3d_m"] = float(
            np.mean(
                [
                    v
                    for k, v in variant_report["pairwise_center_distances"].items()
                    if k.endswith("_3d_m")
                ]
            )
        )
        variant_report["mean_pairwise_center_xz_m"] = float(
            np.mean(
                [
                    v
                    for k, v in variant_report["pairwise_center_distances"].items()
                    if k.endswith("_xz_m")
                ]
            )
        )
        depth_dists = [
            view_info["smpl_to_depth_center_dist_3d_m"]
            for view_info in variant_report["views"].values()
            if "smpl_to_depth_center_dist_3d_m" in view_info
        ]
        if depth_dists:
            variant_report["mean_smpl_to_depth_center_dist_3d_m"] = float(np.mean(depth_dists))
            variant_report["max_smpl_to_depth_center_dist_3d_m"] = float(np.max(depth_dists))
        report["variants"][variant_name] = variant_report

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "world_preview_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[done] wrote {out_dir}")
    for name, item in report["variants"].items():
        print(
            f"  {name:20s} combined_height_y={item['combined']['height_y_m']:.3f} "
            f"mean_center_3d={item['mean_pairwise_center_3d_m']:.3f} "
            f"mean_center_xz={item['mean_pairwise_center_xz_m']:.3f} "
            f"mean_depth_delta={item.get('mean_smpl_to_depth_center_dist_3d_m', float('nan')):.3f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        default="outputs/ii10_pr7_strict_full_final",
        help="II-10 output directory containing meshes/final_<view>.obj",
    )
    p.add_argument(
        "--cam2world",
        default="Tpose/[Result]02_smpl/[Stage1]/tpose/stage0_A_final/cam2world.npy",
        help="4x4 cam2world matrix used by the existing STAND visualization script",
    )
    p.add_argument("--output-dir", default="outputs/ii11_world_preview_pr7")
    p.add_argument(
        "--world-pcd-dir",
        default="Stand/[Result]02_smpl/[Stage1]/stand/visualizations/world_frame",
        help="Existing STAND world-frame depth PLY directory for SMPL/depth overlay diagnostics",
    )
    p.add_argument("--max-depth-points-per-view", type=int, default=50000)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
