"""II-14 diagnostic armor overlay from 4-view canonical SMPL meshes.

This is a visualization/export helper, not a new SMPL fitting stage. It assigns
each SMPL face to a source view by the final single-SMPL surface normal, then
draws the corresponding canonical 4-view mesh surface as colored "armor" over
the final single SMPL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .ii12_canonical_fusion import (
    VIEW_COLORS,
    VIEW_ORDER,
    YAW_PRESETS,
    apply_y_alignment,
    load_obj,
    render_projection_png,
    rot_y,
    view_key,
    write_multi_mesh_ply,
    write_ply,
)
from .ii13_single_refine import canonicalize_view_mesh


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


def face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = face_normals(verts, faces)
    vn = np.zeros_like(verts)
    for j in range(3):
        np.add.at(vn, faces[:, j], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def source_for_normals(normals: np.ndarray) -> np.ndarray:
    """Assign horizontal outward normals to front/right/back/left source views.

    Canonical frame convention here is empirical/diagnostic:
    - +Z dominant -> front
    - -Z dominant -> back
    - +X dominant -> right
    - -X dominant -> left
    """
    x = normals[:, 0]
    z = normals[:, 2]
    out = np.empty(len(normals), dtype=object)
    z_dom = np.abs(z) >= np.abs(x)
    out[z_dom & (z >= 0)] = "front"
    out[z_dom & (z < 0)] = "back"
    out[~z_dom & (x >= 0)] = "right"
    out[~z_dom & (x < 0)] = "left"
    return out


def load_canonical_targets(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], np.ndarray]:
    input_dir = Path(args.input_dir)
    params = np.load(Path(args.final_params), allow_pickle=True)
    yaw_deg = YAW_PRESETS[args.yaw_preset]
    view_names = [str(v) for v in params["view_names"]]
    view_to_index = {view_key(name): i for i, name in enumerate(view_names)}
    targets: dict[str, np.ndarray] = {}
    faces_ref: np.ndarray | None = None
    for view in VIEW_ORDER:
        verts, faces = load_obj(input_dir / "meshes" / f"final_{view}.obj")
        if faces_ref is None:
            faces_ref = faces
        elif not np.array_equal(faces_ref, faces):
            raise ValueError(f"Face topology differs for {view}")
        idx = view_to_index[view]
        targets[view] = canonicalize_view_mesh(
            verts,
            params["transl_final"][idx].astype(np.float64),
            view,
            args.rotation_pivot,
            yaw_deg,
        )
    targets, _ = apply_y_alignment(targets, args.y_align)
    assert faces_ref is not None
    return targets, faces_ref


def make_face_duplicated_mesh(
    faces: np.ndarray,
    targets: dict[str, np.ndarray],
    face_sources: np.ndarray,
    final_normals: np.ndarray,
    offset_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts_out = []
    faces_out = []
    cols_out = []
    for fi, face in enumerate(faces):
        source = str(face_sources[fi])
        tri = targets[source][face].copy()
        tri += final_normals[fi][None, :] * offset_m
        start = len(verts_out)
        verts_out.extend(tri.tolist())
        faces_out.append([start, start + 1, start + 2])
        cols_out.extend([VIEW_COLORS[source]] * 3)
    return (
        np.asarray(verts_out, dtype=np.float64),
        np.asarray(faces_out, dtype=np.int32),
        np.asarray(cols_out, dtype=np.uint8),
    )


def write_colored_ply(path: Path, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    write_ply(path, verts, faces, colors)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_verts, final_faces = load_ascii_ply_mesh(Path(args.final_single_ply))
    targets, target_faces = load_canonical_targets(args)
    if not np.array_equal(final_faces, target_faces):
        raise ValueError("Final single and target meshes have different topology")

    fn = face_normals(final_verts, final_faces)
    vn = vertex_normals(final_verts, final_faces)
    face_sources = source_for_normals(fn)
    vertex_sources = source_for_normals(vn)

    vertex_colors = np.asarray([VIEW_COLORS[str(src)] for src in vertex_sources], dtype=np.uint8)
    write_colored_ply(out_dir / "final_single_view_source_assignment.ply", final_verts, final_faces, vertex_colors)

    armor_v, armor_f, armor_c = make_face_duplicated_mesh(
        final_faces,
        targets,
        face_sources,
        fn,
        args.offset_m,
    )
    write_colored_ply(out_dir / "armor_4view_surface_on_smpl.ply", armor_v, armor_f, armor_c)
    write_multi_mesh_ply(
        out_dir / "final_single_gray_plus_armor_overlay.ply",
        [
            (final_verts, final_faces, (185, 185, 185)),
            (armor_v, armor_f, (255, 255, 255)),
        ],
    )
    # The combined helper colors each mesh uniformly, so also write a fully colored
    # overlay where the armor retains per-face source colors.
    combined_v = np.concatenate([final_verts, armor_v], axis=0)
    combined_f = np.concatenate([final_faces, armor_f + len(final_verts)], axis=0)
    combined_c = np.concatenate(
        [
            np.tile(np.asarray((185, 185, 185), dtype=np.uint8), (len(final_verts), 1)),
            armor_c,
        ],
        axis=0,
    )
    write_colored_ply(out_dir / "final_single_gray_plus_armor_overlay_colored.ply", combined_v, combined_f, combined_c)

    render_items = {view: targets[view] for view in VIEW_ORDER}
    render_dir = out_dir / "renders"
    render_paths = {}
    for plane in ("xz_top", "xy_front", "zy_side"):
        path = render_dir / f"{plane}_armor_sources.png"
        render_projection_png(path, render_items, final_verts, plane)
        render_paths[plane] = str(path.resolve())

    counts = {view: int((face_sources == view).sum()) for view in VIEW_ORDER}
    report = {
        "schema_version": "ii14_armor_overlay_v1",
        "description": "Diagnostic 4-view source surface armor overlay on final single SMPL.",
        "input_dir": str(Path(args.input_dir)),
        "final_single_ply": str(Path(args.final_single_ply)),
        "final_params": str(Path(args.final_params)),
        "yaw_preset": args.yaw_preset,
        "y_align": args.y_align,
        "offset_m": args.offset_m,
        "face_source_counts": counts,
        "face_source_ratio": {view: counts[view] / float(len(final_faces)) for view in VIEW_ORDER},
        "outputs": {
            "source_assignment": str((out_dir / "final_single_view_source_assignment.ply").resolve()),
            "armor_surface": str((out_dir / "armor_4view_surface_on_smpl.ply").resolve()),
            "overlay_colored": str((out_dir / "final_single_gray_plus_armor_overlay_colored.ply").resolve()),
            "render_paths": render_paths,
        },
    }
    with (out_dir / "armor_overlay_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[done] wrote {out_dir}")
    print("  face counts:", counts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="outputs/ii10_pr7_strict_full_final")
    p.add_argument("--final-params", default="outputs/ii10_pr7_strict_full_final/final_smpl_params.npz")
    p.add_argument("--final-single-ply", default="outputs/ii13_single_refine_pr7_yawonly/final_single_smpl_canonical.ply")
    p.add_argument("--output-dir", default="outputs/ii14_armor_overlay_pr7_yawonly")
    p.add_argument("--rotation-pivot", choices=("transl", "bbox_center", "origin"), default="transl")
    p.add_argument("--yaw-preset", choices=tuple(YAW_PRESETS.keys()), default="inverse_sides")
    p.add_argument("--y-align", choices=("none", "floor", "center"), default="floor")
    p.add_argument("--offset-m", type=float, default=0.006)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
