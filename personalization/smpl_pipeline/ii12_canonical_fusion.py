"""II-12 canonical camera-frame fusion preview for STAND 4-view SMPL.

The intended assumption is that the four STAND views share the same vertical
axis and differ primarily by yaw. This script canonicalizes each II-10 view to a
front-camera-like coordinate frame using yaw about each SMPL root/translation
pivot, then exports diagnostics and a conservative single-SMPL parameter
candidate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


VIEW_ORDER = ("front", "right", "back", "left")
VIEW_COLORS = {
    "front": (255, 70, 70),
    "right": (70, 220, 80),
    "back": (70, 110, 255),
    "left": (240, 80, 240),
}
YAW_PRESETS = {
    "stand_world": {"front": 0.0, "right": -90.0, "back": 180.0, "left": 90.0},
    "inverse_sides": {"front": 0.0, "right": 90.0, "back": 180.0, "left": -90.0},
}
CANONICAL_YAW_DEG = YAW_PRESETS["inverse_sides"]


def view_key(name: str) -> str:
    low = name.lower()
    for key in VIEW_ORDER:
        if key in low:
            return key
    raise ValueError(f"Cannot infer view key from {name!r}")


def rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def axis_angle_to_matrix(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vec / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def matrix_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(mat) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if abs(math.pi - theta) < 1e-5:
        vals, vecs = np.linalg.eigh((mat + np.eye(3)) * 0.5)
        axis = vecs[:, int(np.argmax(vals))]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    axis = np.array(
        [mat[2, 1] - mat[1, 2], mat[0, 2] - mat[2, 0], mat[1, 0] - mat[0, 1]],
        dtype=np.float64,
    )
    axis /= 2.0 * math.sin(theta)
    return axis * theta


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
                idx = [int(item.split("/")[0]) - 1 for item in raw]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    if not verts or not faces:
        raise ValueError(f"OBJ has no vertices/faces: {path}")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def write_ply(
    path: Path,
    verts: np.ndarray,
    faces: np.ndarray | None = None,
    colors: np.ndarray | tuple[int, int, int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(verts, dtype=np.float64)
    if colors is None:
        color_arr = None
    elif isinstance(colors, tuple):
        color_arr = np.tile(np.asarray(colors, dtype=np.uint8), (len(verts), 1))
    else:
        color_arr = np.asarray(colors, dtype=np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if color_arr is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if faces is not None:
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        if color_arr is None:
            for p in verts:
                f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")
        else:
            for p, c in zip(verts, color_arr):
                f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        if faces is not None:
            for tri in faces:
                f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def write_multi_mesh_ply(
    path: Path,
    meshes: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
) -> None:
    verts_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []
    offset = 0
    for verts, faces, color in meshes:
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


def bbox_report(verts: np.ndarray) -> dict:
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = (lo + hi) * 0.5
    return {
        "min_xyz": lo.tolist(),
        "max_xyz": hi.tolist(),
        "center_xyz": center.tolist(),
        "size_xyz": (hi - lo).tolist(),
        "height_y_m": float(hi[1] - lo[1]),
    }


def pairwise_center_report(items: dict[str, np.ndarray]) -> dict:
    centers = {k: ((v.min(axis=0) + v.max(axis=0)) * 0.5) for k, v in items.items()}
    out: dict[str, float] = {}
    ordered = [v for v in VIEW_ORDER if v in items]
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            d = centers[a] - centers[b]
            out[f"{a}_{b}_3d_m"] = float(np.linalg.norm(d))
            out[f"{a}_{b}_xz_m"] = float(np.linalg.norm(d[[0, 2]]))
            out[f"{a}_{b}_y_m"] = float(abs(d[1]))
    return out


def mean_named(values: dict[str, float], suffix: str) -> float:
    selected = [v for k, v in values.items() if k.endswith(suffix)]
    return float(np.mean(selected)) if selected else float("nan")


def medoid_view(items: dict[str, np.ndarray]) -> str:
    centers = {k: ((v.min(axis=0) + v.max(axis=0)) * 0.5) for k, v in items.items()}
    best_name = ""
    best_score = float("inf")
    for a, ca in centers.items():
        score = 0.0
        for b, cb in centers.items():
            if a != b:
                score += float(np.linalg.norm(ca - cb))
        if score < best_score:
            best_score = score
            best_name = a
    return best_name


def vertex_distance_report(items: dict[str, np.ndarray], reference: np.ndarray) -> dict:
    report = {}
    for view, verts in items.items():
        d = np.linalg.norm(verts - reference, axis=1)
        report[view] = {
            "mean_m": float(d.mean()),
            "median_m": float(np.median(d)),
            "p90_m": float(np.percentile(d, 90)),
            "p95_m": float(np.percentile(d, 95)),
            "max_m": float(d.max()),
        }
    all_d = np.concatenate([np.linalg.norm(v - reference, axis=1) for v in items.values()])
    report["all_views"] = {
        "mean_m": float(all_d.mean()),
        "median_m": float(np.median(all_d)),
        "p90_m": float(np.percentile(all_d, 90)),
        "p95_m": float(np.percentile(all_d, 95)),
        "max_m": float(all_d.max()),
    }
    return report


def render_projection_png(
    path: Path,
    items: dict[str, np.ndarray],
    candidate: np.ndarray | None,
    plane: str,
    max_points: int = 2500,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[warn] cannot render {path.name}: {exc}")
        return

    dims = {
        "xz_top": (0, 2, "X", "Z"),
        "xy_front": (0, 1, "X", "Y"),
        "zy_side": (2, 1, "Z", "Y"),
    }
    if plane not in dims:
        raise ValueError(f"Unknown projection plane: {plane}")
    a, b, xlabel, ylabel = dims[plane]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    for view, verts in items.items():
        step = max(1, len(verts) // max_points)
        pts = verts[::step]
        col = np.asarray(VIEW_COLORS[view], dtype=np.float64) / 255.0
        ax.scatter(pts[:, a], pts[:, b], s=1.0, color=col, alpha=0.22, linewidths=0)
    if candidate is not None:
        step = max(1, len(candidate) // max_points)
        pts = candidate[::step]
        ax.scatter(pts[:, a], pts[:, b], s=1.2, color=(1.0, 0.86, 0.0), alpha=0.7, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.18)
    ax.set_title(plane)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def apply_y_alignment(items: dict[str, np.ndarray], mode: str) -> tuple[dict[str, np.ndarray], dict]:
    if mode == "none":
        return items, {"mode": mode}
    if mode == "floor":
        values = {view: float(verts[:, 1].min()) for view, verts in items.items()}
    elif mode == "center":
        values = {view: float(((verts[:, 1].min() + verts[:, 1].max()) * 0.5)) for view, verts in items.items()}
    else:
        raise ValueError(f"Unsupported y alignment mode: {mode}")
    target = float(np.median(list(values.values())))
    aligned = {}
    shifts = {}
    for view, verts in items.items():
        shift = target - values[view]
        aligned[view] = verts + np.array([0.0, shift, 0.0], dtype=np.float64)
        shifts[view] = float(shift)
    return aligned, {"mode": mode, "target_y": target, "source_y": values, "shift_y_by_view_m": shifts}


def save_world_copy_if_requested(args: argparse.Namespace, canonical_mesh: np.ndarray, faces: np.ndarray) -> str | None:
    if args.cam2world is None:
        return None
    t_cw = np.load(Path(args.cam2world)).astype(np.float64)
    if t_cw.shape != (4, 4):
        raise ValueError(f"cam2world must be 4x4, got {t_cw.shape}")
    world = (t_cw[:3, :3] @ canonical_mesh.T).T + t_cw[:3, 3]
    out_path = Path(args.output_dir) / "single_candidate_world_cam2world_once.ply"
    write_ply(out_path, world, faces, (255, 220, 50))
    return str(out_path)


def generate_smpl_mesh_if_requested(
    args: argparse.Namespace,
    betas: np.ndarray,
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if args.smpl_model_dir is None:
        return None
    try:
        import torch
        import smplx as smpl_backend
    except ImportError as exc:
        print(f"[warn] cannot generate param mesh because torch/smplx import failed: {exc}")
        return None

    device = torch.device(args.device)
    model = smpl_backend.create(
        model_path=str(Path(args.smpl_model_dir)),
        model_type="smpl",
        gender=args.gender,
        num_betas=10,
        use_pca=False,
        create_transl=False,
        batch_size=1,
    ).to(device).eval()
    with torch.inference_mode():
        out = model(
            betas=torch.from_numpy(betas.reshape(1, 10).astype(np.float32)).to(device),
            body_pose=torch.from_numpy(body_pose.reshape(1, 69).astype(np.float32)).to(device),
            global_orient=torch.from_numpy(global_orient.reshape(1, 3).astype(np.float32)).to(device),
            transl=torch.from_numpy(transl.reshape(1, 3).astype(np.float32)).to(device),
            return_verts=True,
        )
    verts = out.vertices[0].detach().cpu().numpy().astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int32)
    return verts, faces


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = Path(args.params) if args.params else input_dir / "final_smpl_params.npz"
    params = np.load(params_path, allow_pickle=True)

    view_names = [str(v) for v in params["view_names"]]
    view_to_index = {view_key(name): i for i, name in enumerate(view_names)}
    missing = [v for v in VIEW_ORDER if v not in view_to_index]
    if missing:
        raise ValueError(f"Missing views in params: {missing}")
    yaw_deg = YAW_PRESETS[args.yaw_preset]

    canonical_meshes: dict[str, np.ndarray] = {}
    canonical_orient = {}
    canonical_transl = {}
    faces_ref: np.ndarray | None = None
    combined_v: list[np.ndarray] = []
    combined_f: list[np.ndarray] = []
    combined_c: list[np.ndarray] = []
    offset = 0
    report = {
        "schema_version": "ii12_canonical_cam_fusion_v1",
        "input_dir": str(input_dir),
        "params": str(params_path),
        "assumption": (
            "same vertical/y position across views; view differences are canonicalized "
            "by yaw only around the selected per-view pivot"
        ),
        "yaw_preset": args.yaw_preset,
        "canonical_yaw_deg": yaw_deg,
        "rotation_pivot": args.rotation_pivot,
        "y_alignment": None,
        "views": {},
    }

    for view in VIEW_ORDER:
        obj_path = input_dir / "meshes" / f"final_{view}.obj"
        verts, faces = load_obj(obj_path)
        if faces_ref is None:
            faces_ref = faces
        elif not np.array_equal(faces_ref, faces):
            raise ValueError(f"Face topology differs for view {view}")
        r_yaw = rot_y(math.radians(yaw_deg[view]))
        i = view_to_index[view]
        if args.rotation_pivot == "transl":
            pivot = params["transl_final"][i].astype(np.float64)
        elif args.rotation_pivot == "bbox_center":
            pivot = (verts.min(axis=0) + verts.max(axis=0)) * 0.5
        elif args.rotation_pivot == "origin":
            pivot = np.zeros(3, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported rotation_pivot: {args.rotation_pivot}")
        verts_can = pivot + (r_yaw @ (verts - pivot).T).T
        canonical_meshes[view] = verts_can

        r_global = axis_angle_to_matrix(params["global_orient_final"][i])
        r_can = r_yaw @ r_global
        canonical_orient[view] = matrix_to_axis_angle(r_can)
        if args.rotation_pivot == "origin":
            canonical_transl[view] = r_yaw @ params["transl_final"][i].astype(np.float64)
        else:
            canonical_transl[view] = params["transl_final"][i].astype(np.float64)

        write_ply(output_dir / f"canonical_{view}.ply", verts_can, faces, VIEW_COLORS[view])
        report["views"][view] = {
            "source_view_name": view_names[i],
            "source_obj": str(obj_path),
            "yaw_pivot_xyz": pivot.tolist(),
            "bbox": bbox_report(verts_can),
            "canonical_global_orient_axis_angle": canonical_orient[view].tolist(),
            "canonical_transl": canonical_transl[view].tolist(),
        }

    canonical_meshes, y_alignment = apply_y_alignment(canonical_meshes, args.y_align)
    report["y_alignment"] = y_alignment

    combined_v = []
    combined_f = []
    combined_c = []
    offset = 0
    for view in VIEW_ORDER:
        faces = faces_ref
        verts_can = canonical_meshes[view]
        report["views"][view]["bbox"] = bbox_report(verts_can)

        combined_v.append(verts_can)
        combined_f.append(faces + offset)
        combined_c.append(np.tile(np.asarray(VIEW_COLORS[view], dtype=np.uint8), (len(verts_can), 1)))
        offset += len(verts_can)

    all_v = np.concatenate(combined_v, axis=0)
    all_f = np.concatenate(combined_f, axis=0)
    all_c = np.concatenate(combined_c, axis=0)
    write_ply(output_dir / "canonical_4view_overlay_colored.ply", all_v, all_f, all_c)

    mesh_stack = np.stack([canonical_meshes[v] for v in VIEW_ORDER], axis=0)
    fused_mean_mesh = mesh_stack.mean(axis=0)
    fused_median_mesh = np.median(mesh_stack, axis=0)
    write_ply(output_dir / "single_candidate_mean_mesh_canonical.ply", fused_mean_mesh, faces_ref, (255, 220, 50))
    write_ply(output_dir / "single_candidate_median_mesh_canonical.ply", fused_median_mesh, faces_ref, (255, 180, 40))

    centers = {
        view: ((verts.min(axis=0) + verts.max(axis=0)) * 0.5)
        for view, verts in canonical_meshes.items()
    }
    y_values = np.array([centers[v][1] for v in VIEW_ORDER], dtype=np.float64)
    transl_values = np.stack([canonical_transl[v] for v in VIEW_ORDER], axis=0)
    orient_values = np.stack([canonical_orient[v] for v in VIEW_ORDER], axis=0)
    representative = medoid_view(canonical_meshes)
    if args.representative_view != "auto":
        representative = args.representative_view
    pairwise = pairwise_center_report(canonical_meshes)

    body_pose_mode = args.body_pose_mode
    if body_pose_mode == "base":
        body_pose_single = params["body_pose_base_final"].astype(np.float32)
    elif body_pose_mode == "median":
        body_pose_single = np.median(params["body_pose_final"], axis=0).astype(np.float32)
    elif body_pose_mode == "representative":
        body_pose_single = params["body_pose_final"][view_to_index[representative]].astype(np.float32)
    else:
        raise ValueError(f"Unsupported body_pose_mode: {body_pose_mode}")

    fused_transl = np.median(transl_values, axis=0)
    fused_transl[1] = float(np.median(y_values))
    fused_orient = canonical_orient[representative].astype(np.float32)

    single_npz = output_dir / "single_smpl_candidate_params.npz"
    np.savez_compressed(
        single_npz,
        schema_version=np.array("ii12_single_smpl_candidate_v1"),
        source_final_smpl_params=np.array(str(params_path)),
        coordinate_frame=np.array("canonical_front_camera_yaw_only"),
        betas=params["betas_shared_final"].astype(np.float32),
        body_pose=body_pose_single,
        global_orient=fused_orient,
        transl=fused_transl.astype(np.float32),
        representative_view=np.array(representative),
        body_pose_mode=np.array(body_pose_mode),
        per_view_canonical_global_orient=orient_values.astype(np.float32),
        per_view_canonical_transl=transl_values.astype(np.float32),
        view_order=np.asarray(VIEW_ORDER),
        canonical_yaw_deg=np.asarray([yaw_deg[v] for v in VIEW_ORDER], dtype=np.float32),
    )

    report["combined_overlay"] = {
        "path": str((output_dir / "canonical_4view_overlay_colored.ply").resolve()),
        "bbox": bbox_report(all_v),
        "pairwise_center_distances": pairwise,
        "mean_pairwise_center_3d_m": mean_named(pairwise, "_3d_m"),
        "mean_pairwise_center_xz_m": mean_named(pairwise, "_xz_m"),
        "mean_pairwise_center_y_m": mean_named(pairwise, "_y_m"),
        "center_y_per_view_m": {v: float(centers[v][1]) for v in VIEW_ORDER},
        "center_y_std_m": float(y_values.std()),
        "vertex_distance_to_median_mesh": vertex_distance_report(canonical_meshes, fused_median_mesh),
    }
    report["single_candidate"] = {
        "params_path": str(single_npz.resolve()),
        "representative_view_for_global_orient": representative,
        "body_pose_mode": body_pose_mode,
        "transl_fused_median_xyz": fused_transl.tolist(),
        "mesh_mean_path": str((output_dir / "single_candidate_mean_mesh_canonical.ply").resolve()),
        "mesh_median_path": str((output_dir / "single_candidate_median_mesh_canonical.ply").resolve()),
    }
    param_mesh = generate_smpl_mesh_if_requested(
        args,
        params["betas_shared_final"].astype(np.float32),
        body_pose_single,
        fused_orient,
        fused_transl.astype(np.float32),
    )
    if param_mesh is not None:
        param_verts, param_faces = param_mesh
        param_mesh_path = output_dir / "single_candidate_param_mesh_canonical.ply"
        write_ply(param_mesh_path, param_verts, param_faces, (255, 240, 70))
        report["single_candidate"]["param_mesh_canonical_path"] = str(param_mesh_path.resolve())
        report["single_candidate"]["param_mesh_canonical_bbox"] = bbox_report(param_verts)
        report["single_candidate"]["vertex_distance_to_param_mesh"] = vertex_distance_report(
            canonical_meshes, param_verts
        )
        single_overlay_path = output_dir / "canonical_4view_vs_single_param_overlay.ply"
        write_multi_mesh_ply(
            single_overlay_path,
            [
                *[(canonical_meshes[v], faces_ref, VIEW_COLORS[v]) for v in VIEW_ORDER],
                (param_verts, param_faces, (255, 240, 70)),
            ],
        )
        report["single_candidate"]["param_overlay_path"] = str(single_overlay_path.resolve())
        render_dir = output_dir / "renders"
        render_paths = {}
        for plane in ("xz_top", "xy_front", "zy_side"):
            render_path = render_dir / f"{plane}_canonical_4view_vs_single.png"
            render_projection_png(render_path, canonical_meshes, param_verts, plane)
            render_paths[plane] = str(render_path.resolve())
        report["single_candidate"]["render_paths"] = render_paths
        if args.cam2world is not None:
            t_cw = np.load(Path(args.cam2world)).astype(np.float64)
            param_world = (t_cw[:3, :3] @ param_verts.T).T + t_cw[:3, 3]
            param_world_path = output_dir / "single_candidate_param_mesh_world_cam2world_once.ply"
            write_ply(param_world_path, param_world, param_faces, (255, 240, 70))
            report["single_candidate"]["param_mesh_world_cam2world_once_path"] = str(
                param_world_path.resolve()
            )
    world_path = save_world_copy_if_requested(args, fused_median_mesh, faces_ref)
    if world_path is not None:
        report["single_candidate"]["median_mesh_world_cam2world_once_path"] = world_path

    with (output_dir / "canonical_fusion_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[done] wrote {output_dir}")
    print(f"  overlay: {output_dir / 'canonical_4view_overlay_colored.ply'}")
    print(f"  representative: {representative}")
    print(f"  mean_pairwise_center_3d_m={report['combined_overlay']['mean_pairwise_center_3d_m']:.4f}")
    print(f"  mean_pairwise_center_xz_m={report['combined_overlay']['mean_pairwise_center_xz_m']:.4f}")
    print(f"  center_y_std_m={report['combined_overlay']['center_y_std_m']:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="outputs/ii10_pr7_strict_full_final")
    p.add_argument("--params", default=None)
    p.add_argument("--output-dir", default="outputs/ii12_canonical_fusion_pr7")
    p.add_argument(
        "--body-pose-mode",
        choices=("base", "median", "representative"),
        default="base",
        help="Body pose used in the single parameter candidate. Mesh previews are topology-wise vertex averages.",
    )
    p.add_argument(
        "--representative-view",
        choices=("auto", "front", "right", "back", "left"),
        default="auto",
        help="Representative view for single global_orient and representative body pose mode.",
    )
    p.add_argument(
        "--rotation-pivot",
        choices=("transl", "bbox_center", "origin"),
        default="transl",
        help="Pivot for yaw canonicalization. Use transl for same-place subject rotation.",
    )
    p.add_argument(
        "--yaw-preset",
        choices=tuple(YAW_PRESETS.keys()),
        default="inverse_sides",
        help="View yaw mapping used to canonicalize meshes.",
    )
    p.add_argument(
        "--y-align",
        choices=("none", "floor", "center"),
        default="floor",
        help="Optional canonical y alignment across views after yaw rotation.",
    )
    p.add_argument(
        "--cam2world",
        default="Tpose/[Result]02_smpl/[Stage1]/tpose/stage0_A_final/cam2world.npy",
        help="Optional final one-time cam2world export for the fused median mesh.",
    )
    p.add_argument("--smpl-model-dir", default="models")
    p.add_argument("--gender", default="neutral")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
