"""II-13 final single-SMPL refinement from canonicalized STAND 4-view meshes.

This stage starts from the II-12 single candidate and optimizes one valid SMPL
parameter set against all four canonicalized II-10 view meshes. No scale
optimization is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .depth_utils import load_depth_observation
from .ii8_init import find_depth_path
from .ii12_canonical_fusion import (
    CANONICAL_YAW_DEG,
    VIEW_COLORS,
    VIEW_ORDER,
    YAW_PRESETS,
    apply_y_alignment,
    axis_angle_to_matrix,
    bbox_report,
    load_obj,
    matrix_to_axis_angle,
    pairwise_center_report,
    render_projection_png,
    rot_y,
    vertex_distance_report,
    view_key,
    write_multi_mesh_ply,
    write_ply,
)
from .region_map import CLOTH_LABELS, SHOE_LABELS, SKIN_LABELS, load_partseg
from .view_corrections import (
    apply_camera_correction_torch,
    invert_camera_correction_np,
    load_view_corrections,
)


LOWER_BODY_LABELS = {7, 11, 12, 16, 20}
LEG_SKIN_LABELS = {7, 11, 16, 20}
LOWER_CLOTH_LABELS = {12}


def canonicalize_view_mesh(
    verts: np.ndarray,
    transl: np.ndarray,
    view: str,
    pivot_mode: str,
    yaw_deg: dict[str, float],
) -> np.ndarray:
    r_yaw = rot_y(math.radians(yaw_deg[view]))
    if pivot_mode == "transl":
        pivot = transl.astype(np.float64)
    elif pivot_mode == "bbox_center":
        pivot = (verts.min(axis=0) + verts.max(axis=0)) * 0.5
    elif pivot_mode == "origin":
        pivot = np.zeros(3, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported pivot mode: {pivot_mode}")
    return pivot + (r_yaw @ (verts - pivot).T).T


def canonicalize_orient(axis_angle: np.ndarray, view: str, yaw_deg: dict[str, float]) -> np.ndarray:
    r_yaw = rot_y(math.radians(yaw_deg[view]))
    return matrix_to_axis_angle(r_yaw @ axis_angle_to_matrix(axis_angle))


def load_canonical_targets(
    input_dir: Path,
    final_params: Any,
    pivot_mode: str,
    yaw_deg: dict[str, float],
    y_align: str,
    view_corrections: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    view_names = [str(v) for v in final_params["view_names"]]
    view_to_index = {view_key(name): i for i, name in enumerate(view_names)}
    missing = [v for v in VIEW_ORDER if v not in view_to_index]
    if missing:
        raise ValueError(f"Missing views in final params: {missing}")
    targets: dict[str, np.ndarray] = {}
    faces_ref: np.ndarray | None = None
    for view in VIEW_ORDER:
        verts, faces = load_obj(input_dir / "meshes" / f"final_{view}.obj")
        if faces_ref is None:
            faces_ref = faces
        elif not np.array_equal(faces_ref, faces):
            raise ValueError(f"Face topology differs for view {view}")
        idx = view_to_index[view]
        verts = invert_camera_correction_np(verts, view, view_corrections or {})
        targets[view] = canonicalize_view_mesh(
            verts,
            final_params["transl_final"][idx].astype(np.float64),
            view,
            pivot_mode,
            yaw_deg,
        )
    targets, y_alignment = apply_y_alignment(targets, y_align)
    assert faces_ref is not None
    return targets, faces_ref, y_alignment


def load_initial_candidate(candidate_path: Path, final_params: Any, yaw_deg: dict[str, float]) -> dict[str, np.ndarray]:
    if candidate_path.exists():
        cand = np.load(candidate_path, allow_pickle=True)
        return {
            "betas": cand["betas"].astype(np.float32),
            "body_pose": cand["body_pose"].astype(np.float32),
            "global_orient": cand["global_orient"].astype(np.float32),
            "transl": cand["transl"].astype(np.float32),
        }

    view_names = [str(v) for v in final_params["view_names"]]
    view_to_index = {view_key(name): i for i, name in enumerate(view_names)}
    idx = view_to_index["front"]
    return {
        "betas": final_params["betas_shared_final"].astype(np.float32),
        "body_pose": final_params["body_pose_final"][idx].astype(np.float32),
        "global_orient": canonicalize_orient(final_params["global_orient_final"][idx], "front", yaw_deg).astype(np.float32),
        "transl": final_params["transl_final"][idx].astype(np.float32),
    }


def tensor_stats_from_dist(dist: np.ndarray) -> dict:
    return {
        "mean_m": float(dist.mean()),
        "median_m": float(np.median(dist)),
        "p90_m": float(np.percentile(dist, 90)),
        "p95_m": float(np.percentile(dist, 95)),
        "max_m": float(dist.max()),
    }


def target_distance_report(verts: np.ndarray, targets: dict[str, np.ndarray]) -> dict:
    report = {}
    for view, target in targets.items():
        report[view] = tensor_stats_from_dist(np.linalg.norm(verts - target, axis=1))
    all_d = np.concatenate([np.linalg.norm(verts - target, axis=1) for target in targets.values()])
    report["all_views"] = tensor_stats_from_dist(all_d)
    return report


def sample_vertices(num_vertices: int, max_vertices: int, seed: int) -> np.ndarray:
    if max_vertices <= 0 or max_vertices >= num_vertices:
        return np.arange(num_vertices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_vertices, size=max_vertices, replace=False)).astype(np.int64)


def robust_l2(x: Any, eps: float) -> Any:
    return (x.square().sum(dim=-1) + eps * eps).sqrt()


def torch_axis_angle_to_matrix(vec: Any) -> Any:
    import torch

    theta = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = vec / theta
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)
    k = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=-1,
    ).reshape(vec.shape[:-1] + (3, 3))
    eye = torch.eye(3, device=vec.device, dtype=vec.dtype).expand(vec.shape[:-1] + (3, 3))
    theta_e = theta[..., None]
    return eye + torch.sin(theta_e) * k + (1.0 - torch.cos(theta_e)) * (k @ k)


def torch_matrix_to_axis_angle(mat: Any) -> Any:
    import torch

    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    axis = torch.stack(
        [
            mat[..., 2, 1] - mat[..., 1, 2],
            mat[..., 0, 2] - mat[..., 2, 0],
            mat[..., 1, 0] - mat[..., 0, 1],
        ],
        dim=-1,
    )
    axis = axis / (2.0 * torch.sin(theta)[..., None]).clamp_min(1e-8)
    return axis * theta[..., None]


def torch_rot_y(theta: Any) -> Any:
    import torch

    c = torch.cos(theta).reshape(-1)
    s = torch.sin(theta).reshape(-1)
    zeros = torch.zeros_like(c)
    ones = torch.ones_like(c)
    return torch.stack(
        [
            c,
            zeros,
            s,
            zeros,
            ones,
            zeros,
            -s,
            zeros,
            c,
        ],
        dim=-1,
    ).reshape(theta.shape + (3, 3))


def raw_depth_region_mask(label_map: np.ndarray, valid_mask: np.ndarray, region: str) -> np.ndarray:
    usable = valid_mask.astype(bool)
    if region == "body_no_shoe":
        labels = (SKIN_LABELS | CLOTH_LABELS) - SHOE_LABELS
    elif region == "lower_body_no_shoe":
        labels = LOWER_BODY_LABELS - SHOE_LABELS
    elif region == "leg_skin_only":
        labels = LEG_SKIN_LABELS
    elif region == "lower_cloth":
        labels = LOWER_CLOTH_LABELS
    else:
        raise ValueError(f"Unsupported raw depth region: {region}")
    return (np.isin(label_map, list(labels)) & usable).astype(bool)


def sample_depth_vertices(num_vertices: int, max_vertices: int, seed: int) -> np.ndarray:
    if max_vertices <= 0 or max_vertices >= num_vertices:
        return np.arange(num_vertices, dtype=np.int64)
    rng = np.random.default_rng(seed + 100_003)
    return np.sort(rng.choice(num_vertices, size=max_vertices, replace=False)).astype(np.int64)


def prepare_raw_depth_terms(args: argparse.Namespace, final_params: Any, view_to_index: dict[str, int], device: Any) -> dict | None:
    if args.raw_depth_weight <= 0:
        return None
    if not args.depth_dir or not args.partseg_dir:
        raise ValueError("--depth-dir and --partseg-dir are required when --raw-depth-weight > 0")

    import torch

    views = [v.strip() for v in args.raw_depth_views.split(",") if v.strip()]
    unknown = [v for v in views if v not in VIEW_ORDER]
    if unknown:
        raise ValueError(f"Unsupported raw depth views: {unknown}")
    view_names = [str(v) for v in final_params["view_names"]]
    data = {}
    for view in views:
        idx = view_to_index[view]
        view_name = view_names[idx]
        k = final_params["K"][idx].astype(np.float64)
        depth_path = find_depth_path(Path(args.depth_dir), view_name)
        obs = load_depth_observation(depth_path, k.astype(np.float32), args.depth_y_axis)
        regions = load_partseg(
            Path(args.partseg_dir),
            view_name,
            obs["valid_mask"],
            allow_png_label_fallback=args.allow_png_label_fallback,
        )
        mask = raw_depth_region_mask(regions["label_map"], obs["valid_mask"], args.raw_depth_region)
        z_obs = obs["z_obs"].astype(np.float32)
        h, w = z_obs.shape
        data[view] = {
            "view_name": view_name,
            "depth_path": str(depth_path),
            "height": h,
            "width": w,
            "K": torch.from_numpy(k.astype(np.float32)).to(device),
            "z_obs": torch.from_numpy(z_obs).to(device),
            "mask": torch.from_numpy(mask).to(device),
            "valid_pixels": int(mask.sum()),
            "partseg_source": str(regions.get("partseg_source", "unknown")),
        }
    return {
        "views": views,
        "data": data,
    }


def raw_depth_report_metadata(raw_depth_terms: dict | None) -> dict:
    if raw_depth_terms is None:
        return {}
    return {
        view: {
            "view_name": item["view_name"],
            "depth_path": item["depth_path"],
            "height": item["height"],
            "width": item["width"],
            "valid_pixels": item["valid_pixels"],
            "partseg_source": item["partseg_source"],
        }
        for view, item in raw_depth_terms["data"].items()
    }


def torch_canonical_to_view_camera(
    verts_canonical: Any,
    view: str,
    raw_depth_terms: dict,
    pivots: dict[str, Any],
    y_shifts: dict[str, float],
    yaw_deg: dict[str, float],
) -> Any:
    import torch

    pivot = pivots[view]
    shift = torch.tensor([0.0, y_shifts.get(view, 0.0), 0.0], device=verts_canonical.device, dtype=verts_canonical.dtype)
    r = torch_rot_y(torch.tensor([math.radians(yaw_deg[view])], device=verts_canonical.device, dtype=verts_canonical.dtype))[0]
    return pivot + ((verts_canonical - shift - pivot) @ r)


def raw_depth_projective_loss(
    verts_canonical: Any,
    raw_depth_terms: dict | None,
    pivots: dict[str, Any],
    y_shifts: dict[str, float],
    yaw_deg: dict[str, float],
    args: argparse.Namespace,
) -> tuple[Any, dict]:
    import torch

    zero = verts_canonical.sum() * 0.0
    if raw_depth_terms is None:
        return zero, {"count": 0, "mean_abs_m": None}

    losses = []
    residual_chunks = []
    counts = {}
    for view in raw_depth_terms["views"]:
        item = raw_depth_terms["data"][view]
        pts = torch_canonical_to_view_camera(verts_canonical, view, raw_depth_terms, pivots, y_shifts, yaw_deg)
        pts = apply_camera_correction_torch(pts, view, raw_depth_terms.get("view_corrections", {}))
        k = item["K"]
        z = pts[:, 2]
        positive = z > 1e-4
        u = k[0, 0] * (pts[:, 0] / z.clamp_min(1e-4)) + k[0, 2]
        v = k[1, 2] - k[1, 1] * (pts[:, 1] / z.clamp_min(1e-4))
        x = torch.round(u).long()
        y = torch.round(v).long()
        inside = positive & (x >= 0) & (x < item["width"]) & (y >= 0) & (y < item["height"])
        if not bool(inside.any()):
            counts[view] = 0
            continue
        x_in = x[inside]
        y_in = y[inside]
        z_in = z[inside]
        valid = item["mask"][y_in, x_in]
        if not bool(valid.any()):
            counts[view] = 0
            continue
        z_model = z_in[valid]
        z_obs = item["z_obs"][y_in[valid], x_in[valid]]
        residual = z_model - z_obs
        plausible = torch.isfinite(residual) & (torch.abs(residual.detach()) <= args.raw_depth_max_abs_m)
        residual = residual[plausible]
        counts[view] = int(residual.numel())
        if residual.numel() == 0:
            continue
        residual_chunks.append(residual.detach())
        losses.append(torch.sqrt(residual.square() + args.raw_depth_robust_eps * args.raw_depth_robust_eps).mean())

    if not losses:
        return zero, {"count": 0, "mean_abs_m": None, "counts_by_view": counts}
    all_residual = torch.cat(residual_chunks)
    return torch.stack(losses).mean(), {
        "count": int(all_residual.numel()),
        "mean_abs_m": float(all_residual.abs().mean().cpu()),
        "signed_mean_m": float(all_residual.mean().cpu()),
        "counts_by_view": counts,
    }


def run(args: argparse.Namespace) -> None:
    import torch
    import smplx as smpl_backend

    input_dir = Path(args.input_dir)
    ii12_dir = Path(args.ii12_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_params_path = Path(args.final_params) if args.final_params else input_dir / "final_smpl_params.npz"
    final_params = np.load(final_params_path, allow_pickle=True)
    yaw_deg = YAW_PRESETS[args.yaw_preset]
    view_corrections = load_view_corrections(args.view_corrections_json)
    candidate_path = Path(args.initial_candidate) if args.initial_candidate else ii12_dir / "single_smpl_candidate_params.npz"
    init = load_initial_candidate(candidate_path, final_params, yaw_deg)
    targets, faces, y_alignment = load_canonical_targets(
        input_dir, final_params, args.rotation_pivot, yaw_deg, args.y_align, view_corrections
    )
    view_names = [str(v) for v in final_params["view_names"]]
    view_to_index = {view_key(name): i for i, name in enumerate(view_names)}
    target_stack_np = np.stack([targets[v] for v in VIEW_ORDER], axis=0).astype(np.float32)
    target_median_np = np.median(target_stack_np, axis=0).astype(np.float32)
    sample_idx = sample_vertices(target_stack_np.shape[1], args.max_vertices, args.seed)
    depth_sample_idx = sample_depth_vertices(target_stack_np.shape[1], args.raw_depth_max_vertices, args.seed)

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

    target_all = torch.from_numpy(target_stack_np[:, sample_idx]).to(device)
    target_median = torch.from_numpy(target_median_np[sample_idx]).to(device)
    depth_sample_idx_t = torch.from_numpy(depth_sample_idx).to(device)
    betas_init = torch.from_numpy(init["betas"].reshape(1, 10)).to(device)
    body_init = torch.from_numpy(init["body_pose"].reshape(1, 69)).to(device)
    orient_init = torch.from_numpy(init["global_orient"].reshape(1, 3)).to(device)
    transl_init = torch.from_numpy(init["transl"].reshape(1, 3)).to(device)
    raw_depth_terms = prepare_raw_depth_terms(args, final_params, view_to_index, device)
    if raw_depth_terms is not None:
        raw_depth_terms["view_corrections"] = view_corrections
    y_shifts = y_alignment.get("shift_y_by_view_m", {}) if isinstance(y_alignment, dict) else {}
    pivots = {
        view: torch.from_numpy(final_params["transl_final"][view_to_index[view]].astype(np.float32)).to(device)
        for view in VIEW_ORDER
        if view in view_to_index
    }

    betas = betas_init.detach().clone().requires_grad_(args.lr_betas > 0)
    body_pose = body_init.detach().clone().requires_grad_(args.lr_body > 0)
    if args.orient_mode == "full":
        global_orient = orient_init.detach().clone().requires_grad_(args.lr_orient > 0)
        yaw_delta = None
    elif args.orient_mode == "fixed":
        global_orient = orient_init.detach().clone().requires_grad_(False)
        yaw_delta = None
    elif args.orient_mode == "yaw_delta":
        global_orient = orient_init.detach().clone().requires_grad_(False)
        yaw_delta = torch.zeros((1,), device=device, dtype=torch.float32, requires_grad=args.lr_orient > 0)
    else:
        raise ValueError(f"Unsupported orient_mode: {args.orient_mode}")
    transl = transl_init.detach().clone().requires_grad_(args.lr_transl > 0)

    params = []
    if betas.requires_grad:
        params.append({"params": [betas], "lr": args.lr_betas})
    if body_pose.requires_grad:
        params.append({"params": [body_pose], "lr": args.lr_body})
    if args.orient_mode == "full" and global_orient.requires_grad:
        params.append({"params": [global_orient], "lr": args.lr_orient})
    if args.orient_mode == "yaw_delta" and yaw_delta is not None and yaw_delta.requires_grad:
        params.append({"params": [yaw_delta], "lr": args.lr_orient})
    if transl.requires_grad:
        params.append({"params": [transl], "lr": args.lr_transl})
    if not params:
        raise ValueError("At least one learning rate must be > 0")
    opt = torch.optim.Adam(params, amsgrad=True)

    history = []
    best = {"loss": float("inf"), "payload": None}
    for it in range(args.iters + 1):
        opt.zero_grad(set_to_none=True)
        if args.orient_mode == "yaw_delta":
            orient_mat = torch_rot_y(yaw_delta)[0] @ torch_axis_angle_to_matrix(global_orient)[0]
            orient_for_model = torch_matrix_to_axis_angle(orient_mat.reshape(1, 3, 3))
        else:
            orient_for_model = global_orient
        out = model(
            betas=betas,
            body_pose=body_pose,
            global_orient=orient_for_model,
            transl=transl,
            return_verts=True,
        )
        verts = out.vertices[0, sample_idx]
        residual_all = verts[None, :, :] - target_all
        residual_median = verts - target_median
        loss_views = robust_l2(residual_all, args.robust_eps).mean()
        loss_median = robust_l2(residual_median, args.robust_eps).mean()
        loss_raw_depth, raw_depth_diag = raw_depth_projective_loss(
            out.vertices[0, depth_sample_idx_t],
            raw_depth_terms,
            pivots,
            y_shifts,
            yaw_deg,
            args,
        )
        height = out.vertices[0, :, 1].max() - out.vertices[0, :, 1].min()
        loss_height = (height - args.target_height_m).square()
        loss_body_prior = (body_pose - body_init).square().mean()
        if args.orient_mode == "yaw_delta":
            loss_orient_prior = yaw_delta.square().mean()
        else:
            loss_orient_prior = (global_orient - orient_init).square().mean()
        loss_transl_prior = (transl - transl_init).square().mean()
        loss_beta_prior = (betas - betas_init).square().mean()
        loss = (
            args.w_views * loss_views
            + args.w_median * loss_median
            + args.raw_depth_weight * loss_raw_depth
            + args.w_height * loss_height
            + args.w_body_prior * loss_body_prior
            + args.w_orient_prior * loss_orient_prior
            + args.w_transl_prior * loss_transl_prior
            + args.w_beta_prior * loss_beta_prior
        )
        if it > 0:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for group in params for p in group["params"]], args.grad_clip)
            opt.step()
        row = {
            "iter": it,
            "loss": float(loss.detach().cpu()),
            "loss_views": float(loss_views.detach().cpu()),
            "loss_median": float(loss_median.detach().cpu()),
            "loss_raw_depth": float(loss_raw_depth.detach().cpu()),
            "raw_depth_count": raw_depth_diag["count"],
            "raw_depth_mean_abs_m": raw_depth_diag["mean_abs_m"],
            "loss_height": float(loss_height.detach().cpu()),
            "height_m": float(height.detach().cpu()),
        }
        history.append(row)
        if row["loss"] < best["loss"]:
            best["loss"] = row["loss"]
            best["payload"] = {
                "betas": betas.detach().clone(),
                "body_pose": body_pose.detach().clone(),
                "global_orient": orient_for_model.detach().clone(),
                "transl": transl.detach().clone(),
                "yaw_delta": yaw_delta.detach().clone() if yaw_delta is not None else None,
            }
        if args.print_every > 0 and it % args.print_every == 0:
            print(
                f"iter={it:04d} loss={row['loss']:.6f} views={row['loss_views']:.4f} "
                f"median={row['loss_median']:.4f} raw_depth={row['loss_raw_depth']:.4f} "
                f"raw_n={row['raw_depth_count']} h={row['height_m']:.4f}"
            )

    assert best["payload"] is not None
    with torch.inference_mode():
        out = model(
            betas=best["payload"]["betas"],
            body_pose=best["payload"]["body_pose"],
            global_orient=best["payload"]["global_orient"],
            transl=best["payload"]["transl"],
            return_verts=True,
        )
    verts_final = out.vertices[0].detach().cpu().numpy().astype(np.float64)
    betas_final = best["payload"]["betas"].detach().cpu().numpy().reshape(10).astype(np.float32)
    body_final = best["payload"]["body_pose"].detach().cpu().numpy().reshape(23, 3).astype(np.float32)
    orient_final = best["payload"]["global_orient"].detach().cpu().numpy().reshape(3).astype(np.float32)
    transl_final = best["payload"]["transl"].detach().cpu().numpy().reshape(3).astype(np.float32)

    write_ply(output_dir / "final_single_smpl_canonical.ply", verts_final, faces, (255, 235, 40))
    write_multi_mesh_ply(
        output_dir / "canonical_4view_vs_final_single_overlay.ply",
        [
            *[(targets[v], faces, VIEW_COLORS[v]) for v in VIEW_ORDER],
            (verts_final, faces, (255, 235, 40)),
        ],
    )
    if args.cam2world:
        t_cw = np.load(Path(args.cam2world)).astype(np.float64)
        verts_world = (t_cw[:3, :3] @ verts_final.T).T + t_cw[:3, 3]
        write_ply(output_dir / "final_single_smpl_world_cam2world_once.ply", verts_world, faces, (255, 235, 40))

    render_dir = output_dir / "renders"
    render_paths = {}
    for plane in ("xz_top", "xy_front", "zy_side"):
        path = render_dir / f"{plane}_canonical_4view_vs_final_single.png"
        render_projection_png(path, targets, verts_final, plane)
        render_paths[plane] = str(path.resolve())

    np.savez_compressed(
        output_dir / "final_single_smpl_params.npz",
        schema_version=np.array("ii13_final_single_smpl_refine_v1"),
        source_ii10_final_params=np.array(str(final_params_path)),
        source_ii12_candidate=np.array(str(candidate_path)),
        coordinate_frame=np.array("canonical_front_camera_yaw_only_transl_pivot"),
        betas=betas_final,
        body_pose=body_final,
        global_orient=orient_final,
        transl=transl_final,
        target_height_m=np.array(args.target_height_m, dtype=np.float32),
        canonical_yaw_deg=np.asarray([yaw_deg[v] for v in VIEW_ORDER], dtype=np.float32),
        yaw_preset=np.array(args.yaw_preset),
        y_align=np.array(args.y_align),
        view_order=np.asarray(VIEW_ORDER),
    )

    with (output_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    centers = {view: ((target.min(axis=0) + target.max(axis=0)) * 0.5) for view, target in targets.items()}
    target_pairwise = pairwise_center_report(targets)
    final_distance = target_distance_report(verts_final, targets)
    median_distance = vertex_distance_report(targets, target_median_np)
    report = {
        "schema_version": "ii13_final_single_smpl_refine_v1",
        "input_dir": str(input_dir),
        "ii12_dir": str(ii12_dir),
        "final_params": str((output_dir / "final_single_smpl_params.npz").resolve()),
        "final_single_canonical_ply": str((output_dir / "final_single_smpl_canonical.ply").resolve()),
        "final_single_world_ply": str((output_dir / "final_single_smpl_world_cam2world_once.ply").resolve()) if args.cam2world else None,
        "overlay_ply": str((output_dir / "canonical_4view_vs_final_single_overlay.ply").resolve()),
        "render_paths": render_paths,
        "optimization": {
            "iters": args.iters,
            "best_loss": best["loss"],
            "weights": {
                "views": args.w_views,
                "median": args.w_median,
                "height": args.w_height,
                "raw_depth": args.raw_depth_weight,
                "body_prior": args.w_body_prior,
                "orient_prior": args.w_orient_prior,
                "transl_prior": args.w_transl_prior,
                "beta_prior": args.w_beta_prior,
            },
            "learning_rates": {
                "betas": args.lr_betas,
                "body": args.lr_body,
                "orient": args.lr_orient,
                "transl": args.lr_transl,
            },
            "max_vertices": args.max_vertices,
            "raw_depth": {
                "enabled": bool(args.raw_depth_weight > 0),
                "region": args.raw_depth_region,
                "views": raw_depth_terms["views"] if raw_depth_terms is not None else [],
                "max_vertices": args.raw_depth_max_vertices,
                "max_abs_residual_m": args.raw_depth_max_abs_m,
                "robust_eps": args.raw_depth_robust_eps,
                "loaded_views": raw_depth_report_metadata(raw_depth_terms),
            },
            "view_corrections": view_corrections,
            "yaw_preset": args.yaw_preset,
            "y_align": args.y_align,
            "orient_mode": args.orient_mode,
            "yaw_delta_rad": float(best["payload"]["yaw_delta"].detach().cpu()[0]) if best["payload"]["yaw_delta"] is not None else None,
            "y_alignment": y_alignment,
        },
        "target_canonical_overlay": {
            "bbox": bbox_report(np.concatenate([targets[v] for v in VIEW_ORDER], axis=0)),
            "pairwise_center_distances": target_pairwise,
            "center_y_per_view_m": {v: float(centers[v][1]) for v in VIEW_ORDER},
            "center_y_std_m": float(np.std([centers[v][1] for v in VIEW_ORDER])),
        },
        "final_single": {
            "bbox": bbox_report(verts_final),
            "height_error_m": float(bbox_report(verts_final)["height_y_m"] - args.target_height_m),
            "distance_to_targets": final_distance,
            "distance_to_median_reference": tensor_stats_from_dist(np.linalg.norm(verts_final - target_median_np, axis=1)),
        },
        "median_reference": {
            "distance_to_targets": median_distance,
        },
    }
    with (output_dir / "final_refine_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[done] wrote {output_dir}")
    print(f"  final height={report['final_single']['bbox']['height_y_m']:.4f}m")
    print(f"  final all-view mean distance={final_distance['all_views']['mean_m']:.4f}m")
    print(f"  final all-view p95 distance={final_distance['all_views']['p95_m']:.4f}m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="outputs/ii10_pr7_strict_full_final")
    p.add_argument("--ii12-dir", default="outputs/ii12_canonical_fusion_pr7_best")
    p.add_argument("--output-dir", default="outputs/ii13_single_refine_pr7")
    p.add_argument("--final-params", default=None)
    p.add_argument("--initial-candidate", default=None)
    p.add_argument("--smpl-model-dir", default="models")
    p.add_argument("--gender", default="neutral")
    p.add_argument("--device", default="cpu")
    p.add_argument("--rotation-pivot", choices=("transl", "bbox_center", "origin"), default="transl")
    p.add_argument("--yaw-preset", choices=tuple(YAW_PRESETS.keys()), default="inverse_sides")
    p.add_argument("--y-align", choices=("none", "floor", "center"), default="floor")
    p.add_argument("--view-corrections-json", default=None)
    p.add_argument("--cam2world", default="Tpose/[Result]02_smpl/[Stage1]/tpose/stage0_A_final/cam2world.npy")
    p.add_argument("--iters", type=int, default=450)
    p.add_argument("--max-vertices", type=int, default=3500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--robust-eps", type=float, default=0.015)
    p.add_argument("--target-height-m", type=float, default=1.810)
    p.add_argument("--lr-betas", type=float, default=0.0)
    p.add_argument("--lr-body", type=float, default=0.006)
    p.add_argument("--lr-orient", type=float, default=0.01)
    p.add_argument("--lr-transl", type=float, default=0.01)
    p.add_argument("--w-views", type=float, default=0.70)
    p.add_argument("--w-median", type=float, default=0.30)
    p.add_argument("--w-height", type=float, default=0.02)
    p.add_argument("--w-body-prior", type=float, default=0.02)
    p.add_argument("--w-orient-prior", type=float, default=0.005)
    p.add_argument("--w-transl-prior", type=float, default=0.005)
    p.add_argument("--w-beta-prior", type=float, default=0.10)
    p.add_argument("--raw-depth-weight", type=float, default=0.0)
    p.add_argument("--depth-dir", default=None)
    p.add_argument("--partseg-dir", default=None)
    p.add_argument("--depth-y-axis", choices=("up", "down"), default="down")
    p.add_argument("--allow-png-label-fallback", action="store_true")
    p.add_argument("--raw-depth-region", choices=("body_no_shoe", "lower_body_no_shoe", "leg_skin_only", "lower_cloth"), default="lower_body_no_shoe")
    p.add_argument("--raw-depth-views", default="front,right,back,left")
    p.add_argument("--raw-depth-max-vertices", type=int, default=3500)
    p.add_argument("--raw-depth-max-abs-m", type=float, default=0.35)
    p.add_argument("--raw-depth-robust-eps", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument(
        "--orient-mode",
        choices=("full", "fixed", "yaw_delta"),
        default="yaw_delta",
        help="Limit final single orientation. yaw_delta keeps initial pitch/roll fixed and optimizes only a Y-axis yaw increment.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
