#!/usr/bin/env python3
"""II-10 STAND 4-view joint SMPL fitting.

Specific path-B fitting:
one shared SMPL shape, four view-specific pose/orientation/translation states,
no point-cloud fusion, no global scale, no shoe proxy fitting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .depth_utils import load_depth_observation
from .diagnostics import height_report, leg_cross_section_report, save_overlay, write_obj
from .ii8_init import find_depth_path, find_rgb_path, load_k_by_view
from .losses import cloth_inside_loss, robust_clamped, skin_depth_loss
from .region_map import load_metadata, load_partseg, save_binary_mask_png, save_region_debug, shoe_points_from_xyz
from .smpl_surface import surface_points
from .visibility import gather_projected_observation, project_y_up


SCHEMA_VERSION = "ii10_stand_4view_smpl_fit_v1"
VIEW_ORDER = ("front", "back", "left", "right")


@dataclass
class StageConfig:
    name: str
    iters: int
    lr_betas: float
    lr_pose_base: float
    lr_pose_delta: float
    lr_orient: float
    lr_transl: float
    grad_clip: float
    residual_clamp: float
    weights: dict[str, float]


def default_stage_configs(args: argparse.Namespace) -> list[StageConfig]:
    return [
        StageConfig(
            "stage_a",
            args.iters_a,
            0.0,
            0.0,
            0.0,
            3e-3,
            1e-2,
            5.0,
            0.15,
            {
                "skin_depth": 0.5,
                "cloth_inside": 0.1,
                "silhouette": 0.05,
                "fg_center": args.stage_a_center_anchor_weight,
                "shape_prior": 0.0,
                "beta_delta": 0.0,
                "pose_init": 0.0,
                "pose_consistency": 0.0,
                "transl_prior": 0.05,
                "orient_prior": 0.02,
            },
        ),
        StageConfig(
            "stage_b",
            args.iters_b,
            0.0,
            2e-3,
            1e-3,
            2e-3,
            5e-3,
            2.0,
            0.12,
            {
                "skin_depth": 0.8,
                "cloth_inside": 0.2,
                "silhouette": 0.05,
                "fg_center": 0.0,
                "shape_prior": 0.0,
                "beta_delta": 0.0,
                "pose_init": 0.02,
                "pose_consistency": 0.05,
                "transl_prior": 0.02,
                "orient_prior": 0.01,
            },
        ),
        StageConfig(
            "stage_c",
            args.iters_c,
            8e-4,
            1e-3,
            6e-4,
            1e-3,
            2e-3,
            1.0,
            0.08,
            {
                "skin_depth": 1.0,
                "cloth_inside": 0.3,
                "silhouette": 0.03,
                "fg_center": 0.0,
                "shape_prior": 0.02,
                "beta_delta": 0.01,
                "pose_init": 0.02,
                "pose_consistency": 0.05,
                "transl_prior": 0.01,
                "orient_prior": 0.005,
            },
        ),
        StageConfig(
            "stage_d",
            args.iters_d,
            2e-4,
            3e-4,
            2e-4,
            5e-4,
            1e-3,
            1.0,
            0.05,
            {
                "skin_depth": 1.0,
                "cloth_inside": 0.25,
                "silhouette": 0.02,
                "fg_center": 0.0,
                "shape_prior": 0.02,
                "beta_delta": 0.005,
                "pose_init": 0.015,
                "pose_consistency": 0.04,
                "transl_prior": 0.005,
                "orient_prior": 0.002,
            },
        ),
    ]


def stage_letter(stage_name: str) -> str:
    return stage_name.rsplit("_", 1)[-1].upper()


def stage_visibility_margin(args: argparse.Namespace, stage_name: str) -> float:
    letter = stage_letter(stage_name)
    return {
        "A": args.stage_a_near_surface_m,
        "B": args.stage_b_near_surface_m,
        "C": args.stage_c_near_surface_m,
        "D": args.stage_d_near_surface_m,
    }[letter]


def set_deterministic(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def beta_raw_from_beta(beta: Any, bound: float) -> Any:
    import torch

    x = (beta / bound).clamp(min=-0.999, max=0.999)
    return bound * torch.atanh(x)


def bounded_beta(beta_raw: Any, bound: float) -> Any:
    return bound * (beta_raw / bound).tanh()


def mask_bbox_center_xy(mask: np.ndarray, fallback_hw: tuple[int, int]) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        h, w = fallback_hw
        return np.array([0.5 * (w - 1), 0.5 * (h - 1)], dtype=np.float32)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=np.float32)


def make_tensor_obs(obs: dict[str, np.ndarray], regions: dict[str, np.ndarray], fg_dist: np.ndarray, device: Any) -> dict[str, Any]:
    import torch

    def t(arr: np.ndarray, channels: bool = False) -> Any:
        arr = arr.astype(np.float32)
        if channels:
            arr = np.moveaxis(arr, -1, 0)
        else:
            arr = arr[None]
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)[None]

    return {
        "xyz": t(obs["xyz_obs"], channels=True),
        "normal": t(obs["normal_obs"], channels=True),
        "valid": t(obs["valid_mask"].astype(np.float32)),
        "skin": t(regions["skin_mask"].astype(np.float32)),
        "cloth": t(regions["cloth_mask"].astype(np.float32)),
        "shoe": t(regions["shoe_mask"].astype(np.float32)),
        "fg": t((regions["skin_mask"] | regions["cloth_mask"] | regions["shoe_mask"]).astype(np.float32)),
        "fg_dist": t(fg_dist.astype(np.float32)),
        "fg_center_xy": torch.from_numpy(
            mask_bbox_center_xy(
                regions["skin_mask"] | regions["cloth_mask"] | regions["shoe_mask"],
                tuple(int(x) for x in obs["valid_mask"].shape),
            )
        ).to(device=device, dtype=torch.float32),
    }


def foreground_distance(fg: np.ndarray) -> np.ndarray:
    try:
        import cv2

        inv = (~fg.astype(bool)).astype(np.uint8)
        return cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    except ImportError:
        yy, xx = np.indices(fg.shape)
        inside = np.argwhere(fg)
        if len(inside) == 0:
            return np.full(fg.shape, 999.0, dtype=np.float32)
        # Slow fallback, intended only for tiny tests.
        d2 = np.full(fg.shape, np.inf, dtype=np.float32)
        for y, x in inside[:: max(1, len(inside) // 500)]:
            d2 = np.minimum(d2, (yy - y) ** 2 + (xx - x) ** 2)
        return np.sqrt(d2).astype(np.float32)


def smpl_forward(model: Any, betas: Any, body_pose: Any, orient: Any, transl: Any) -> Any:
    return model(
        betas=betas,
        body_pose=body_pose.reshape(1, 69),
        global_orient=orient.reshape(1, 3),
        transl=transl.reshape(1, 3),
        return_verts=True,
    )


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_config_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        def emit(obj: Any, indent: int = 0) -> None:
            pad = " " * indent
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        f.write(f"{pad}{k}:\n")
                        emit(v, indent + 2)
                    else:
                        f.write(f"{pad}{k}: {v}\n")
            elif isinstance(obj, list):
                for v in obj:
                    if isinstance(v, (dict, list)):
                        f.write(f"{pad}-\n")
                        emit(v, indent + 2)
                    else:
                        f.write(f"{pad}- {v}\n")
        emit(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="II-10 STAND 4-view joint SMPL fitting.")
    parser.add_argument("--init-npz", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--partseg-dir", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--k-json", type=Path, required=True)
    parser.add_argument("--smpl-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-y-axis", choices=("up", "down"), required=True)
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visibility", choices=("approx", "pytorch3d"), default="approx")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--beta-bound", type=float, default=3.0)
    parser.add_argument("--cloth-margin-m", type=float, default=0.005)
    parser.add_argument("--max-model-points-per-view", type=int, default=20000)
    parser.add_argument("--min-skin-samples-per-view", type=int, default=300)
    parser.add_argument("--min-total-body-samples-per-view", type=int, default=1000)
    parser.add_argument("--iters-a", type=int, default=5)
    parser.add_argument("--iters-b", type=int, default=300)
    parser.add_argument("--iters-c", type=int, default=400)
    parser.add_argument("--iters-d", type=int, default=150)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--audit-only", action="store_true", help="Validate inputs and sampling without optimization.")
    parser.add_argument("--allow-png-label-fallback", action="store_true")
    parser.add_argument("--max-stage", choices=("A", "B", "C", "D"), default="D")
    parser.add_argument("--target-height-m", type=float, default=1.810)
    parser.add_argument("--height-prior-weight", type=float, default=0.0)
    parser.add_argument("--foot-pose-prior-mult", type=float, default=4.0)
    parser.add_argument("--precalibrate-beta-height", action="store_true")
    parser.add_argument("--height-calib-iters", type=int, default=100)
    parser.add_argument("--height-calib-lr", type=float, default=5e-3)
    parser.add_argument("--height-calib-weight", type=float, default=10.0)
    parser.add_argument("--height-calib-beta-prior", type=float, default=0.0)
    parser.add_argument("--height-calib-shape-prior", type=float, default=0.01)
    parser.add_argument("--sample-drop-ratio-guard", type=float, default=0.35)
    parser.add_argument("--min-body-samples-per-view", type=int, default=800)
    parser.add_argument("--stage-a-near-surface-m", type=float, default=0.30)
    parser.add_argument("--stage-b-near-surface-m", type=float, default=0.20)
    parser.add_argument("--stage-c-near-surface-m", type=float, default=0.12)
    parser.add_argument("--stage-d-near-surface-m", type=float, default=0.08)
    parser.add_argument("--visibility-retry-growth", type=float, default=1.5)
    parser.add_argument("--visibility-retry-max-m", type=float, default=0.30)
    parser.add_argument("--stage-a-use-foreground-for-correspondence", action="store_true")
    parser.add_argument("--stage-a-center-anchor-weight", type=float, default=0.10)
    parser.add_argument("--stage-a-max-transl-delta-m", type=float, default=0.10)
    parser.add_argument("--stage-a-max-orient-delta-rad", type=float, default=0.20)
    parser.add_argument("--stage-a-min-initial-skin-ratio", type=float, default=0.45)
    parser.add_argument("--stage-a-min-initial-body-ratio", type=float, default=0.55)
    parser.add_argument("--use-symmetric-depth-window", action="store_true")
    parser.add_argument("--allow-low-samples-debug", action="store_true")
    parser.add_argument("--skin-warning-patience", type=int, default=3)
    return parser.parse_args()


def bbox_yx(mask: np.ndarray) -> list[int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return [int(y0), int(x0), int(y1), int(x1)]


def save_alignment_debug(
    path: Path,
    rgb_path: Path | None,
    label_map: np.ndarray,
    depth_valid: np.ndarray,
    regions: dict[str, Any],
) -> None:
    from PIL import Image

    h, w = depth_valid.shape
    if rgb_path is not None and rgb_path.exists():
        rgb = Image.open(rgb_path).convert("RGB").resize((w, h))
        rgb_np = np.asarray(rgb, dtype=np.float32)
    else:
        rgb_np = np.zeros((h, w, 3), dtype=np.float32) + 40
    overlay = rgb_np.copy()
    colors = [
        (regions["skin_mask"], np.array([0, 255, 0], dtype=np.float32)),
        (regions["cloth_mask"], np.array([0, 80, 255], dtype=np.float32)),
        (regions["shoe_mask"], np.array([255, 150, 0], dtype=np.float32)),
    ]
    for mask, color in colors:
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * color
    invalid = ~depth_valid
    overlay[invalid] = 0.75 * overlay[invalid] + 0.25 * np.array([255, 0, 0], dtype=np.float32)
    # Add faint label boundaries by comparing 4-neighbours.
    edge = np.zeros((h, w), dtype=bool)
    edge[:, 1:] |= label_map[:, 1:] != label_map[:, :-1]
    edge[1:, :] |= label_map[1:, :] != label_map[:-1, :]
    overlay[edge] = np.array([255, 255, 255], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.visibility == "pytorch3d":
        print("[warning] pytorch3d visibility requested but v1 uses approx unless an optional renderer is added.")

    import torch
    import smplx as smpl_backend

    set_deterministic(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu.")

    init = np.load(args.init_npz, allow_pickle=False)
    view_names = [str(x) for x in init["view_names"].tolist()]
    if len(view_names) != 4:
        raise ValueError(f"II-10 STAND requires exactly 4 views, got {len(view_names)}.")
    for token in VIEW_ORDER:
        if not any(token in v.lower() for v in view_names):
            raise ValueError(f"II-10 expected a {token} view in view_names={view_names}.")

    k_by_view, _ = load_k_by_view(args.k_json, view_names)
    k_np = np.stack([k_by_view[v] for v in view_names], axis=0).astype(np.float32)
    image_hws = init["image_hw_per_view"].astype(np.int32)
    if args.depth_y_axis != str(init["source_depth_y_axis"].item()):
        raise ValueError(
            f"--depth-y-axis {args.depth_y_axis} does not match II-8 metadata {init['source_depth_y_axis']}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        "checkpoints": args.output_dir / "checkpoints",
        "meshes": args.output_dir / "meshes",
        "overlays": args.output_dir / "overlays",
        "regions": args.output_dir / "regions",
        "shoe_export": args.output_dir / "shoe_export",
        "diagnostics": args.output_dir / "diagnostics",
        "audit": args.output_dir / "audit",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    observations: list[dict[str, Any]] = []
    obs_tensors: list[dict[str, Any]] = []
    sample_count_diags: dict[str, Any] = {}
    shoe_summary: dict[str, Any] = {"metadata": load_metadata(args.partseg_dir), "views": []}
    partseg_sources: dict[str, str] = {}
    audit_records: dict[str, dict[str, Any]] = {}

    for i, view_name in enumerate(view_names):
        depth_path = find_depth_path(args.depth_dir, view_name)
        obs = load_depth_observation(depth_path, k_np[i], args.depth_y_axis)
        if tuple(obs["valid_mask"].shape) != tuple(int(x) for x in image_hws[i]):
            raise ValueError(f"{view_name}: depth shape {obs['valid_mask'].shape} != image HW {tuple(image_hws[i])}.")
        regions = load_partseg(
            args.partseg_dir,
            view_name,
            obs["valid_mask"],
            allow_png_label_fallback=args.allow_png_label_fallback,
        )
        partseg_sources[view_name] = str(regions["partseg_source"])
        save_region_debug(dirs["regions"] / f"{view_name}_region_debug.png", regions)
        save_region_debug(dirs["audit"] / f"{view_name}_region_debug.png", regions)
        rgb_path = find_rgb_path(args.rgb_dir, view_name, f"{view_name}.jpg") if args.rgb_dir else None
        save_alignment_debug(
            dirs["audit"] / f"{view_name}_rgb_partseg_depth_alignment.png",
            rgb_path,
            regions["label_map"],
            obs["valid_mask"],
            regions,
        )
        shoe_points = shoe_points_from_xyz(obs["xyz_obs"], regions["shoe_mask"])
        left_shoe_points = shoe_points_from_xyz(obs["xyz_obs"], regions["left_shoe_mask"])
        right_shoe_points = shoe_points_from_xyz(obs["xyz_obs"], regions["right_shoe_mask"])
        np.save(dirs["shoe_export"] / f"{view_name}_shoe_points.npy", shoe_points)
        np.save(dirs["shoe_export"] / f"{view_name}_left_shoe_points.npy", left_shoe_points)
        np.save(dirs["shoe_export"] / f"{view_name}_right_shoe_points.npy", right_shoe_points)
        save_binary_mask_png(dirs["shoe_export"] / f"{view_name}_left_shoe_mask.png", regions["left_shoe_mask"])
        save_binary_mask_png(dirs["shoe_export"] / f"{view_name}_right_shoe_mask.png", regions["right_shoe_mask"])
        left_visible_ratio = float(regions["left_shoe_mask"].sum() / max(int(regions["fgmask"].sum()), 1))
        right_visible_ratio = float(regions["right_shoe_mask"].sum() / max(int(regions["fgmask"].sum()), 1))
        shoe_summary["views"].append(
            {
                "view_name": view_name,
                "shoe_points": int(len(shoe_points)),
                "left_shoe_point_count": int(len(left_shoe_points)),
                "right_shoe_point_count": int(len(right_shoe_points)),
                "left_shoe_bbox_yx": bbox_yx(regions["left_shoe_mask"]),
                "right_shoe_bbox_yx": bbox_yx(regions["right_shoe_mask"]),
                "left_visible_ratio": left_visible_ratio,
                "right_visible_ratio": right_visible_ratio,
                "occlusion_warning": bool(len(left_shoe_points) < 100 or len(right_shoe_points) < 100),
                "shoe_bbox_yx": bbox_yx(regions["shoe_mask"]),
            }
        )
        fg = regions["skin_mask"] | regions["cloth_mask"] | regions["shoe_mask"]
        fg_dist = foreground_distance(fg)
        observations.append({"depth_path": str(depth_path), "obs": obs, "regions": regions, "fg_dist": fg_dist})
        obs_tensors.append(make_tensor_obs(obs, regions, fg_dist, device))
        sample_count_diags[view_name] = {
            "skin_pixels": int(regions["skin_mask"].sum()),
            "cloth_pixels": int(regions["cloth_mask"].sum()),
            "shoe_pixels": int(regions["shoe_mask"].sum()),
            "valid_depth_pixels": int(obs["valid_mask"].sum()),
            "normal_diagnostics": obs["normal_diagnostics"],
            "partseg_source": regions["partseg_source"],
        }
        audit_records[view_name] = {
            "label_histogram": regions["label_histogram"],
            "region_counts": {
                "skin_pixel_count": int(regions["skin_mask"].sum()),
                "cloth_pixel_count": int(regions["cloth_mask"].sum()),
                "shoe_pixel_count": int(regions["shoe_mask"].sum()),
                "ignore_pixel_count": int(regions["ignore_mask"].sum()),
            },
            "depth_valid_counts": {
                "valid_depth_pixel_count": int(obs["valid_mask"].sum()),
                "invalid_depth_pixel_count": int((~obs["valid_mask"]).sum()),
            },
            "region_depth_intersections": {
                "skin_valid_depth_count": int((regions["skin_mask"] & obs["valid_mask"]).sum()),
                "cloth_valid_depth_count": int((regions["cloth_mask"] & obs["valid_mask"]).sum()),
                "shoe_valid_depth_count": int((regions["shoe_mask"] & obs["valid_mask"]).sum()),
            },
            "partseg_source": regions["partseg_source"],
        }
        write_json(dirs["audit"] / f"{view_name}_label_histogram.json", audit_records[view_name]["label_histogram"])
        write_json(dirs["audit"] / f"{view_name}_region_counts.json", audit_records[view_name]["region_counts"])
        write_json(dirs["audit"] / f"{view_name}_depth_valid_counts.json", audit_records[view_name]["depth_valid_counts"])
        write_json(
            dirs["audit"] / f"{view_name}_region_depth_intersections.json",
            audit_records[view_name]["region_depth_intersections"],
        )
    with (dirs["shoe_export"] / "shoe_summary.json").open("w", encoding="utf-8") as f:
        json.dump(shoe_summary, f, ensure_ascii=False, indent=2)
    with (dirs["shoe_export"] / "shoe_visibility_summary.json").open("w", encoding="utf-8") as f:
        json.dump(shoe_summary, f, ensure_ascii=False, indent=2)
    with (dirs["diagnostics"] / "per_view_sample_counts.json").open("w", encoding="utf-8") as f:
        json.dump(sample_count_diags, f, ensure_ascii=False, indent=2)

    model = smpl_backend.create(
        model_path=str(args.smpl_model_dir),
        model_type="smpl",
        gender=args.gender,
        num_betas=10,
        use_pca=False,
        create_transl=False,
        batch_size=1,
    ).to(device).eval()
    faces_np = np.asarray(model.faces, dtype=np.int64)
    faces_t = torch.from_numpy(faces_np).to(device=device, dtype=torch.long)
    k_t = torch.from_numpy(k_np).to(device=device, dtype=torch.float32)

    betas_init = torch.from_numpy(init["betas_shared_init"].astype(np.float32)).to(device).reshape(1, 10)
    body_init = torch.from_numpy(init["body_pose_init"].astype(np.float32)).to(device)
    orient_init = torch.from_numpy(init["global_orient_init"].astype(np.float32)).to(device)
    transl_init = torch.from_numpy(init["transl_init"].astype(np.float32)).to(device)

    body_base_init = body_init.mean(dim=0)
    body_delta_init = body_init - body_base_init[None]
    beta_raw = beta_raw_from_beta(betas_init, args.beta_bound).detach().clone().requires_grad_(False)
    body_pose_base = body_base_init.detach().clone().requires_grad_(False)
    body_pose_delta = body_delta_init.detach().clone().requires_grad_(False)
    global_orient = orient_init.detach().clone().requires_grad_(False)
    transl = transl_init.detach().clone().requires_grad_(False)

    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        beta_raw.data.copy_(ckpt["beta_raw"])
        body_pose_base.data.copy_(ckpt["body_pose_base"])
        body_pose_delta.data.copy_(ckpt["body_pose_delta"])
        global_orient.data.copy_(ckpt["global_orient"])
        transl.data.copy_(ckpt["transl"])

    all_stage_configs = default_stage_configs(args)
    max_stage_index = {"A": 0, "B": 1, "C": 2, "D": 3}[args.max_stage]
    stage_configs = all_stage_configs[: max_stage_index + 1]
    runtime_config = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "command_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "stage_configs": [asdict(s) for s in stage_configs],
            "partseg_sources": partseg_sources,
        }
    write_config_yaml(args.output_dir / "final_config.yaml", runtime_config)

    loss_csv = args.output_dir / "loss_history.csv"
    csv_file = loss_csv.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "stage",
            "iter",
            "loss_total",
            "skin_depth",
            "cloth_inside",
            "silhouette",
            "fg_center",
            "shape_prior",
            "beta_delta",
            "pose_init",
            "pose_consistency",
            "transl_prior",
            "orient_prior",
            "height_prior",
            "height_prior_raw",
            "height_prior_weighted",
            "per_view_json",
        ],
    )
    csv_writer.writeheader()
    sample_history_csv = dirs["diagnostics"] / "sample_history.csv"
    sample_csv_file = sample_history_csv.open("w", newline="", encoding="utf-8")
    sample_csv_writer = csv.DictWriter(
        sample_csv_file,
        fieldnames=[
            "stage",
            "iter",
            "view_name",
            "skin_samples",
            "cloth_samples",
            "body_samples",
            "shoe_samples",
            "foreground_body_samples",
            "visibility_margin_used",
            "step_rejected",
            "reject_reason",
        ],
    )
    sample_csv_writer.writeheader()

    best_overall = {"loss": math.inf}
    sample_guard_by_stage: dict[str, Any] = {}
    visibility_retry_by_stage: dict[str, Any] = {}

    def current_betas() -> Any:
        return bounded_beta(beta_raw, args.beta_bound)

    def current_body_pose() -> Any:
        return body_pose_base[None] + body_pose_delta

    def current_vertices_per_view() -> list[np.ndarray]:
        verts: list[np.ndarray] = []
        with torch.inference_mode():
            betas = current_betas()
            body_pose = current_body_pose()
            for i in range(len(view_names)):
                out = smpl_forward(model, betas, body_pose[i], global_orient[i], transl[i])
                verts.append(out.vertices[0].detach().cpu().numpy().astype(np.float32))
        return verts

    def height_entry() -> dict[str, Any]:
        report = height_report(current_vertices_per_view(), real_height_m=args.target_height_m)
        report["betas"] = current_betas().detach().cpu().numpy().reshape(10).astype(float).tolist()
        return report

    height_by_stage: dict[str, Any] = {"init_before_precalib": height_entry(), "init": height_entry()}

    def checkpoint_payload(stage_name: str, loss_value: float) -> dict[str, Any]:
        return {
            "stage": stage_name,
            "loss": loss_value,
            "beta_raw": beta_raw.detach().clone(),
            "body_pose_base": body_pose_base.detach().clone(),
            "body_pose_delta": body_pose_delta.detach().clone(),
            "global_orient": global_orient.detach().clone(),
            "transl": transl.detach().clone(),
        }

    def restore_payload(payload: dict[str, Any]) -> None:
        beta_raw.data.copy_(payload["beta_raw"])
        body_pose_base.data.copy_(payload["body_pose_base"])
        body_pose_delta.data.copy_(payload["body_pose_delta"])
        global_orient.data.copy_(payload["global_orient"])
        transl.data.copy_(payload["transl"])

    def total_loss(stage: StageConfig, visibility_margin_override: float | None = None) -> tuple[Any, dict[str, Any]]:
        betas = current_betas()
        body_pose = body_pose_base[None] + body_pose_delta
        terms = {k: torch.zeros((), device=device) for k in stage.weights}
        height_values: list[Any] = []
        per_view: list[dict[str, Any]] = []
        visibility_margin = (
            float(visibility_margin_override)
            if visibility_margin_override is not None
            else stage_visibility_margin(args, stage.name)
        )

        for i, view_name in enumerate(view_names):
            out = smpl_forward(model, betas, body_pose[i], global_orient[i], transl[i])
            height_values.append(out.vertices[0, :, 1].max() - out.vertices[0, :, 1].min())
            pts = surface_points(out.vertices[0], faces_t, args.max_model_points_per_view)
            is_stage_a = stage.name == "stage_a"
            gathered = gather_projected_observation(
                pts,
                k_t[i],
                obs_tensors[i],
                tuple(int(x) for x in image_hws[i]),
                visibility_margin_m=visibility_margin,
                use_foreground_for_body=is_stage_a and args.stage_a_use_foreground_for_correspondence,
                use_symmetric_depth_window=args.use_symmetric_depth_window,
            )
            skin = gathered["skin_mask"]
            cloth = gathered["cloth_mask"]
            body = gathered["body_mask"]
            skin_count = int(skin.detach().sum().cpu())
            body_count = int(body.detach().sum().cpu())
            min_body = min(args.min_body_samples_per_view, args.min_total_body_samples_per_view)
            if skin_count < args.min_skin_samples_per_view or body_count < min_body:
                if len(per_view) == 0:
                    print(
                        f"[warning] low samples may weaken fitting: {view_name} skin={skin_count}, body={body_count}"
                    )
            skin_l = skin_depth_loss(
                gathered["p_model"], gathered["p_obs"], gathered["n_obs"], skin, stage.residual_clamp
            )
            if is_stage_a and args.stage_a_use_foreground_for_correspondence:
                fg = gathered["foreground_body_mask"]
                fg_z = robust_clamped(gathered["z_model"] - gathered["p_obs"][:, 2], stage.residual_clamp)
                fg_l = fg_z[fg].mean() if int(fg.detach().sum().cpu()) else pts.sum() * 0.0
                skin_l = 0.7 * skin_l + 0.3 * fg_l
            cloth_l = cloth_inside_loss(
                gathered["z_model"], gathered["p_obs"][:, 2], cloth, stage.residual_clamp, args.cloth_margin_m
            )
            sil_candidates = gathered["inside_mask"]
            sil = gathered["silhouette_distance"][sil_candidates]
            sil_l = sil.mean() / max(float(max(image_hws[i])), 1.0) if sil.numel() else pts.sum() * 0.0
            if int(sil_candidates.detach().sum().cpu()) > 0:
                u_in = gathered["u"][sil_candidates]
                v_in = gathered["v"][sil_candidates]
                model_center = torch.stack(
                    [
                        0.5 * (u_in.min() + u_in.max()),
                        0.5 * (v_in.min() + v_in.max()),
                    ]
                )
                target_center = obs_tensors[i]["fg_center_xy"]
                w_norm = max(float(image_hws[i][1]), 1.0)
                h_norm = max(float(image_hws[i][0]), 1.0)
                center_l = (
                    ((model_center[0] - target_center[0]) / w_norm).square()
                    + ((model_center[1] - target_center[1]) / h_norm).square()
                )
                center_error_px = (model_center - target_center).detach().abs().cpu().numpy().astype(float).tolist()
            else:
                center_l = pts.sum() * 0.0 + 1.0
                center_error_px = [math.inf, math.inf]
            terms["skin_depth"] = terms["skin_depth"] + skin_l
            terms["cloth_inside"] = terms["cloth_inside"] + cloth_l
            terms["silhouette"] = terms["silhouette"] + sil_l
            terms["fg_center"] = terms["fg_center"] + center_l
            per_view.append(
                {
                    "view_name": view_name,
                    "skin_samples": skin_count,
                    "cloth_samples": int(cloth.detach().sum().cpu()),
                    "shoe_samples": int(gathered["shoe_mask"].detach().sum().cpu()),
                    "body_samples": body_count,
                    "foreground_body_samples": int(gathered["foreground_body_mask"].detach().sum().cpu()),
                    "inside_samples": int(gathered["inside_mask"].detach().sum().cpu()),
                    "valid_depth_samples": int(gathered["valid_depth_mask"].detach().sum().cpu()),
                    "fg_samples": int(gathered["fg_mask"].detach().sum().cpu()),
                    "visibility_margin_used": visibility_margin,
                    "skin_depth": float(skin_l.detach().cpu()),
                    "cloth_inside": float(cloth_l.detach().cpu()),
                    "silhouette": float(sil_l.detach().cpu()),
                    "fg_center": float(center_l.detach().cpu()),
                    "fg_center_error_px": center_error_px,
                }
            )

        for k in ("skin_depth", "cloth_inside", "silhouette", "fg_center"):
            terms[k] = terms[k] / len(view_names)
        terms["shape_prior"] = betas.square().mean()
        terms["beta_delta"] = (betas - betas_init).square().mean()
        terms["pose_init"] = (body_pose - body_init).square().mean()
        terms["pose_consistency"] = body_pose_delta.square().mean()
        # Stronger foot/ankle regularization: SMPL feet should not chase shoes.
        foot_joint_indices = [6, 7, 9, 10]
        terms["pose_init"] = terms["pose_init"] + 2.0 * (
            args.foot_pose_prior_mult
            * (
            body_pose[:, foot_joint_indices] - body_init[:, foot_joint_indices]
            ).square().mean()
        )
        terms["transl_prior"] = (transl - transl_init).square().mean()
        terms["orient_prior"] = (global_orient - orient_init).square().mean()
        mesh_height_mean = torch.stack(height_values).mean()
        height_prior = (mesh_height_mean - args.target_height_m) ** 2
        height_prior_weighted = (
            args.height_prior_weight * height_prior if stage.name in {"stage_c", "stage_d"} else height_prior * 0.0
        )

        loss = torch.zeros((), device=device)
        for name, value in terms.items():
            loss = loss + stage.weights[name] * value
        loss = loss + height_prior_weighted
        stats = {name: float(value.detach().cpu()) for name, value in terms.items()}
        stats["height_prior"] = float(height_prior.detach().cpu())
        stats["height_prior_raw"] = float(height_prior.detach().cpu())
        stats["height_prior_weighted"] = float(height_prior_weighted.detach().cpu())
        stats["mesh_height_mean_m"] = float(mesh_height_mean.detach().cpu())
        stats["loss_total"] = float(loss.detach().cpu())
        stats["per_view"] = per_view
        return loss, stats

    audit_stage = stage_configs[0] if stage_configs else all_stage_configs[0]
    _, audit_stats = total_loss(audit_stage)
    insufficient_views: list[dict[str, Any]] = []
    for view_stats in audit_stats["per_view"]:
        view_name = view_stats["view_name"]
        sample_payload = {
            "final_sampled_skin_count": int(view_stats["skin_samples"]),
            "final_sampled_cloth_count": int(view_stats["cloth_samples"]),
            "final_sampled_shoe_count": int(view_stats["shoe_samples"]),
            "total_body_samples": int(view_stats["body_samples"]),
            "partseg_source": partseg_sources[view_name],
        }
        audit_records[view_name]["sample_counts"] = sample_payload
        write_json(dirs["audit"] / f"{view_name}_sample_counts.json", sample_payload)
        if (
            audit_records[view_name]["region_depth_intersections"]["skin_valid_depth_count"] < args.min_skin_samples_per_view
            or sample_payload["total_body_samples"] < args.min_total_body_samples_per_view
        ):
            insufficient_views.append(
                {
                    "view_name": view_name,
                    "skin_valid_depth_count": audit_records[view_name]["region_depth_intersections"][
                        "skin_valid_depth_count"
                    ],
                    "total_body_samples": sample_payload["total_body_samples"],
                }
            )
    audit_summary = {
        "partseg_sources": partseg_sources,
        "insufficient_views": insufficient_views,
        "per_view": audit_records,
    }
    write_json(dirs["audit"] / "audit_summary.json", audit_summary)
    if insufficient_views:
        print(f"[AUDIT WARNING] insufficient samples: {insufficient_views}")
    if args.audit_only:
        csv_file.close()
        sample_csv_file.close()
        print("\nII-10 audit complete")
        print(f"  audit dir: {dirs['audit']}")
        print(f"  partseg sources: {partseg_sources}")
        print(f"  insufficient views: {insufficient_views}")
        return

    def run_stage_0_height_precalib() -> dict[str, Any] | None:
        if not args.precalibrate_beta_height:
            height_by_stage["after_precalib"] = height_entry()
            return None
        beta_raw.requires_grad_(True)
        body_pose_base.requires_grad_(False)
        body_pose_delta.requires_grad_(False)
        global_orient.requires_grad_(False)
        transl.requires_grad_(False)
        opt = torch.optim.Adam([beta_raw], lr=args.height_calib_lr, amsgrad=True)
        best = {"loss": math.inf, "payload": checkpoint_payload("stage_0_height", math.inf)}
        print(f"\n[stage_0_height] iters={args.height_calib_iters} target={args.target_height_m:.3f}m")
        for it in range(1, args.height_calib_iters + 1):
            opt.zero_grad(set_to_none=True)
            betas = current_betas()
            body_pose = current_body_pose()
            heights = []
            for i in range(len(view_names)):
                out = smpl_forward(model, betas, body_pose[i], global_orient[i], transl[i])
                heights.append(out.vertices[0, :, 1].max() - out.vertices[0, :, 1].min())
            mean_h = torch.stack(heights).mean()
            height_loss = (mean_h - args.target_height_m) ** 2
            beta_prior = (betas - betas_init).square().mean()
            shape_prior = betas.square().mean()
            loss = (
                args.height_calib_weight * height_loss
                + args.height_calib_beta_prior * beta_prior
                + args.height_calib_shape_prior * shape_prior
            )
            if not torch.isfinite(loss):
                raise RuntimeError("stage_0_height: NaN/Inf loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([beta_raw], 1.0)
            opt.step()
            value = float(loss.detach().cpu())
            if value < best["loss"]:
                best = {"loss": value, "payload": checkpoint_payload("stage_0_height", value)}
                save_checkpoint(dirs["checkpoints"] / "best_stage_0_height.pt", best["payload"])
            if it == 1 or it % args.log_every == 0 or it == args.height_calib_iters:
                print(
                    f"  {it:4d} loss={value:.6f} height={float(mean_h.detach().cpu()):.4f} "
                    f"beta_prior={float(beta_prior.detach().cpu()):.6f} |beta|max={float(betas.abs().max().detach().cpu()):.3f}"
                )
        restore_payload(best["payload"])
        report = height_entry()
        height_by_stage["after_precalib"] = report
        write_json(dirs["diagnostics"] / "precalib_height_report.json", report)
        return {"stage": "stage_0_height", "loss": best["loss"], "height_report": report}

    precalib_summary = run_stage_0_height_precalib()

    def run_stage(stage: StageConfig) -> dict[str, Any]:
        nonlocal best_overall
        beta_raw.requires_grad_(stage.lr_betas > 0)
        body_pose_base.requires_grad_(stage.lr_pose_base > 0)
        body_pose_delta.requires_grad_(stage.lr_pose_delta > 0)
        global_orient.requires_grad_(stage.lr_orient > 0)
        transl.requires_grad_(stage.lr_transl > 0)

        params = []
        if beta_raw.requires_grad:
            params.append({"params": [beta_raw], "lr": stage.lr_betas})
        if body_pose_base.requires_grad:
            params.append({"params": [body_pose_base], "lr": stage.lr_pose_base})
        if body_pose_delta.requires_grad:
            params.append({"params": [body_pose_delta], "lr": stage.lr_pose_delta})
        if global_orient.requires_grad:
            params.append({"params": [global_orient], "lr": stage.lr_orient})
        if transl.requires_grad:
            params.append({"params": [transl], "lr": stage.lr_transl})
        if not params or stage.iters <= 0:
            loss, stats = total_loss(stage)
            return {"stage": stage.name, "iters": 0, **stats}

        opt = torch.optim.Adam(params, amsgrad=True)
        best_stage = {"loss": math.inf}
        nan_count = 0
        guard_events: list[dict[str, Any]] = []
        retry_events: list[dict[str, Any]] = []
        _, accepted_stats = total_loss(stage)
        accepted_counts = {
            v["view_name"]: {"skin": int(v["skin_samples"]), "body": int(v["body_samples"])}
            for v in accepted_stats["per_view"]
        }
        stage_initial_counts = {
            view_name: {"skin": values["skin"], "body": values["body"]}
            for view_name, values in accepted_counts.items()
        }
        stage_start_orient = global_orient.detach().clone()
        stage_start_transl = transl.detach().clone()
        low_skin_streak = {view_name: 0 for view_name in view_names}
        last_stats: dict[str, Any] = {}
        print(f"\n[{stage.name}] iters={stage.iters}")
        for it in range(1, stage.iters + 1):
            before_payload = checkpoint_payload(stage.name, best_stage["loss"])
            opt.zero_grad(set_to_none=True)
            loss, stats = total_loss(stage)
            if not torch.isfinite(loss):
                nan_count += 1
                if best_stage.get("payload") is not None:
                    restore_payload(best_stage["payload"])
                for group in opt.param_groups:
                    group["lr"] *= 0.5
                if nan_count > 3:
                    raise RuntimeError(f"{stage.name}: repeated NaN/Inf loss.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for group in params for p in group["params"] if p.grad is not None], stage.grad_clip
            )
            opt.step()
            _, post_stats = total_loss(stage)
            def evaluate_guard(stats_for_guard: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
                reject_reasons: list[dict[str, Any]] = []
                warnings: list[dict[str, Any]] = []
                for view_stats in stats_for_guard["per_view"]:
                    view_name = view_stats["view_name"]
                    prev = accepted_counts[view_name]
                    skin_now = int(view_stats["skin_samples"])
                    body_now = int(view_stats["body_samples"])
                    min_body = min(args.min_body_samples_per_view, args.min_total_body_samples_per_view)
                    if body_now < min_body:
                        warnings.append(
                            {"view_name": view_name, "reason": "body_below_min", "body_samples": body_now}
                        )
                        if not args.allow_low_samples_debug:
                            reject_reasons.append(
                                {"view_name": view_name, "reason": "body_below_min", "body_samples": body_now}
                            )
                    if skin_now < args.min_skin_samples_per_view:
                        warnings.append(
                            {"view_name": view_name, "reason": "skin_below_min", "skin_samples": skin_now}
                        )
                        if low_skin_streak[view_name] + 1 > args.skin_warning_patience:
                            reject_reasons.append(
                                {
                                    "view_name": view_name,
                                    "reason": "skin_below_min_repeated",
                                    "skin_samples": skin_now,
                                    "streak": low_skin_streak[view_name] + 1,
                                }
                            )
                    if skin_now < prev["skin"] * (1.0 - args.sample_drop_ratio_guard):
                        reject_reasons.append(
                            {
                                "view_name": view_name,
                                "reason": "skin_drop",
                                "prev_skin": prev["skin"],
                                "skin_samples": skin_now,
                            }
                        )
                    if body_now < prev["body"] * (1.0 - args.sample_drop_ratio_guard):
                        reject_reasons.append(
                            {
                                "view_name": view_name,
                                "reason": "body_drop",
                                "prev_body": prev["body"],
                                "body_samples": body_now,
                            }
                        )
                    if stage.name == "stage_a":
                        initial = stage_initial_counts[view_name]
                        min_stage_skin = int(math.ceil(initial["skin"] * args.stage_a_min_initial_skin_ratio))
                        min_stage_body = int(math.ceil(initial["body"] * args.stage_a_min_initial_body_ratio))
                        if skin_now < min_stage_skin:
                            reject_reasons.append(
                                {
                                    "view_name": view_name,
                                    "reason": "stage_a_initial_skin_floor",
                                    "initial_skin": initial["skin"],
                                    "min_skin": min_stage_skin,
                                    "skin_samples": skin_now,
                                }
                            )
                        if body_now < min_stage_body:
                            reject_reasons.append(
                                {
                                    "view_name": view_name,
                                    "reason": "stage_a_initial_body_floor",
                                    "initial_body": initial["body"],
                                    "min_body": min_stage_body,
                                    "body_samples": body_now,
                                }
                            )
                if stage.name == "stage_a":
                    transl_delta = torch.linalg.norm(transl.detach() - stage_start_transl, dim=1)
                    orient_delta = torch.linalg.norm(global_orient.detach() - stage_start_orient, dim=1)
                    for view_name, t_delta, o_delta in zip(view_names, transl_delta, orient_delta):
                        t_value = float(t_delta.cpu())
                        o_value = float(o_delta.cpu())
                        if t_value > args.stage_a_max_transl_delta_m:
                            reject_reasons.append(
                                {
                                    "view_name": view_name,
                                    "reason": "stage_a_transl_trust_region",
                                    "transl_delta_m": t_value,
                                    "max_transl_delta_m": args.stage_a_max_transl_delta_m,
                                }
                            )
                        if o_value > args.stage_a_max_orient_delta_rad:
                            reject_reasons.append(
                                {
                                    "view_name": view_name,
                                    "reason": "stage_a_orient_trust_region",
                                    "orient_delta_rad": o_value,
                                    "max_orient_delta_rad": args.stage_a_max_orient_delta_rad,
                                }
                            )
                return reject_reasons, warnings

            reject_reasons, warnings = evaluate_guard(post_stats)
            retry_used = False
            if reject_reasons or warnings:
                base_margin = stage_visibility_margin(args, stage.name)
                base_rejects = list(reject_reasons)
                base_warnings = list(warnings)
                retry_margin = min(base_margin * args.visibility_retry_growth, args.visibility_retry_max_m)
                retry_attempts: list[dict[str, Any]] = []
                best_relaxed: dict[str, Any] | None = None
                while retry_margin > base_margin and retry_margin <= args.visibility_retry_max_m + 1e-9:
                    _, relaxed_stats = total_loss(stage, visibility_margin_override=retry_margin)
                    relaxed_rejects, relaxed_warnings = evaluate_guard(relaxed_stats)
                    recovered = len(relaxed_rejects) == 0
                    attempt = {
                        "retry_margin": retry_margin,
                        "recovered": recovered,
                        "relaxed_rejects": relaxed_rejects,
                        "relaxed_warnings": relaxed_warnings,
                    }
                    retry_attempts.append(attempt)
                    if recovered:
                        candidate = {
                            "stats": relaxed_stats,
                            "rejects": relaxed_rejects,
                            "warnings": relaxed_warnings,
                            "margin": retry_margin,
                        }
                        if best_relaxed is None or len(relaxed_warnings) < len(best_relaxed["warnings"]):
                            best_relaxed = candidate
                        if not relaxed_warnings:
                            break
                    next_margin = min(retry_margin * args.visibility_retry_growth, args.visibility_retry_max_m)
                    if next_margin <= retry_margin + 1e-9:
                        break
                    retry_margin = next_margin
                if best_relaxed is not None:
                    post_stats = best_relaxed["stats"]
                    reject_reasons = best_relaxed["rejects"]
                    warnings = best_relaxed["warnings"]
                    retry_used = True
                if retry_attempts:
                    retry_events.append(
                        {
                            "stage": stage.name,
                            "iter": it,
                            "base_margin": base_margin,
                            "retry_margin": best_relaxed["margin"] if best_relaxed is not None else retry_attempts[-1]["retry_margin"],
                            "recovered": retry_used,
                            "base_rejects": base_rejects,
                            "base_warnings": base_warnings,
                            "attempts": retry_attempts,
                        }
                    )
                if reject_reasons:
                    restore_payload(before_payload)
                    for group in opt.param_groups:
                        group["lr"] *= 0.5
                    guard_events.append(
                        {
                            "iter": it,
                            "rejected": True,
                            "reasons": reject_reasons,
                            "warnings": warnings,
                            "retry_used": retry_used,
                        }
                    )
                    for view_stats in post_stats["per_view"]:
                        sample_csv_writer.writerow(
                            {
                                "stage": stage.name,
                                "iter": it,
                                "view_name": view_stats["view_name"],
                                "skin_samples": view_stats["skin_samples"],
                                "cloth_samples": view_stats["cloth_samples"],
                                "body_samples": view_stats["body_samples"],
                                "shoe_samples": view_stats["shoe_samples"],
                                "foreground_body_samples": view_stats["foreground_body_samples"],
                                "visibility_margin_used": view_stats["visibility_margin_used"],
                                "step_rejected": True,
                                "reject_reason": json.dumps(reject_reasons, ensure_ascii=False),
                            }
                        )
                    print(f"  [guard reject] iter={it} reasons={reject_reasons}")
                    continue
                if warnings:
                    guard_events.append(
                        {"iter": it, "rejected": False, "warnings": warnings, "retry_used": retry_used}
                    )
                    print(f"  [guard warning] iter={it} warnings={warnings}")
            stats = post_stats
            for view_stats in stats["per_view"]:
                view_name = view_stats["view_name"]
                if int(view_stats["skin_samples"]) < args.min_skin_samples_per_view:
                    low_skin_streak[view_name] += 1
                else:
                    low_skin_streak[view_name] = 0
                sample_csv_writer.writerow(
                    {
                        "stage": stage.name,
                        "iter": it,
                        "view_name": view_name,
                        "skin_samples": view_stats["skin_samples"],
                        "cloth_samples": view_stats["cloth_samples"],
                        "body_samples": view_stats["body_samples"],
                        "shoe_samples": view_stats["shoe_samples"],
                        "foreground_body_samples": view_stats["foreground_body_samples"],
                        "visibility_margin_used": view_stats["visibility_margin_used"],
                        "step_rejected": False,
                        "reject_reason": "",
                    }
                )
            accepted_counts = {
                v["view_name"]: {"skin": int(v["skin_samples"]), "body": int(v["body_samples"])}
                for v in stats["per_view"]
            }
            if stage.name == "stage_a" and any(streak >= 3 for streak in low_skin_streak.values()):
                print(f"  [early stop] {stage.name}: persistent low skin samples {low_skin_streak}")
                last_stats = stats
                break
            last_stats = stats
            loss_value = stats["loss_total"]
            payload = checkpoint_payload(stage.name, loss_value)
            if loss_value < best_stage["loss"]:
                best_stage = {"loss": loss_value, "payload": payload}
                save_checkpoint(dirs["checkpoints"] / f"best_{stage.name}.pt", payload)
            if loss_value < best_overall["loss"]:
                best_overall = {"loss": loss_value, "payload": payload}
                save_checkpoint(dirs["checkpoints"] / "best_overall.pt", payload)
            csv_writer.writerow(
                {
                    "stage": stage.name,
                    "iter": it,
                    "loss_total": loss_value,
                    **{k: stats[k] for k in stage.weights},
                    "height_prior": stats["height_prior"],
                    "height_prior_raw": stats["height_prior_raw"],
                    "height_prior_weighted": stats["height_prior_weighted"],
                    "per_view_json": json.dumps(stats["per_view"], ensure_ascii=False),
                }
            )
            if it == 1 or it % args.log_every == 0 or it == stage.iters:
                counts = [(v["view_name"], v["skin_samples"], v["body_samples"]) for v in stats["per_view"]]
                print(
                    f"  {it:4d} loss={loss_value:.6f} skin={stats['skin_depth']:.6f} "
                    f"cloth={stats['cloth_inside']:.6f} skin/body={counts}"
                )
        if best_stage.get("payload") is not None:
            restore_payload(best_stage["payload"])
        sample_guard_by_stage[stage.name] = {
            "rejected_step_count": int(sum(1 for event in guard_events if event.get("rejected"))),
            "warning_step_count": int(sum(1 for event in guard_events if not event.get("rejected"))),
            "events": guard_events,
            "sample_drop_ratio_guard": args.sample_drop_ratio_guard,
            "min_skin_samples_per_view": args.min_skin_samples_per_view,
            "min_body_samples_per_view": args.min_body_samples_per_view,
            "stage_a_min_initial_skin_ratio": args.stage_a_min_initial_skin_ratio if stage.name == "stage_a" else None,
            "stage_a_min_initial_body_ratio": args.stage_a_min_initial_body_ratio if stage.name == "stage_a" else None,
            "stage_a_max_transl_delta_m": args.stage_a_max_transl_delta_m if stage.name == "stage_a" else None,
            "stage_a_max_orient_delta_rad": args.stage_a_max_orient_delta_rad if stage.name == "stage_a" else None,
        }
        visibility_retry_by_stage[stage.name] = {
            "retry_count": len(retry_events),
            "recovered_count": int(sum(1 for event in retry_events if event.get("recovered"))),
            "events": retry_events,
        }
        if stage.name == "stage_a":
            write_json(dirs["diagnostics"] / "stage_a_sample_guard.json", sample_guard_by_stage[stage.name])
        return {"stage": stage.name, "iters": stage.iters, **last_stats}

    def export_stage_snapshot(stage_name: str) -> None:
        betas_np = current_betas().detach().cpu().numpy().reshape(10).astype(np.float32)
        body_base_np = body_pose_base.detach().cpu().numpy().astype(np.float32)
        body_delta_np = body_pose_delta.detach().cpu().numpy().astype(np.float32)
        body_np = body_base_np[None] + body_delta_np
        orient_np = global_orient.detach().cpu().numpy().astype(np.float32)
        transl_np = transl.detach().cpu().numpy().astype(np.float32)
        out_npz = args.output_dir / f"{stage_name}_smpl_params.npz"
        np.savez_compressed(
            out_npz,
            schema_version=np.array(SCHEMA_VERSION),
            pose_set=np.array("STAND"),
            view_names=np.asarray(view_names),
            betas_shared=betas_np,
            body_pose_base=body_base_np,
            body_pose_delta=body_delta_np,
            body_pose=body_np,
            global_orient=orient_np,
            transl=transl_np,
        )
        h = height_entry()
        write_json(args.output_dir / f"{stage_name}_height_report.json", h)
        overlay_dir = args.output_dir / f"{stage_name}_overlays"
        verts_list = current_vertices_per_view()
        for i, view_name in enumerate(view_names):
            short = next((token for token in VIEW_ORDER if token in view_name.lower()), view_name)
            rgb_path = find_rgb_path(args.rgb_dir, view_name, f"{view_name}.jpg") if args.rgb_dir else None
            save_overlay(rgb_path, overlay_dir / f"{short}_overlay.png", verts_list[i], k_np[i], f"{view_name} {stage_name}")
            if stage_name == "stage_a":
                save_overlay(
                    rgb_path,
                    dirs["overlays"] / f"{short}_stage_a_overlay.png",
                    verts_list[i],
                    k_np[i],
                    f"{view_name} Stage A",
                )

    stage_summaries = []
    final_stage_payload: dict[str, Any] | None = None
    for stage in stage_configs:
        stage_summaries.append(run_stage(stage))
        height_by_stage[stage.name] = height_entry()
        export_stage_snapshot(stage.name)
        final_stage_payload = checkpoint_payload(stage.name, stage_summaries[-1].get("loss_total", math.inf))
    if final_stage_payload is not None:
        restore_payload(final_stage_payload)
    height_by_stage["final"] = height_entry()
    csv_file.close()
    sample_csv_file.close()
    write_json(dirs["diagnostics"] / "sample_guard_by_stage.json", sample_guard_by_stage)
    write_json(dirs["diagnostics"] / "visibility_retry_by_stage.json", visibility_retry_by_stage)

    betas_final = current_betas().detach().cpu().numpy().reshape(10).astype(np.float32)
    body_base_final = body_pose_base.detach().cpu().numpy().astype(np.float32)
    body_delta_final = body_pose_delta.detach().cpu().numpy().astype(np.float32)
    body_final = body_base_final[None] + body_delta_final
    orient_final = global_orient.detach().cpu().numpy().astype(np.float32)
    transl_final = transl.detach().cpu().numpy().astype(np.float32)

    mesh_paths: list[str] = []
    vertices_per_view: list[np.ndarray] = []
    with torch.inference_mode():
        betas_t = current_betas()
        body_t = body_pose_base[None] + body_pose_delta
        for i, view_name in enumerate(view_names):
            out = smpl_forward(model, betas_t, body_t[i], global_orient[i], transl[i])
            verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
            vertices_per_view.append(verts)
            short = next((token for token in VIEW_ORDER if token in view_name.lower()), view_name)
            mesh_path = dirs["meshes"] / f"final_{short}.obj"
            write_obj(mesh_path, verts, faces_np)
            mesh_paths.append(str(mesh_path.resolve()))
            rgb_path = find_rgb_path(args.rgb_dir, view_name, f"{view_name}.jpg") if args.rgb_dir else None
            save_overlay(rgb_path, dirs["overlays"] / f"{short}_overlay.png", verts, k_np[i], f"{view_name} II-10")
            save_alignment_debug(
                dirs["overlays"] / f"{short}_silhouette_overlay.png",
                rgb_path,
                observations[i]["regions"]["label_map"],
                observations[i]["obs"]["valid_mask"],
                observations[i]["regions"],
            )
            pts = surface_points(out.vertices[0], faces_t, args.max_model_points_per_view)
            gathered = gather_projected_observation(
                pts,
                k_t[i],
                obs_tensors[i],
                tuple(int(x) for x in image_hws[i]),
                visibility_margin_m=args.stage_d_near_surface_m,
                use_foreground_for_body=False,
            )
            uv = torch.stack([gathered["u"], gathered["v"]], dim=-1).detach().cpu().numpy()
            save_overlay(
                rgb_path,
                dirs["overlays"] / f"{short}_sample_overlay.png",
                verts,
                k_np[i],
                f"{view_name} samples",
                skin_uv=uv[gathered["skin_mask"].detach().cpu().numpy()],
                cloth_uv=uv[gathered["cloth_mask"].detach().cpu().numpy()],
                shoe_uv=uv[gathered["shoe_mask"].detach().cpu().numpy()],
            )

    height_diag = height_report(vertices_per_view)
    leg_diag = leg_cross_section_report(vertices_per_view[0])
    with (dirs["diagnostics"] / "final_height_report.json").open("w", encoding="utf-8") as f:
        json.dump(height_diag, f, ensure_ascii=False, indent=2)
    with (dirs["diagnostics"] / "height_by_stage.json").open("w", encoding="utf-8") as f:
        json.dump(height_by_stage, f, ensure_ascii=False, indent=2)
    with (dirs["diagnostics"] / "leg_cross_section_report.json").open("w", encoding="utf-8") as f:
        json.dump(leg_diag, f, ensure_ascii=False, indent=2)
    with (dirs["diagnostics"] / "per_view_depth_residuals.json").open("w", encoding="utf-8") as f:
        json.dump({"stages": stage_summaries}, f, ensure_ascii=False, indent=2)

    metadata = {
        "seed": args.seed,
        "visibility": args.visibility,
        "depth_y_axis": args.depth_y_axis,
        "partseg_metadata": load_metadata(args.partseg_dir),
        "partseg_sources": partseg_sources,
        "png_fallback_used": any(source == "png_fallback" for source in partseg_sources.values()),
        "precalibration": precalib_summary,
        "stage_summaries": stage_summaries,
    }
    final_npz = args.output_dir / "final_smpl_params.npz"
    np.savez_compressed(
        final_npz,
        schema_version=np.array(SCHEMA_VERSION),
        pose_set=np.array("STAND"),
        view_names=np.asarray(view_names),
        K=k_np,
        image_hw_per_view=image_hws,
        coordinate_convention=init["coordinate_convention"],
        betas_shared_final=betas_final,
        body_pose_base_final=body_base_final,
        body_pose_delta_final=body_delta_final,
        body_pose_final=body_final.astype(np.float32),
        global_orient_final=orient_final,
        transl_final=transl_final,
        betas_shared_init=init["betas_shared_init"].astype(np.float32),
        body_pose_init=init["body_pose_init"].astype(np.float32),
        global_orient_init=init["global_orient_init"].astype(np.float32),
        transl_init=init["transl_init"].astype(np.float32),
        loss_history_path=np.array(str(loss_csv.resolve())),
        mesh_paths=np.asarray(mesh_paths),
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "final_smpl_params": str(final_npz.resolve()),
        "loss_history": str(loss_csv.resolve()),
        "mesh_paths": mesh_paths,
        "height_report": height_diag,
        "leg_cross_section_report": leg_diag,
        "shoe_export": str(dirs["shoe_export"].resolve()),
        "height_by_stage": str((dirs["diagnostics"] / "height_by_stage.json").resolve()),
        "png_fallback_used": any(source == "png_fallback" for source in partseg_sources.values()),
        "insufficient_views": insufficient_views,
        "manual_checks_required": [
            "inspect region debug images",
            "inspect SMPL overlays",
            "inspect final height report",
            "inspect leg cross-section report",
            "verify shoe pixels were excluded from body fitting",
        ],
    }
    with (args.output_dir / "fit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nII-10 complete")
    print(f"  final params : {final_npz}")
    print(f"  loss history : {loss_csv}")
    print(f"  meshes       : {dirs['meshes']}")
    print(f"  overlays     : {dirs['overlays']}")
    print(f"  regions      : {dirs['regions']}")
    print(f"  shoe export  : {dirs['shoe_export']}")
    print(f"  mean height  : {height_diag['mean_height_m']:.3f} m")
    print(f"  |betas|max   : {float(np.abs(betas_final).max()):.3f}")


if __name__ == "__main__":
    main()
