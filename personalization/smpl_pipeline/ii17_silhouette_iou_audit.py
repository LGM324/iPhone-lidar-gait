"""II-17 silhouette IoU audit for final single-SMPL candidates.

This projects a canonical final SMPL mesh back into each STAND source view and
compares the rendered mesh silhouette against Sapiens body masks. It is a
diagnostic only; it does not change fitting results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .depth_utils import load_depth_observation
from .ii8_init import find_depth_path, find_rgb_path
from .ii12_canonical_fusion import VIEW_ORDER, YAW_PRESETS, view_key
from .ii15_depth_error_audit import (
    canonical_to_view_camera,
    compute_y_alignment,
    load_ascii_ply_mesh,
    project_y_up_np,
)
from .region_map import CLOTH_LABELS, IGNORE_LABELS, SHOE_LABELS, SKIN_LABELS, load_partseg
from .view_corrections import apply_camera_correction_np, load_view_corrections


LOWER_BODY_LABELS = {7, 11, 12, 16, 20}
LEG_LABELS = {7, 11, 16, 20}
TORSO_CLOTH_LABELS = {1, 12, 21, 22}
ARM_HAND_HEAD_LABELS = {2, 3, 5, 6, 10, 14, 15, 19, 23, 24, 25, 26, 27}


def label_mask(label_map: np.ndarray, valid: np.ndarray, region: str) -> np.ndarray:
    if region == "body_no_shoe":
        labels = (SKIN_LABELS | CLOTH_LABELS) - SHOE_LABELS
    elif region == "torso_legs_no_shoe":
        labels = (TORSO_CLOTH_LABELS | LEG_LABELS) - SHOE_LABELS - ARM_HAND_HEAD_LABELS
    elif region == "lower_body_no_shoe":
        labels = LOWER_BODY_LABELS - SHOE_LABELS
    elif region == "full_human_no_shoe":
        labels = set(range(1, 28)) - SHOE_LABELS - {25, 26, 27}
    else:
        raise ValueError(f"Unsupported region: {region}")
    return (np.isin(label_map, list(labels)) & valid).astype(bool)


def project_mesh_silhouette(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    k: np.ndarray,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    import cv2

    h, w = image_hw
    u, v, z = project_y_up_np(verts_cam, k)
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > 1e-4)
    pts2 = np.stack([u, v], axis=1)
    sil = np.zeros((h, w), dtype=np.uint8)
    drawn = 0
    skipped_z = 0
    skipped_far = 0
    margin = max(h, w)
    for tri in faces:
        if not finite[tri].all():
            skipped_z += 1
            continue
        poly = pts2[tri]
        if (
            poly[:, 0].max() < -margin
            or poly[:, 0].min() > w + margin
            or poly[:, 1].max() < -margin
            or poly[:, 1].min() > h + margin
        ):
            skipped_far += 1
            continue
        clipped = np.empty_like(poly)
        clipped[:, 0] = np.clip(poly[:, 0], -margin, w + margin)
        clipped[:, 1] = np.clip(poly[:, 1], -margin, h + margin)
        cv2.fillConvexPoly(sil, np.rint(clipped).astype(np.int32), 1)
        drawn += 1
    return sil.astype(bool), {
        "drawn_faces": int(drawn),
        "skipped_nonpositive_or_invalid_z_faces": int(skipped_z),
        "skipped_far_outside_faces": int(skipped_far),
    }


def bbox_from_mask(mask: np.ndarray) -> dict | None:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return None
    return {
        "x_min": int(xx.min()),
        "x_max": int(xx.max()),
        "y_min": int(yy.min()),
        "y_max": int(yy.max()),
        "center_x": float((xx.min() + xx.max()) * 0.5),
        "center_y": float((yy.min() + yy.max()) * 0.5),
        "width": int(xx.max() - xx.min() + 1),
        "height": int(yy.max() - yy.min() + 1),
    }


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = pred & target
    union = pred | target
    pred_count = int(pred.sum())
    target_count = int(target.sum())
    inter_count = int(inter.sum())
    union_count = int(union.sum())
    return {
        "iou": float(inter_count / union_count) if union_count else None,
        "precision": float(inter_count / pred_count) if pred_count else None,
        "recall": float(inter_count / target_count) if target_count else None,
        "pred_pixels": pred_count,
        "target_pixels": target_count,
        "intersection_pixels": inter_count,
        "union_pixels": union_count,
        "pred_bbox": bbox_from_mask(pred),
        "target_bbox": bbox_from_mask(target),
    }


def crop_mask_to_bbox(mask: np.ndarray, bbox: dict | None, pad: int, image_hw: tuple[int, int]) -> np.ndarray:
    if bbox is None:
        return np.zeros_like(mask, dtype=bool)
    h, w = image_hw
    x0 = max(0, int(bbox["x_min"]) - pad)
    x1 = min(w - 1, int(bbox["x_max"]) + pad)
    y0 = max(0, int(bbox["y_min"]) - pad)
    y1 = min(h - 1, int(bbox["y_max"]) + pad)
    crop = np.zeros_like(mask, dtype=bool)
    crop[y0 : y1 + 1, x0 : x1 + 1] = mask[y0 : y1 + 1, x0 : x1 + 1]
    return crop


def save_overlay(path: Path, rgb_path: Path | None, mesh_mask: np.ndarray, target_mask: np.ndarray) -> None:
    from PIL import Image

    h, w = target_mask.shape
    if rgb_path is not None and rgb_path.exists():
        base = Image.open(rgb_path).convert("RGB").resize((w, h))
        arr = np.asarray(base).copy()
    else:
        arr = np.zeros((h, w, 3), dtype=np.uint8) + 20
    out = arr.astype(np.float32)
    mesh_only = mesh_mask & ~target_mask
    target_only = target_mask & ~mesh_mask
    overlap = mesh_mask & target_mask
    out[target_only] = out[target_only] * 0.35 + np.array([40, 190, 255]) * 0.65
    out[mesh_only] = out[mesh_only] * 0.35 + np.array([255, 60, 210]) * 0.65
    out[overlap] = out[overlap] * 0.30 + np.array([255, 230, 40]) * 0.70
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path)


def aggregate_metric(view_reports: dict, key: str) -> dict:
    pred_total = 0
    target_total = 0
    inter_total = 0
    union_total = 0
    for report in view_reports.values():
        item = report[key]
        pred_total += item["pred_pixels"]
        target_total += item["target_pixels"]
        inter_total += item["intersection_pixels"]
        union_total += item["union_pixels"]
    return {
        "iou": float(inter_total / union_total) if union_total else None,
        "precision": float(inter_total / pred_total) if pred_total else None,
        "recall": float(inter_total / target_total) if target_total else None,
        "pred_pixels": int(pred_total),
        "target_pixels": int(target_total),
        "intersection_pixels": int(inter_total),
        "union_pixels": int(union_total),
    }


def metric_with_optional_crop(
    mesh_mask: np.ndarray,
    target_mask: np.ndarray,
    image_hw: tuple[int, int],
    crop_pad_px: int,
) -> tuple[dict, np.ndarray]:
    target_bbox = bbox_from_mask(target_mask)
    if crop_pad_px >= 0:
        pred = crop_mask_to_bbox(mesh_mask, target_bbox, crop_pad_px, image_hw)
    else:
        pred = mesh_mask
    return metrics(pred, target_mask), pred


def apply_view_correction(verts_cam: np.ndarray, view: str, corrections: dict) -> np.ndarray:
    return apply_camera_correction_np(verts_cam, view, corrections)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_params = np.load(Path(args.final_params), allow_pickle=True)
    view_names = [str(v) for v in final_params["view_names"]]
    view_to_index = {view_key(name): i for i, name in enumerate(view_names)}
    yaw_deg = YAW_PRESETS[args.yaw_preset]
    y_shifts = compute_y_alignment(
        Path(args.input_dir),
        final_params,
        view_to_index,
        yaw_deg,
        args.y_align,
        args.rotation_pivot,
    )
    view_corrections = load_view_corrections(args.view_corrections_json)
    verts_canonical, faces = load_ascii_ply_mesh(Path(args.final_single_mesh))
    report = {
        "schema_version": "ii17_silhouette_iou_audit_v1",
        "mesh": str(Path(args.final_single_mesh).resolve()),
        "yaw_preset": args.yaw_preset,
        "y_align": args.y_align,
        "view_corrections": view_corrections,
        "views": {},
        "aggregate": {},
        "overlay_legend": {
            "cyan": "target partseg mask only",
            "magenta": "SMPL projected silhouette only",
            "yellow": "overlap",
        },
        "target_regions": {
            "full_human_no_shoe": "all foreground human labels except shoes, teeth, tongue",
            "body_no_shoe": "legacy fitting/evaluation body mask; includes lower arms",
            "torso_legs_no_shoe": "torso, clothing, upper/lower legs; excludes head, hands, upper arms, lower arms, shoes",
            "lower_body_no_shoe": "upper/lower legs and lower clothing; excludes shoes",
        },
    }
    for view in VIEW_ORDER:
        idx = view_to_index[view]
        view_name = view_names[idx]
        k = final_params["K"][idx].astype(np.float64)
        depth_path = find_depth_path(Path(args.depth_dir), view_name)
        obs = load_depth_observation(depth_path, k.astype(np.float32), args.depth_y_axis)
        image_hw = obs["valid_mask"].shape
        regions = load_partseg(
            Path(args.partseg_dir),
            view_name,
            obs["valid_mask"],
            allow_png_label_fallback=args.allow_png_label_fallback,
        )
        verts_cam = canonical_to_view_camera(verts_canonical, view, final_params, view_to_index, yaw_deg, y_shifts)
        verts_cam = apply_view_correction(verts_cam, view, view_corrections)
        mesh_mask, render_diag = project_mesh_silhouette(verts_cam, faces, k, image_hw)
        rgb = find_rgb_path(Path(args.rgb_dir), view_name, f"{view_name}.png") if args.rgb_dir else None
        region_reports = {}
        debug_paths = {}
        for region in ("full_human_no_shoe", "body_no_shoe", "torso_legs_no_shoe", "lower_body_no_shoe"):
            target_mask = label_mask(regions["label_map"], obs["valid_mask"], region)
            crop_pad = args.lower_crop_pad_px if region == "lower_body_no_shoe" else args.body_crop_pad_px
            region_metrics, pred_mask = metric_with_optional_crop(mesh_mask, target_mask, image_hw, crop_pad)
            region_reports[region] = region_metrics
            debug_path = out_dir / "debug" / f"{view}_{region}_silhouette_iou.png"
            save_overlay(debug_path, rgb, pred_mask, target_mask)
            debug_paths[f"debug_{region}_overlay"] = str(debug_path.resolve())
        report["views"][view] = {
            "view_name": view_name,
            "depth_path": str(depth_path),
            "partseg_source": str(regions.get("partseg_source", "unknown")),
            "render": render_diag,
            **region_reports,
            **debug_paths,
        }
    for region in ("full_human_no_shoe", "body_no_shoe", "torso_legs_no_shoe", "lower_body_no_shoe"):
        report["aggregate"][region] = aggregate_metric(report["views"], region)
    with (out_dir / "silhouette_iou_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[done] wrote {out_dir}")
    for key in ("full_human_no_shoe", "body_no_shoe", "torso_legs_no_shoe", "lower_body_no_shoe"):
        agg = report["aggregate"][key]
        print(f"  {key}: iou={agg['iou']} precision={agg['precision']} recall={agg['recall']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="outputs/ii10_pr7_strict_full_final")
    p.add_argument("--final-params", default="outputs/ii10_pr7_strict_full_final/final_smpl_params.npz")
    p.add_argument("--final-single-mesh", required=True)
    p.add_argument("--output-dir", default="outputs/ii17_silhouette_iou")
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--partseg-dir", required=True)
    p.add_argument("--rgb-dir", default=None)
    p.add_argument("--depth-y-axis", choices=("up", "down"), default="down")
    p.add_argument("--allow-png-label-fallback", action="store_true")
    p.add_argument("--rotation-pivot", choices=("transl", "bbox_center", "origin"), default="transl")
    p.add_argument("--yaw-preset", choices=tuple(YAW_PRESETS.keys()), default="inverse_sides")
    p.add_argument("--y-align", choices=("none", "floor", "center"), default="floor")
    p.add_argument("--view-corrections-json", default=None)
    p.add_argument("--body-crop-pad-px", type=int, default=-1, help="Crop SMPL silhouette to target bbox for non-lower regions; -1 disables crop.")
    p.add_argument("--lower-crop-pad-px", type=int, default=80)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
