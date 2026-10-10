#!/usr/bin/env python3
"""II-8: metric-depth-aware SMPL initialization for STAND 4-view fitting.

This module converts raw HMR2 / 4D-Humans predictions plus Part-I metric depth
into ``init_params.npz`` for a later II-10 fitting stage. It does not optimize,
merge point clouds, canonicalize views, or modify depth inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "ii8_smpl_metric_init_v1"
COORDINATE_CONVENTION = "SMPL-camera: x-right,y-up,z-forward"
REQUIRED_RAW_KEYS = (
    "view_names",
    "betas_per_view",
    "body_pose_aa_per_view",
    "global_orient_aa_per_view",
)


def load_k_matrix(path: Path) -> np.ndarray:
    """Load a 3x3 intrinsic matrix from common JSON layouts."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return k_matrix_from_payload(payload)


def k_matrix_from_payload(payload: Any) -> np.ndarray:
    """Convert a single intrinsic payload into a 3x3 matrix."""
    if isinstance(payload, list):
        k = np.asarray(payload, dtype=np.float32)
    elif isinstance(payload, dict) and "K" in payload:
        k = np.asarray(payload["K"], dtype=np.float32)
    elif isinstance(payload, dict) and all(k in payload for k in ("fx", "fy", "cx", "cy")):
        k = np.array(
            [[payload["fx"], 0.0, payload["cx"]], [0.0, payload["fy"], payload["cy"]], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    elif isinstance(payload, dict) and "intrinsics" in payload:
        intrinsics = payload["intrinsics"]
        if isinstance(intrinsics, dict) and all(k in intrinsics for k in ("fx", "fy", "cx", "cy")):
            k = np.array(
                [
                    [intrinsics["fx"], 0.0, intrinsics["cx"]],
                    [0.0, intrinsics["fy"], intrinsics["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        else:
            k = np.asarray(intrinsics, dtype=np.float32)
    else:
        raise ValueError(
            "Unsupported K.json. Expected {'K': [[...]]}, {'fx','fy','cx','cy'}, "
            "a 3x3 list, or {'intrinsics': ...}."
        )

    if k.shape != (3, 3):
        raise ValueError(f"K must be [3,3], got {k.shape}.")
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError(f"Invalid K values:\n{k}")
    return k.astype(np.float32)


def load_k_by_view(path: Path, view_names: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray | dict[str, Any]]:
    """Load either one shared K or a view-name keyed intrinsics JSON."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    try:
        shared = k_matrix_from_payload(payload)
        return {name: shared for name in view_names}, shared
    except ValueError:
        pass

    if not isinstance(payload, dict):
        raise ValueError("K JSON must be a single intrinsics payload or a view-name keyed dict.")

    missing = [name for name in view_names if name not in payload]
    if missing:
        raise ValueError("View-keyed intrinsics JSON missing views: " + ", ".join(missing))
    by_view = {name: k_matrix_from_payload(payload[name]) for name in view_names}
    return by_view, {name: by_view[name].astype(float).tolist() for name in view_names}


def find_depth_path(depth_dir: Path, view_name: str) -> Path:
    """Strictly find a depth file matching the HMR2 view stem."""
    candidates = [
        depth_dir / f"{view_name}.npy",
        depth_dir / f"{view_name}_depth.npy",
        depth_dir / f"{view_name}_metric_depth.npy",
        depth_dir / f"{view_name}_final_metric_depth.npy",
    ]
    existing = [p for p in candidates if p.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError(f"More than one depth candidate for {view_name}: " + ", ".join(str(p) for p in existing))

    fuzzy = sorted(depth_dir.glob(f"{view_name}*.npy"))
    if len(fuzzy) == 1:
        return fuzzy[0]
    if len(fuzzy) > 1:
        raise RuntimeError(f"Ambiguous fuzzy depth match for {view_name}: " + ", ".join(p.name for p in fuzzy))
    raise FileNotFoundError(
        f"Depth not found for view '{view_name}' in {depth_dir}. "
        "Expected e.g. <view>.npy or <view>_metric_depth.npy."
    )


def find_rgb_path(rgb_dir: Path, view_name: str, frame_name: str) -> Path | None:
    """Return a matching RGB image for debug export, if one exists."""
    exact = rgb_dir / frame_name
    if exact.exists():
        return exact

    candidates: list[Path] = []
    stems = [view_name, Path(frame_name).stem]
    numeric_suffix = view_name.rsplit("_", 1)[-1]
    if numeric_suffix.isdigit():
        stems.append(numeric_suffix)
    frame_numeric_suffix = Path(frame_name).stem.rsplit("_", 1)[-1]
    if frame_numeric_suffix.isdigit():
        stems.append(frame_numeric_suffix)
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        for stem in stems:
            candidate = rgb_dir / f"{stem}{ext}"
            if candidate.exists() and candidate not in candidates:
                candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0]
    return None


def load_observed_xyz(depth_path: Path, k: np.ndarray, depth_y_axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Load depth/XYZ and return SMPL-compatible camera XYZ plus a valid mask."""
    if depth_y_axis not in {"up", "down"}:
        raise ValueError("--depth-y-axis must be 'up' or 'down'.")

    arr = np.load(depth_path)
    if arr.ndim == 2:
        z = arr.astype(np.float32)
        h, w = z.shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        x = (u - k[0, 2]) * z / k[0, 0]
        y = (v - k[1, 2]) * z / k[1, 1]
        xyz = np.stack([x, y, z], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        xyz = arr.astype(np.float32, copy=False)
    elif arr.ndim == 3 and arr.shape[0] == 3:
        xyz = np.moveaxis(arr.astype(np.float32, copy=False), 0, -1)
    else:
        raise ValueError(
            f"Unsupported depth shape {arr.shape} for {depth_path.name}. "
            "Expected [H,W], [H,W,3], or [3,H,W]."
        )

    valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 1e-4)
    if depth_y_axis == "down":
        xyz = xyz.copy()
        xyz[..., 1] *= -1.0
    return xyz, valid


def manifest_by_view(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in manifest.get("views", []):
        frame_name = record.get("frame_name")
        if frame_name:
            records[Path(frame_name).stem] = record
    return records


def robust_beta_mean(betas: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Robustly average [V,10] beta predictions using median-distance MAD filtering."""
    if betas.ndim != 2 or betas.shape[1] != 10:
        raise ValueError(f"Expected betas [V,10], got {betas.shape}.")
    if not np.isfinite(betas).all():
        raise ValueError("betas contain NaN/Inf.")

    median = np.median(betas, axis=0)
    distance = np.linalg.norm(betas - median[None], axis=1)
    med_distance = float(np.median(distance))
    mad = float(np.median(np.abs(distance - med_distance)))
    robust_sigma = max(1.4826 * mad, 1e-4)
    threshold = med_distance + 3.0 * robust_sigma
    inlier = distance <= threshold

    min_inliers = min(2, len(betas))
    if int(inlier.sum()) < min_inliers:
        closest = np.argsort(distance)[:min_inliers]
        inlier = np.zeros(len(betas), dtype=bool)
        inlier[closest] = True

    diagnostics = {
        "beta_distance_from_coordinatewise_median": distance.astype(float).tolist(),
        "beta_mad_threshold": float(threshold),
        "beta_inlier_count": int(inlier.sum()),
        "beta_inlier_mask": inlier.astype(bool).tolist(),
        "beta_per_dimension_std": betas.std(axis=0).astype(float).tolist(),
    }
    return betas[inlier].mean(axis=0).astype(np.float32), inlier, diagnostics


def clipped_roi_from_bbox(
    bbox_xyxy: np.ndarray,
    image_hw: tuple[int, int],
    x_fraction: tuple[float, float] = (0.30, 0.70),
    y_fraction: tuple[float, float] = (0.45, 0.72),
) -> tuple[int, int, int, int]:
    """Central lower-torso / pelvis proxy ROI clipped to image bounds."""
    h, w = image_hw
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    bw = max(x1 - x0, 2.0)
    bh = max(y1 - y0, 2.0)

    rx0 = int(np.floor(x0 + x_fraction[0] * bw))
    rx1 = int(np.ceil(x0 + x_fraction[1] * bw))
    ry0 = int(np.floor(y0 + y_fraction[0] * bh))
    ry1 = int(np.ceil(y0 + y_fraction[1] * bh))

    rx0 = int(np.clip(rx0, 0, w - 1))
    rx1 = int(np.clip(rx1, rx0 + 1, w))
    ry0 = int(np.clip(ry0, 0, h - 1))
    ry1 = int(np.clip(ry1, ry0 + 1, h))
    return rx0, ry0, rx1, ry1


def robust_observed_reference_point(
    xyz: np.ndarray,
    valid: np.ndarray,
    bbox_xyxy: np.ndarray,
    min_points: int = 120,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """Estimate a robust central torso/pelvis proxy point from metric observations."""
    h, w = valid.shape
    roi = clipped_roi_from_bbox(bbox_xyxy, (h, w))
    x0, y0, x1, y1 = roi
    roi_xyz = xyz[y0:y1, x0:x1][valid[y0:y1, x0:x1]]

    if len(roi_xyz) < min_points:
        roi = clipped_roi_from_bbox(bbox_xyxy, (h, w), x_fraction=(0.20, 0.80), y_fraction=(0.35, 0.78))
        x0, y0, x1, y1 = roi
        roi_xyz = xyz[y0:y1, x0:x1][valid[y0:y1, x0:x1]]

    if len(roi_xyz) < min_points:
        raise RuntimeError(
            "Too few valid metric-depth points in central torso/pelvis ROI "
            f"({len(roi_xyz)} < {min_points}). Check depth alignment, mask, and bbox."
        )
    return np.median(roi_xyz, axis=0).astype(np.float32), roi, int(len(roi_xyz))


def smpl_pelvis_at_zero_translation(
    smpl_model: Any,
    betas: np.ndarray,
    body_pose_aa: np.ndarray,
    global_orient_aa: np.ndarray,
    device: Any,
) -> np.ndarray:
    """Return standard SMPL joint 0 under shared beta, view pose/orient, and transl=0."""
    import torch

    with torch.inference_mode():
        out = smpl_model(
            betas=torch.from_numpy(betas[None]).to(device=device, dtype=torch.float32),
            body_pose=torch.from_numpy(body_pose_aa.reshape(1, -1)).to(device=device, dtype=torch.float32),
            global_orient=torch.from_numpy(global_orient_aa.reshape(1, 3)).to(device=device, dtype=torch.float32),
            return_verts=False,
        )
    return out.joints[0, 0].detach().cpu().numpy().astype(np.float32)


def project_smpl_y_up_to_image(points_xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Project x-right/y-up/z-forward camera points to pixels."""
    z = np.clip(points_xyz[:, 2], 1e-6, None)
    u = k[0, 0] * (points_xyz[:, 0] / z) + k[0, 2]
    v = k[1, 2] - k[1, 1] * (points_xyz[:, 1] / z)
    return np.stack([u, v], axis=-1)


def save_debug_image(
    rgb_path: Path,
    output_path: Path,
    bbox_xyxy: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    reference_xyz: np.ndarray,
    pelvis_xyz: np.ndarray,
    k: np.ndarray,
    label: str,
) -> None:
    """Save a 2D sanity overlay without claiming mesh alignment."""
    try:
        import cv2
    except ImportError:
        save_debug_image_pillow(rgb_path, output_path, bbox_xyxy, roi_xyxy, reference_xyz, pelvis_xyz, k, label)
        return

    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        return

    x0, y0, x1, y1 = [int(round(v)) for v in bbox_xyxy]
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 2)

    rx0, ry0, rx1, ry1 = roi_xyxy
    cv2.rectangle(image, (rx0, ry0), (rx1, ry1), (255, 255, 0), 2)

    projected = project_smpl_y_up_to_image(np.stack([reference_xyz, pelvis_xyz], axis=0), k)
    for point, color, name in zip(projected, [(0, 255, 0), (0, 0, 255)], ["depth-ref", "smpl-pelvis"]):
        px, py = int(round(point[0])), int(round(point[1]))
        cv2.circle(image, (px, py), 6, color, -1)
        cv2.putText(image, name, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    cv2.putText(image, label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def save_debug_image_pillow(
    rgb_path: Path,
    output_path: Path,
    bbox_xyxy: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    reference_xyz: np.ndarray,
    pelvis_xyz: np.ndarray,
    k: np.ndarray,
    label: str,
) -> None:
    """Pillow fallback for environments without OpenCV."""
    from PIL import Image, ImageDraw

    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = [int(round(v)) for v in bbox_xyxy]
    draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=3)

    rx0, ry0, rx1, ry1 = roi_xyxy
    draw.rectangle((rx0, ry0, rx1, ry1), outline=(0, 255, 255), width=3)

    projected = project_smpl_y_up_to_image(np.stack([reference_xyz, pelvis_xyz], axis=0), k)
    for point, color, name in zip(projected, [(0, 255, 0), (255, 0, 0)], ["depth-ref", "smpl-pelvis"]):
        px, py = int(round(point[0])), int(round(point[1]))
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)
        draw.text((px + 8, py - 8), name, fill=color)
    draw.text((18, 18), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def load_raw_hmr2_bundle(bundle_path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the strict STAND 4-view HMR2 raw bundle."""
    raw = np.load(bundle_path, allow_pickle=False)
    missing = [key for key in REQUIRED_RAW_KEYS if key not in raw.files]
    if missing:
        raise ValueError("Raw HMR2 bundle missing required keys: " + ", ".join(missing))

    view_names = [str(x) for x in raw["view_names"].tolist()]
    betas_per_view = raw["betas_per_view"].astype(np.float32)
    body_pose = raw["body_pose_aa_per_view"].astype(np.float32)
    global_orient = raw["global_orient_aa_per_view"].astype(np.float32)

    if len(view_names) != 4:
        raise ValueError(f"STAND first-pass expects exactly 4 views, got {len(view_names)}.")
    if betas_per_view.shape != (4, 10):
        raise ValueError(f"Expected betas [4,10], got {betas_per_view.shape}.")
    if body_pose.shape != (4, 23, 3):
        raise ValueError(f"Expected body_pose [4,23,3], got {body_pose.shape}.")
    if global_orient.shape != (4, 3):
        raise ValueError(f"Expected global_orient [4,3], got {global_orient.shape}.")
    for name, arr in (("betas", betas_per_view), ("body_pose", body_pose), ("global_orient", global_orient)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf.")
    return view_names, betas_per_view, body_pose, global_orient


def save_init_npz(
    output_npz: Path,
    *,
    view_names: list[str],
    k: np.ndarray,
    image_hw_per_view: np.ndarray,
    depth_y_axis: str,
    betas_shared_init: np.ndarray,
    betas_per_view_raw: np.ndarray,
    beta_inlier_mask: np.ndarray,
    body_pose_init: np.ndarray,
    global_orient_init: np.ndarray,
    transl_init: np.ndarray,
    observed_pelvis_proxy_cam: np.ndarray,
    smpl_pelvis_zero_translation: np.ndarray,
    depth_files: list[str],
) -> None:
    """Write the required II-8 init_params.npz contract."""
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        schema_version=np.array(SCHEMA_VERSION),
        pose_set=np.array("STAND"),
        view_names=np.asarray(view_names),
        K=k.astype(np.float32),
        image_hw_per_view=image_hw_per_view.astype(np.int32),
        coordinate_convention=np.array(COORDINATE_CONVENTION),
        source_depth_y_axis=np.array(depth_y_axis),
        betas_shared_init=betas_shared_init.astype(np.float32),
        betas_per_view_raw=betas_per_view_raw.astype(np.float32),
        beta_inlier_mask=beta_inlier_mask.astype(bool),
        body_pose_init=body_pose_init.astype(np.float32),
        global_orient_init=global_orient_init.astype(np.float32),
        transl_init=transl_init.astype(np.float32),
        observed_pelvis_proxy_cam=observed_pelvis_proxy_cam.astype(np.float32),
        smpl_pelvis_zero_translation=smpl_pelvis_zero_translation.astype(np.float32),
        depth_files=np.asarray(depth_files),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "II-8: build STAND 4-view shared beta and metric translation initialization. "
            "No iterative fitting, point-cloud fusion, or canonicalization is performed."
        )
    )
    parser.add_argument("--raw-hmr2-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--k-json", type=Path, required=True)
    parser.add_argument("--smpl-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    parser.add_argument(
        "--depth-y-axis",
        choices=("up", "down"),
        required=True,
        help="'down' flips observed Y into SMPL y-up coordinates; choose from Part-I depth convention.",
    )
    parser.add_argument("--rgb-dir", type=Path, default=None, help="Optional RGB directory for 2D debug overlays.")
    parser.add_argument("--device", default="cuda", help="Torch device. Use cpu if CUDA is unavailable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.raw_hmr2_dir, args.depth_dir, args.k_json, args.smpl_model_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    bundle_path = args.raw_hmr2_dir / "hmr2_stand_4view_raw.npz"
    manifest_path = args.raw_hmr2_dir / "hmr2_stand_4view_manifest.json"
    if not bundle_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "Expected hmr2_stand_4view_raw.npz and hmr2_stand_4view_manifest.json "
            f"inside {args.raw_hmr2_dir}"
        )

    view_names, betas_per_view, body_pose, global_orient = load_raw_hmr2_bundle(bundle_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    by_view = manifest_by_view(manifest)
    missing_manifest = [name for name in view_names if name not in by_view]
    if missing_manifest:
        raise ValueError("Manifest does not contain all bundle views: " + ", ".join(missing_manifest))

    lower_names = [name.lower() for name in view_names]
    missing_tokens = [token for token in ("front", "back", "left", "right") if not any(token in n for n in lower_names)]
    if missing_tokens:
        print("[warning] view_names do not clearly contain all STAND direction tokens: " + ", ".join(missing_tokens))

    k_by_view, k_output_metadata = load_k_by_view(args.k_json, view_names)
    betas_shared, beta_inlier, beta_diag = robust_beta_mean(betas_per_view)

    try:
        import torch
        import smplx as smpl_backend
    except ImportError as exc:
        raise SystemExit(
            "torch and the SMPL body-model backend are required for full II-8 execution. "
            "Use the previous SMPL fitting environment if available."
        ) from exc

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Rerun with --device cpu or install CUDA-enabled PyTorch.")
    smpl_model = smpl_backend.create(
        model_path=str(args.smpl_model_dir),
        model_type="smpl",
        gender=args.gender,
        num_betas=10,
        use_pca=False,
        create_transl=False,
        batch_size=1,
    ).to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output_dir / "debug"
    translations: list[np.ndarray] = []
    observed_refs: list[np.ndarray] = []
    pelvis_zeroes: list[np.ndarray] = []
    image_hws: list[np.ndarray] = []
    depth_files: list[str] = []
    per_view_diag: list[dict[str, Any]] = []

    for idx, view_name in enumerate(view_names):
        record = by_view[view_name]
        frame_name = str(record["frame_name"])
        expected_hw = tuple(int(x) for x in record["source_rgb_hw"])
        bbox = np.asarray(record["selected_bbox_xyxy"], dtype=np.float32)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError(f"{view_name}: selected_bbox_xyxy must be four finite numbers.")

        k_view = k_by_view[view_name]
        depth_path = find_depth_path(args.depth_dir, view_name)
        xyz, valid = load_observed_xyz(depth_path, k_view, args.depth_y_axis)
        if xyz.shape[:2] != expected_hw:
            raise ValueError(
                f"{view_name}: depth H/W {xyz.shape[:2]} != original RGB H/W {expected_hw}. "
                "Fix rotation, flip, crop/full-frame restoration, or matching filenames before II-10."
            )

        valid_count = int(valid.sum())
        valid_ratio = float(valid.mean())
        if valid_ratio < 0.005:
            raise RuntimeError(f"{view_name}: only {valid_ratio:.3%} valid metric-depth pixels.")

        reference, roi, roi_valid_count = robust_observed_reference_point(xyz, valid, bbox)
        pelvis_zero = smpl_pelvis_at_zero_translation(
            smpl_model,
            betas=betas_shared,
            body_pose_aa=body_pose[idx],
            global_orient_aa=global_orient[idx],
            device=device,
        )
        translation = (reference - pelvis_zero).astype(np.float32)
        pelvis_after = pelvis_zero + translation

        translations.append(translation)
        observed_refs.append(reference)
        pelvis_zeroes.append(pelvis_zero)
        image_hws.append(np.asarray(expected_hw, dtype=np.int32))
        depth_files.append(str(depth_path.resolve()))

        per_view_diag.append(
            {
                "view_name": view_name,
                "frame_name": frame_name,
                "depth_file": str(depth_path),
                "rgb_hw": list(expected_hw),
                "depth_shape": list(xyz.shape[:2]),
                "valid_depth_pixels": valid_count,
                "valid_depth_ratio": valid_ratio,
                "selected_bbox_xyxy": bbox.astype(float).tolist(),
                "pelvis_proxy_roi_xyxy": [int(x) for x in roi],
                "pelvis_proxy_valid_points": roi_valid_count,
                "observed_reference_xyz_smpl_camera": reference.astype(float).tolist(),
                "smpl_pelvis_zero_translation": pelvis_zero.astype(float).tolist(),
                "transl_init": translation.astype(float).tolist(),
                "pelvis_after_translation": pelvis_after.astype(float).tolist(),
                "pelvis_reference_residual_m": float(np.linalg.norm(pelvis_after - reference)),
            }
        )

        if args.rgb_dir is not None:
            rgb_path = find_rgb_path(args.rgb_dir, view_name, frame_name)
            if rgb_path is None:
                print(f"[warning] no RGB debug image for {view_name}; skipping overlay.")
            else:
                save_debug_image(
                    rgb_path=rgb_path,
                    output_path=debug_dir / f"{view_name}_ii8_debug.png",
                    bbox_xyxy=bbox,
                    roi_xyxy=roi,
                    reference_xyz=reference,
                    pelvis_xyz=pelvis_after,
                    k=k_view,
                    label=f"{view_name} | y={args.depth_y_axis}",
                )

    translations_np = np.stack(translations, axis=0).astype(np.float32)
    references_np = np.stack(observed_refs, axis=0).astype(np.float32)
    pelvis_zero_np = np.stack(pelvis_zeroes, axis=0).astype(np.float32)
    image_hws_np = np.stack(image_hws, axis=0).astype(np.int32)

    output_npz = args.output_dir / "init_params.npz"
    k_npz = np.stack([k_by_view[name] for name in view_names], axis=0).astype(np.float32)
    save_init_npz(
        output_npz,
        view_names=view_names,
        k=k_npz,
        image_hw_per_view=image_hws_np,
        depth_y_axis=args.depth_y_axis,
        betas_shared_init=betas_shared,
        betas_per_view_raw=betas_per_view,
        beta_inlier_mask=beta_inlier,
        body_pose_init=body_pose,
        global_orient_init=global_orient,
        transl_init=translations_np,
        observed_pelvis_proxy_cam=references_np,
        smpl_pelvis_zero_translation=pelvis_zero_np,
        depth_files=depth_files,
    )

    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "Coarse initializer only. II-10 Stage A must refine global_orient/transl against depth.",
        "coordinate_convention": {
            "smpl_camera": "x-right, y-up, z-forward",
            "source_depth_y_axis": args.depth_y_axis,
            "conversion_applied": "Y multiplied by -1 before matching SMPL" if args.depth_y_axis == "down" else "none",
        },
        "K": k_output_metadata,
        "beta_diagnostics": beta_diag,
        "per_view": per_view_diag,
        "required_manual_check": [
            "Open debug overlays and verify depth-ref (green) and smpl-pelvis (red) land in the torso/pelvis region.",
            "Verify depth H/W exactly matches RGB H/W for all four views.",
            "Use this only as initialization; do not judge final fit from this output.",
            "If overlays mirror vertically, rerun with the opposite --depth-y-axis after checking Part-I coordinates.",
        ],
    }
    diagnostics_path = args.output_dir / "init_diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print("\nII-8 complete")
    print(f"  init params : {output_npz}")
    print(f"  diagnostics : {diagnostics_path}")
    if args.rgb_dir is not None:
        print(f"  debug images: {debug_dir}")
    print(f"  beta inliers: {beta_inlier.astype(int).tolist()}")
    print("  translation norms (m):", np.round(np.linalg.norm(translations_np, axis=1), 3).tolist())
    print("\nNext: inspect debug overlays, then use init_params.npz in II-10 Stage A.")


if __name__ == "__main__":
    main()
