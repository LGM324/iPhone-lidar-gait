"""Sapiens v1 28-class region mapping for II-10."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SAPIENS_V1 = {
    0: "Background",
    1: "Apparel",
    2: "Face_Neck",
    3: "Hair",
    4: "Left_Foot",
    5: "Left_Hand",
    6: "Left_Lower_Arm",
    7: "Left_Lower_Leg",
    8: "Left_Shoe",
    9: "Left_Sock",
    10: "Left_Upper_Arm",
    11: "Left_Upper_Leg",
    12: "Lower_Clothing",
    13: "Right_Foot",
    14: "Right_Hand",
    15: "Right_Lower_Arm",
    16: "Right_Lower_Leg",
    17: "Right_Shoe",
    18: "Right_Sock",
    19: "Right_Upper_Arm",
    20: "Right_Upper_Leg",
    21: "Torso",
    22: "Upper_Clothing",
    23: "Lower_Lip",
    24: "Upper_Lip",
    25: "Lower_Teeth",
    26: "Upper_Teeth",
    27: "Tongue",
}

SKIN_LABELS = {7, 16, 11, 20, 6, 15}
CLOTH_LABELS = {1, 12, 21, 22}
SHOE_LABELS = {4, 8, 9, 13, 17, 18}
LEFT_SHOE_LABELS = {4, 8, 9}
RIGHT_SHOE_LABELS = {13, 17, 18}
IGNORE_LABELS = {0, 2, 3, 5, 14, 23, 24, 25, 26, 27}


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(bool)
    try:
        import cv2

        kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), np.uint8)
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except ImportError:
        out = mask.astype(bool).copy()
        for _ in range(pixels):
            padded = np.pad(out, 1, mode="constant", constant_values=False)
            neigh = [
                padded[1:-1, 1:-1],
                padded[:-2, 1:-1],
                padded[2:, 1:-1],
                padded[1:-1, :-2],
                padded[1:-1, 2:],
            ]
            out = np.logical_and.reduce(neigh)
        return out


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(bool)
    try:
        import cv2

        kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except ImportError:
        out = mask.astype(bool).copy()
        for _ in range(pixels):
            padded = np.pad(out, 1, mode="constant", constant_values=False)
            neigh = [
                padded[1:-1, 1:-1],
                padded[:-2, 1:-1],
                padded[2:, 1:-1],
                padded[1:-1, :-2],
                padded[1:-1, 2:],
            ]
            out = np.logical_or.reduce(neigh)
        return out


def build_region_maps(
    label_map: np.ndarray,
    fgmask: np.ndarray,
    depth_valid: np.ndarray,
    *,
    skin_erosion_px: int = 3,
    cloth_erosion_px: int = 2,
    shoe_erosion_px: int = 2,
    clothing_boundary_ignore_px: int = 5,
) -> dict[str, np.ndarray]:
    """Map Sapiens labels to mutually exclusive fitting regions."""
    label_map = np.asarray(label_map)
    usable = fgmask.astype(bool) & depth_valid.astype(bool)
    skin = np.isin(label_map, list(SKIN_LABELS)) & usable
    cloth = np.isin(label_map, list(CLOTH_LABELS)) & usable
    shoe = np.isin(label_map, list(SHOE_LABELS)) & usable
    ignore = (~usable) | np.isin(label_map, list(IGNORE_LABELS))

    boundary = dilate_mask(cloth, clothing_boundary_ignore_px) & dilate_mask(skin, clothing_boundary_ignore_px)
    skin = erode_mask(skin, skin_erosion_px) & ~boundary
    cloth = erode_mask(cloth, cloth_erosion_px) & ~boundary
    shoe = erode_mask(shoe, shoe_erosion_px)

    cloth = cloth & ~skin
    shoe = shoe & ~skin & ~cloth
    ignore = ~(skin | cloth | shoe)
    return {
        "skin_mask": skin.astype(bool),
        "cloth_mask": cloth.astype(bool),
        "shoe_mask": shoe.astype(bool),
        "ignore_mask": ignore.astype(bool),
    }


def label_histogram(label_map: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(label_map, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def load_partseg(
    partseg_dir: Path,
    view_name: str,
    depth_valid: np.ndarray,
    *,
    allow_png_label_fallback: bool = False,
) -> dict[str, np.ndarray]:
    label_path = partseg_dir / "labels" / f"{view_name}_partseg28_label.npy"
    mask_path = partseg_dir / "masks" / f"{view_name}_partseg28_fgmask.npy"
    if label_path.exists() and mask_path.exists():
        label_map = np.load(label_path)
        fgmask = np.load(mask_path).astype(bool)
        partseg_source = "npy"
    else:
        if not allow_png_label_fallback:
            raise FileNotFoundError(
                f"Missing Sapiens .npy partseg for {view_name}. Expected {label_path} and {mask_path}. "
                "If you intentionally want to use label-index PNGs, pass --allow-png-label-fallback."
            )
        png_path = partseg_dir / "mask" / f"{view_name}_seg_label.png"
        if not png_path.exists():
            png_path = partseg_dir / f"{view_name}_seg_label.png"
        if not png_path.exists():
            raise FileNotFoundError(
                f"Missing Sapiens partseg for {view_name}. Expected .npy files or label-index PNG "
                f"{partseg_dir / 'mask' / f'{view_name}_seg_label.png'}."
            )
        import cv2

        encoded = np.fromfile(str(png_path), dtype=np.uint8)
        label_map_raw = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if label_map_raw is None:
            raise ValueError(f"Could not read PNG label map: {png_path}")
        print(
            f"[partseg png fallback] {view_name}: shape={label_map_raw.shape} dtype={label_map_raw.dtype} "
            f"min={label_map_raw.min()} max={label_map_raw.max()} unique={np.unique(label_map_raw).tolist()[:64]}"
        )
        if label_map_raw.ndim != 2:
            raise ValueError(
                f"{view_name}: PNG fallback must be a single-channel label-index image, got shape {label_map_raw.shape}."
            )
        unique = np.unique(label_map_raw)
        if unique.min() < 0 or unique.max() > 27:
            raise ValueError(
                f"{view_name}: PNG fallback labels are not plausible Sapiens ids 0..27: {unique.tolist()}"
            )
        label_map = label_map_raw.astype(np.int32)
        fgmask = label_map > 0
        partseg_source = "png_fallback"
    if label_map.shape != depth_valid.shape or fgmask.shape != depth_valid.shape:
        raise ValueError(
            f"{view_name}: partseg shape mismatch. label={label_map.shape}, mask={fgmask.shape}, depth={depth_valid.shape}."
        )
    regions = build_region_maps(label_map, fgmask, depth_valid)
    regions.update(
        {
            "label_map": label_map,
            "fgmask": fgmask,
            "partseg_source": partseg_source,
            "label_histogram": label_histogram(label_map),
            "left_shoe_mask": np.isin(label_map, list(LEFT_SHOE_LABELS)) & fgmask & depth_valid,
            "right_shoe_mask": np.isin(label_map, list(RIGHT_SHOE_LABELS)) & fgmask & depth_valid,
        }
    )
    return regions


def load_metadata(partseg_dir: Path) -> dict:
    path = partseg_dir / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_region_debug(path: Path, regions: dict[str, np.ndarray]) -> None:
    from PIL import Image

    h, w = regions["ignore_mask"].shape
    image = np.zeros((h, w, 4), dtype=np.uint8)
    image[regions["skin_mask"]] = [0, 255, 0, 180]
    image[regions["cloth_mask"]] = [0, 80, 255, 180]
    image[regions["shoe_mask"]] = [255, 150, 0, 220]
    image[regions["ignore_mask"]] = [255, 0, 0, 35]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGBA").save(path)


def shoe_points_from_xyz(xyz: np.ndarray, shoe_mask: np.ndarray, max_points: int = 5000) -> np.ndarray:
    pts = xyz[shoe_mask & np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 1e-4)]
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
        pts = pts[idx]
    return pts.astype(np.float32)


def save_binary_mask_png(path: Path, mask: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)
