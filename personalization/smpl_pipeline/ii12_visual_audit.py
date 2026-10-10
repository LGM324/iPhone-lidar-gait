"""Render shaded diagnostic views for II-12 canonical SMPL overlays."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .ii12_canonical_fusion import VIEW_COLORS, VIEW_ORDER


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
            n = int(parts[0])
            if n != 3:
                continue
            faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def render_mesh_grid(input_dir: Path, output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 9), dpi=150)
    cameras = [
        ("diag", 12, -55),
        ("front", 0, -90),
        ("side", 0, 0),
        ("top", 90, -90),
    ]
    meshes = []
    all_pts = []
    for view in VIEW_ORDER:
        verts, faces = load_ascii_ply_mesh(input_dir / f"canonical_{view}.ply")
        # Display with Z-up in matplotlib while preserving SMPL's Y-up geometry.
        verts = verts[:, [0, 2, 1]]
        meshes.append((view, verts, faces))
        all_pts.append(verts)
    all_pts_np = np.concatenate(all_pts, axis=0)
    center = (all_pts_np.min(axis=0) + all_pts_np.max(axis=0)) * 0.5
    radius = float(np.max(all_pts_np.max(axis=0) - all_pts_np.min(axis=0)) * 0.55)

    for idx, (name, elev, azim) in enumerate(cameras, 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        for view, verts, faces in meshes:
            col = np.asarray(VIEW_COLORS[view], dtype=np.float64) / 255.0
            tri = verts[faces]
            poly = Poly3DCollection(tri, facecolor=col, edgecolor="none", alpha=0.55)
            poly.set_zsort("average")
            ax.add_collection3d(poly)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--title", default="canonical audit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_mesh_grid(Path(args.input_dir), Path(args.output), args.title)
