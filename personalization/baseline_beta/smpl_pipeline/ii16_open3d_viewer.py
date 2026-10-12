"""Interactive Open3D viewer for II-16 depth/SMPL overlays.

Canonical mode:
  1/2/3/4 toggle front/right/back/left depth point clouds
  F toggles final single SMPL

Per-view mode:
  1/2/3/4 switches front/right/back/left view
  D toggles that view's depth point cloud
  S toggles that view's SMPL mesh
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .depth_utils import load_depth_observation
from .ii8_init import find_depth_path
from .ii12_canonical_fusion import YAW_PRESETS, apply_y_alignment, load_obj, rot_y, view_key
from .ii13_single_refine import canonicalize_view_mesh
from .ii16_depth_mesh_overlays import (
    DEPTH_COLORS,
    FINAL_SMPL_COLOR,
    VIEW_ORDER,
    VIEW_SMPL_COLORS,
    body_depth_points,
    depth_cam_to_canonical,
    load_ascii_ply_mesh,
)


def o3d_color(color: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(color, dtype=np.float64) / 255.0


def make_pcd(points: np.ndarray, color: tuple[int, int, int]):
    import open3d as o3d

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.tile(o3d_color(color), (len(points), 1)))
    return pc


def make_mesh(verts: np.ndarray, faces: np.ndarray, color: tuple[int, int, int]):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.paint_uniform_color(o3d_color(color))
    mesh.compute_vertex_normals()
    return mesh


def compute_y_alignment(ii10_dir: Path, final_params, view_to_index: dict[str, int], yaw_deg, args) -> dict[str, float]:
    targets = {}
    for view in VIEW_ORDER:
        verts, _ = load_obj(ii10_dir / "meshes" / f"final_{view}.obj")
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


def load_scene_data(args: argparse.Namespace):
    ii10_dir = Path(args.ii10_dir)
    final_params = np.load(Path(args.final_params), allow_pickle=True)
    view_names = [str(v) for v in final_params["view_names"]]
    view_to_index = {view_key(name): i for i, name in enumerate(view_names)}
    yaw_deg = YAW_PRESETS[args.yaw_preset]
    y_shifts = compute_y_alignment(ii10_dir, final_params, view_to_index, yaw_deg, args)

    depth_cam = {}
    depth_can = {}
    view_meshes = {}
    for view in VIEW_ORDER:
        idx = view_to_index[view]
        view_name = view_names[idx]
        k = final_params["K"][idx].astype(np.float64)
        pts_cam, _ = body_depth_points(view, view_name, k, args)
        depth_cam[view] = pts_cam
        depth_can[view] = depth_cam_to_canonical(
            pts_cam, view, final_params, view_to_index, yaw_deg, y_shifts, args
        )
        view_meshes[view] = load_obj(ii10_dir / "meshes" / f"final_{view}.obj")

    final_mesh = load_ascii_ply_mesh(Path(args.final_single_ply))
    return depth_cam, depth_can, view_meshes, final_mesh


class ToggleViewer:
    def __init__(self, geometries: dict[str, object], initially_visible: set[str], title: str):
        self.geometries = geometries
        self.visible = {name: name in initially_visible for name in geometries}
        self.title = title
        self.vis = None

    def add_visible(self):
        for name, geom in self.geometries.items():
            if self.visible[name]:
                self.vis.add_geometry(geom, reset_bounding_box=False)

    def toggle(self, name: str):
        def cb(vis):
            if self.visible[name]:
                vis.remove_geometry(self.geometries[name], reset_bounding_box=False)
                self.visible[name] = False
            else:
                vis.add_geometry(self.geometries[name], reset_bounding_box=False)
                self.visible[name] = True
            print(f"[toggle] {name}: {'ON' if self.visible[name] else 'OFF'}")
            return False

        return cb

    def run(self, keymap: dict[int, str | callable]):
        import open3d as o3d

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=self.title, width=1400, height=950)
        self.add_visible()
        opt = self.vis.get_render_option()
        opt.point_size = 3.0
        opt.mesh_show_back_face = True
        opt.background_color = np.array([0.02, 0.02, 0.025])
        for key, target in keymap.items():
            if isinstance(target, str):
                self.vis.register_key_callback(key, self.toggle(target))
            else:
                self.vis.register_key_callback(key, target)
        print(self.title)
        print("Keys:", {chr(k): v if isinstance(v, str) else "callback" for k, v in keymap.items()})
        self.vis.run()
        self.vis.destroy_window()


def run_canonical(args: argparse.Namespace) -> None:
    depth_cam, depth_can, view_meshes, final_mesh = load_scene_data(args)
    geoms = {}
    for view in VIEW_ORDER:
        geoms[f"{view}_depth"] = make_pcd(depth_can[view], DEPTH_COLORS[view])
    final_verts, final_faces = final_mesh
    geoms["final_smpl"] = make_mesh(final_verts, final_faces, FINAL_SMPL_COLOR)
    initially = set(geoms)
    viewer = ToggleViewer(geoms, initially, "II-16 canonical: 4-view depth + final single SMPL")
    viewer.run(
        {
            ord("1"): "front_depth",
            ord("2"): "right_depth",
            ord("3"): "back_depth",
            ord("4"): "left_depth",
            ord("F"): "final_smpl",
        }
    )


def run_per_view(args: argparse.Namespace) -> None:
    import open3d as o3d

    depth_cam, depth_can, view_meshes, final_mesh = load_scene_data(args)
    state = {"view": "front", "depth_on": True, "smpl_on": True}
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="II-16 per-view: depth + view SMPL", width=1400, height=950)
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.mesh_show_back_face = True
    opt.background_color = np.array([0.02, 0.02, 0.025])
    active = {"depth": None, "smpl": None}

    def rebuild(reset=True):
        for geom in list(active.values()):
            if geom is not None:
                vis.remove_geometry(geom, reset_bounding_box=False)
        view = state["view"]
        active["depth"] = make_pcd(depth_cam[view], DEPTH_COLORS[view])
        verts, faces = view_meshes[view]
        active["smpl"] = make_mesh(verts, faces, VIEW_SMPL_COLORS[view])
        if state["depth_on"]:
            vis.add_geometry(active["depth"], reset_bounding_box=reset)
        if state["smpl_on"]:
            vis.add_geometry(active["smpl"], reset_bounding_box=reset)
        print(f"[view] {view} depth={'ON' if state['depth_on'] else 'OFF'} smpl={'ON' if state['smpl_on'] else 'OFF'}")

    def switch(view):
        def cb(_vis):
            state["view"] = view
            rebuild(reset=True)
            return False

        return cb

    def toggle_depth(_vis):
        state["depth_on"] = not state["depth_on"]
        rebuild(reset=False)
        return False

    def toggle_smpl(_vis):
        state["smpl_on"] = not state["smpl_on"]
        rebuild(reset=False)
        return False

    vis.register_key_callback(ord("1"), switch("front"))
    vis.register_key_callback(ord("2"), switch("right"))
    vis.register_key_callback(ord("3"), switch("back"))
    vis.register_key_callback(ord("4"), switch("left"))
    vis.register_key_callback(ord("D"), toggle_depth)
    vis.register_key_callback(ord("S"), toggle_smpl)
    print("II-16 per-view viewer")
    print("Keys: 1 front, 2 right, 3 back, 4 left, D toggle depth, S toggle SMPL")
    rebuild(reset=True)
    vis.run()
    vis.destroy_window()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("canonical", "per-view"), default="canonical")
    p.add_argument("--ii10-dir", default="outputs/ii10_pr7_strict_full_final")
    p.add_argument("--final-params", default="outputs/ii10_pr7_strict_full_final/final_smpl_params.npz")
    p.add_argument("--final-single-ply", default="outputs/ii13_single_refine_pr7_yawonly/final_single_smpl_canonical.ply")
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--partseg-dir", required=True)
    p.add_argument("--depth-y-axis", choices=("up", "down"), default="down")
    p.add_argument("--allow-png-label-fallback", action="store_true")
    p.add_argument("--depth-region", choices=("body_no_shoe", "foreground_no_shoe", "lower_body_no_shoe"), default="body_no_shoe")
    p.add_argument("--max-depth-points-per-view", type=int, default=90000)
    p.add_argument("--rotation-pivot", choices=("transl", "bbox_center", "origin"), default="transl")
    p.add_argument("--yaw-preset", choices=tuple(YAW_PRESETS.keys()), default="inverse_sides")
    p.add_argument("--y-align", choices=("none", "floor", "center"), default="floor")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "canonical":
        run_canonical(args)
    else:
        run_per_view(args)
