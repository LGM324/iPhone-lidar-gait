#!/usr/bin/env python3
"""
depth_viewer.py — 세션 person metric depth 시퀀스 Open3D viewer (전역 bbox 고정)
[colab_kit] 0420 B_viz/walk_depth_viewer.py 이식: 세션 경로(walkcfg), 한글 경로 대응, LiDAR 겹쳐보기(L).
로컬 PC 전용 (GUI). 실행 예:
  python depth_viewer.py --session 0729_full_3kmh
  python depth_viewer.py --session 0729_full_3kmh --before   # 재-anchor 전(.bak) 보기


컨트롤:
  → (Right) : 다음 프레임 (+1)
  ← (Left)  : 이전 프레임 (-1)
  ↑ (Up)    : +10 프레임 점프
  ↓ (Down)  : -10 프레임 점프
  Space     : 처음 프레임(idx 0) 리셋
  R         : 카메라 뷰 리셋 (전역 bbox 기준)
  L         : LiDAR conf==2 사람 점(빨강) 켜기/끄기
  S         : 좌우 다리 구분 색 (near/오른 빨강, far/왼 파랑, 상체 회색) 켜기/끄기
  M         : NB5 SMPL fit 메쉬(와이어프레임, 오른다리 빨강·왼다리 파랑) 켜기/끄기
  Q / ESC   : 종료

흔들림 방지 원리:
  1) 프리스캔으로 250f 전체 x/y/z min·max 계산 → 전역 axis-aligned bbox 확정
  2) 8-corner LineSet wireframe 을 초기 add (reset_bounding_box=True)
  3) pcd add 시 reset_bounding_box=False → 이후 update_geometry 는 카메라 리셋 안 함
  4) 각 프레임 pcd 에 전역 bbox 의 8 corner point 를 배경색으로 강제 삽입
     → pcd 자체 bbox 가 프레임 무관 상수 → open3d 내부 카메라 조정 완전 차단

입력: Walking_stand\\new_pipeline\\out\\walking_pipeline\\work\\out\\WALK_side_{fid}_person_metric.npy
intrinsics: Walking\\[Result]00_extract\\intrinsics.json (_default RGB 기준)
"""

import argparse
import json
import os
import sys

import numpy as np
import cv2
import open3d as o3d

os.environ.setdefault("WALK_ROOT", "G:/내 드라이브/walking_0729")   # 로컬 Drive for desktop

# GLFW key codes
KEY_RIGHT = 262
KEY_LEFT  = 263
KEY_DOWN  = 264
KEY_UP    = 265
KEY_SPACE = 32
KEY_R     = 82
KEY_L     = 76
KEY_S     = 83
KEY_M     = 77

BG_COLOR = np.array([0.05, 0.05, 0.05])   # 배경 어두운 회색 (invisible corner 색과 동일)


def load_intrinsics(path):
    K = json.load(open(path))
    if "_default" in K:
        K = K["_default"]
    return float(K["fx"]), float(K["fy"]), float(K["cx"]), float(K["cy"])


def backproject_person(depth, fx, fy, cx, cy, rgb_bgr=None, stride=1):
    """depth(H,W) valid>0 → (N,3) points + (N,3) colors.
    stride 는 시각화 성능용 sub-sampling. intrinsics 는 원본 (fx,fy,cx,cy) 그대로."""
    H, W = depth.shape[:2]
    if stride > 1:
        depth_s = depth[::stride, ::stride]
        Hs, Ws = depth_s.shape
        us, vs = np.meshgrid(np.arange(Ws), np.arange(Hs))
        us = us * stride
        vs = vs * stride
        z = depth_s.reshape(-1)
    else:
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        z = depth.reshape(-1)
    us = us.reshape(-1); vs = vs.reshape(-1)
    valid = (z > 0) & np.isfinite(z)
    z = z[valid]; us = us[valid]; vs = vs[valid]
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    if rgb_bgr is not None:
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        if stride > 1:
            rgb = rgb[::stride, ::stride]
        cols = rgb.reshape(-1, 3)[valid] / 255.0
    else:
        lo, hi = np.percentile(z, [2, 98])
        dn = np.clip((z - lo) / (hi - lo + 1e-8), 0, 1)
        cm = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cols = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
    return pts, cols.astype(np.float32)


def prescan_global_bbox(files, fx, fy, cx, cy, stride):
    """250f 전체 스캔하여 x/y/z 각 축 min·max. 프레임 무관 카메라 고정용."""
    xmin = ymin = zmin = np.inf
    xmax = ymax = zmax = -np.inf
    print(f"[prescan] {len(files)} frames 스캔 중 (전역 bbox 계산)...", flush=True)
    for i, fp in enumerate(files[::5]):          # 5프레임마다 (드라이브 스트리밍 부담 감소)
        depth = np.load(fp)
        if stride > 1:
            depth = depth[::stride, ::stride]
        H, W = depth.shape
        us, vs = np.meshgrid(np.arange(W) * stride, np.arange(H) * stride)
        valid = (depth > 0) & np.isfinite(depth)
        if not valid.any():
            continue
        z = depth[valid]
        u = us[valid]; v = vs[valid]
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        xmin = min(xmin, float(x.min())); xmax = max(xmax, float(x.max()))
        ymin = min(ymin, float(y.min())); ymax = max(ymax, float(y.max()))
        zmin = min(zmin, float(z.min())); zmax = max(zmax, float(z.max()))
        if (i + 1) % 50 == 0:
            print(f"  scanned {i+1}/{len(files)}", flush=True)
    # 약간의 padding (5%)
    def pad(lo, hi, r=0.05):
        m = 0.5 * (lo + hi); h = 0.5 * (hi - lo) * (1 + r)
        return m - h, m + h
    xlo, xhi = pad(xmin, xmax); ylo, yhi = pad(ymin, ymax); zlo, zhi = pad(zmin, zmax)
    print(f"[prescan] x [{xlo:+.3f}, {xhi:+.3f}]  "
          f"y [{ylo:+.3f}, {yhi:+.3f}]  z [{zlo:+.3f}, {zhi:+.3f}] (m)", flush=True)
    return (xlo, ylo, zlo), (xhi, yhi, zhi)


def make_bbox_lineset(lo, hi, color=(0.35, 0.35, 0.35)):
    """8-corner wireframe LineSet. 카메라 안정용 + 시각적 기준선."""
    xl, yl, zl = lo; xh, yh, zh = hi
    pts = np.array([
        [xl, yl, zl], [xh, yl, zl], [xh, yh, zl], [xl, yh, zl],
        [xl, yl, zh], [xh, yl, zh], [xh, yh, zh], [xl, yh, zh],
    ], dtype=np.float32)
    lines = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float64), (len(lines), 1)))
    return ls, pts


def imread_bgr(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def lidar_points(extract_dir, fid, depth, fx, fy, cx, cy):
    """LiDAR conf==2 & 사람 마스크 안 점 → (N,3). RGB intrinsics 를 LiDAR 해상도로 축소."""
    s = f"{fid:05d}"
    lid = np.load(os.path.join(extract_dir, "depth_npy", f"{s}.npy")).astype(np.float32)
    conf = np.load(os.path.join(extract_dir, "conf_npy", f"{s}.npy"))
    H, W = depth.shape; h, w = lid.shape; sx, sy = w / W, h / H
    pm = cv2.resize((depth > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    pm = cv2.erode(pm, np.ones((3, 3), np.uint8)).astype(bool)
    v, u = np.nonzero(pm & (conf >= 2) & (lid > 0))
    z = lid[v, u]
    return np.stack([(u - cx * sx) / (fx * sx) * z, (v - cy * sy) / (fy * sy) * z, z], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None, help="세션 이름 (예: 0729_full_3kmh)")
    ap.add_argument("--before", action="store_true", help="재-anchor 전(.bak) depth 보기")
    ap.add_argument("--depth-dir", default=None)
    ap.add_argument("--k-json",    default=None)
    ap.add_argument("--rgb-dir",   default=None)
    ap.add_argument("--no-rgb", action="store_true", help="depth colormap 만 (RGB 미사용)")
    ap.add_argument("--stride", type=int, default=2,
                    help="viewer 성능 위해 픽셀 sub-sampling (기본 2)")
    ap.add_argument("--no-wireframe", action="store_true",
                    help="bbox wireframe 숨기기 (여전히 pcd corner 는 유지)")
    args = ap.parse_args()
    if args.session:
        os.environ["WALK_SESSION"] = args.session
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import walkcfg
    args.depth_dir = args.depth_dir or walkcfg.DEPTH_SEQ
    args.k_json = args.k_json or walkcfg.CAM_INTR
    args.rgb_dir = args.rgb_dir or os.path.join(walkcfg.EXTRACT_DIR, "rgb")
    suffix = "_person_metric.npy.bak" if args.before else "_person_metric.npy"
    print(f"[session] {walkcfg.SESSION}  ({'재-anchor 전 .bak' if args.before else '최종 depth'})")

    fx, fy, cx, cy = load_intrinsics(args.k_json)
    print(f"[K] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    files = sorted([f for f in os.listdir(args.depth_dir) if f.endswith(suffix)])
    frames = [(f, int(f.split("_")[2])) for f in files]
    if not frames:
        raise SystemExit("depth 파일 없음")
    print(f"[frames] {len(frames)}  fid {frames[0][1]:05d}~{frames[-1][1]:05d}")

    file_paths = [os.path.join(args.depth_dir, f) for f, _ in frames]

    # 1) 프리스캔 → 전역 bbox
    lo, hi = prescan_global_bbox(file_paths, fx, fy, cx, cy, args.stride)
    lineset, corners = make_bbox_lineset(lo, hi)
    corners_cols = np.tile(BG_COLOR, (len(corners), 1))   # invisible corners for pcd anchor

    state = {"idx": 0, "lidar": False, "side": False, "smpl": False}
    pcd = o3d.geometry.PointCloud()
    wire = o3d.geometry.LineSet()
    smpl_ctx = {}

    def smpl_lines(fid):
        """NB5 결과(A0 root + near/far θ + 4DH 나머지 자세)로 SMPL 세워 와이어프레임 LineSet."""
        if not smpl_ctx:
            import torch, smplx, json as _j
            from knee_measurement_basis import assemble_body_pose
            ld = lambda p: np.load(p, allow_pickle=True) if os.path.exists(p) else None
            smpl_ctx.update(torch=torch, assemble=assemble_body_pose,
                            m=smplx.create(walkcfg.SMPL_MODEL, model_type="smpl", gender="neutral", num_betas=10),
                            betas=torch.tensor(np.load(walkcfg.FROZEN_NPZ)["betas"][:10].astype(np.float32)).reshape(1, 10),
                            a0=ld(walkcfg.A0_NPZ), near=ld(walkcfg.NEAR_F3), far=ld(walkcfg.FAR_F2), fdh=ld(walkcfg.FOURDH_SEQ))
            seg = _j.load(open(walkcfg.SEG_JSON))
            col = np.full((6890, 3), 0.85)
            for q in ("rightUpLeg", "rightLeg", "rightFoot", "rightToeBase"): col[seg.get(q, [])] = (0.9, 0.15, 0.15)
            for q in ("leftUpLeg", "leftLeg", "leftFoot", "leftToeBase"): col[seg.get(q, [])] = (0.2, 0.4, 0.95)
            f = smpl_ctx["m"].faces.astype(np.int64)
            edges = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
            smpl_ctx.update(edges=edges, ecol=col[edges[:, 0]])
        c, t = smpl_ctx, smpl_ctx["torch"]
        if c["a0"] is None or c["fdh"] is None:
            return None
        ra = {int(x): i for i, x in enumerate(c["a0"]["frame"])}
        rf = {int(x): i for i, x in enumerate(c["fdh"]["fids"])}
        if fid not in ra or fid not in rf:
            return None
        bp = np.asarray(c["fdh"]["body_pose_aa"]).reshape(len(c["fdh"]["fids"]), -1)[rf[fid]]
        if c["near"] is not None:
            rn = {int(x): i for i, x in enumerate(c["near"]["fid"])}
            if fid in rn:
                rr = {int(x): i for i, x in enumerate(c["far"]["fid"])} if c["far"] is not None else {}
                ft = c["far"]["far_theta_fit"][rr[fid]] if fid in rr else np.full(9, np.nan)
                ok = bool(c["far"]["is_fit"][rr[fid]]) if fid in rr else False
                bp = c["assemble"](bp, c["near"]["theta_fit"][rn[fid]], ft, ok)
        i = ra[fid]
        with t.no_grad():
            V = c["m"](betas=c["betas"], body_pose=t.tensor(np.asarray(bp, np.float32)).reshape(1, 69),
                       global_orient=t.tensor(c["a0"]["global_orient"][i], dtype=t.float32).reshape(1, 3),
                       transl=t.tensor(c["a0"]["transl"][i], dtype=t.float32).reshape(1, 3)).vertices[0].numpy()
        return V

    def load_and_apply(new_idx):
        new_idx = max(0, min(len(frames) - 1, new_idx))
        state["idx"] = new_idx
        fname, fid = frames[new_idx]
        depth = np.load(os.path.join(args.depth_dir, fname))
        rgb = None
        if not args.no_rgb:
            rgb_p = os.path.join(args.rgb_dir, f"{fid:05d}.jpg")
            if os.path.exists(rgb_p):
                rgb = imread_bgr(rgb_p)
        pts, cols = backproject_person(depth, fx, fy, cx, cy,
                                       rgb_bgr=rgb, stride=args.stride)
        if state["side"]:
            from a0_align_v2 import side_from_zorder
            sd = side_from_zorder(pts.astype(np.float32))
            cols = np.array([[0.6, 0.6, 0.6], [0.2, 0.4, 0.95], [0.9, 0.15, 0.15]], np.float32)[sd]
        if state["lidar"]:
            lp = lidar_points(walkcfg.EXTRACT_DIR, fid, depth, fx, fy, cx, cy)
            pts = np.concatenate([pts, lp])
            cols = np.concatenate([cols, np.tile([[1.0, 0.0, 0.0]], (len(lp), 1))]).astype(np.float32)
        # 전역 bbox 8 corner 를 pcd 에 강제 삽입 (배경색 → invisible)
        # → pcd 자체 bounding box 가 프레임마다 동일 → open3d 카메라 자동조정 완전 차단
        pts_all = np.concatenate([corners.astype(np.float32), pts], axis=0)
        cols_all = np.concatenate([corners_cols.astype(np.float32), cols], axis=0)
        pcd.points = o3d.utility.Vector3dVector(pts_all)
        pcd.colors = o3d.utility.Vector3dVector(cols_all)
        V = smpl_lines(fid) if state["smpl"] else None
        if V is not None:
            wire.points = o3d.utility.Vector3dVector(V)
            wire.lines = o3d.utility.Vector2iVector(smpl_ctx["edges"])
            wire.colors = o3d.utility.Vector3dVector(smpl_ctx["ecol"])
        else:
            wire.points = o3d.utility.Vector3dVector(corners[:2].astype(np.float64))
            wire.lines = o3d.utility.Vector2iVector(np.array([[0, 0]], np.int32))
            wire.colors = o3d.utility.Vector3dVector(np.array([BG_COLOR]))
        v = depth[depth > 0]
        print(f"  [idx {new_idx:3d}/{len(frames)-1}]  fid={fid:05d}  "
              f"pts={len(pts):6d}  median={float(np.median(v)):.3f}m", flush=True)

    # 초기 프레임 로드
    load_and_apply(0)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Walking depth viewer (bbox-locked)  ← → ↑ ↓ Space R Q",
                      width=1200, height=900)

    # ── 카메라 흔들림 방지 핵심 ──
    # LineSet 을 먼저 add (reset_bounding_box=True) → 초기 카메라가 전역 bbox 기준으로 세팅
    if not args.no_wireframe:
        vis.add_geometry(lineset, reset_bounding_box=True)
        # pcd 는 reset_bounding_box=False → 이후 카메라 자동리셋 금지
        vis.add_geometry(pcd, reset_bounding_box=False)
        vis.add_geometry(wire, reset_bounding_box=False)
    else:
        # wireframe 없이 pcd 만: pcd 안의 corner 로 bbox 확보
        vis.add_geometry(pcd, reset_bounding_box=True)
        vis.add_geometry(wire, reset_bounding_box=False)

    render = vis.get_render_option()
    render.point_size = 2.0
    render.background_color = BG_COLOR

    def cb_next(vis_):
        load_and_apply(state["idx"] + 1)
        vis_.update_geometry(pcd); vis_.update_geometry(wire)
        return False

    def cb_prev(vis_):
        load_and_apply(state["idx"] - 1)
        vis_.update_geometry(pcd); vis_.update_geometry(wire)
        return False

    def cb_next10(vis_):
        load_and_apply(state["idx"] + 10)
        vis_.update_geometry(pcd); vis_.update_geometry(wire)
        return False

    def cb_prev10(vis_):
        load_and_apply(state["idx"] - 10)
        vis_.update_geometry(pcd); vis_.update_geometry(wire)
        return False

    def cb_frame_reset(vis_):
        load_and_apply(0)
        vis_.update_geometry(pcd); vis_.update_geometry(wire)
        return False

    def make_toggle(key, name):
        def cb(vis_):
            state[key] = not state[key]
            print(f"  {name} {'ON' if state[key] else 'OFF'}")
            load_and_apply(state["idx"])
            vis_.update_geometry(pcd); vis_.update_geometry(wire)
            return False
        return cb

    def cb_lidar(vis_):
        state["lidar"] = not state["lidar"]
        print(f"  LiDAR overlay {'ON' if state['lidar'] else 'OFF'}")
        load_and_apply(state["idx"])
        vis_.update_geometry(pcd)
        return False

    def cb_view_reset(vis_):
        vis_.reset_view_point(True)
        return False

    vis.register_key_callback(KEY_RIGHT, cb_next)
    vis.register_key_callback(KEY_LEFT,  cb_prev)
    vis.register_key_callback(KEY_UP,    cb_next10)
    vis.register_key_callback(KEY_DOWN,  cb_prev10)
    vis.register_key_callback(KEY_SPACE, cb_frame_reset)
    vis.register_key_callback(KEY_R,     cb_view_reset)
    vis.register_key_callback(KEY_L,     cb_lidar)
    vis.register_key_callback(KEY_S,     make_toggle("side", "left/right split colors"))
    vis.register_key_callback(KEY_M,     make_toggle("smpl", "SMPL wireframe"))

    print("\n=== 컨트롤 ===")
    print("  → next   ← prev   ↑ +10   ↓ -10")
    print("  Space = frame 0 로 리셋   R = 카메라 뷰 리셋   L = LiDAR 빨강 점")
    print("  S = 좌우 다리 구분 색   M = SMPL fit 와이어프레임")
    print("  Q / ESC = 종료")
    print("→ 전역 bbox 고정으로 프레임 넘겨도 카메라 흔들림 없음\n", flush=True)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
