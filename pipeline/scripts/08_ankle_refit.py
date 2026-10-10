#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_ankle_refit.py  —  발목 전용 재fit

[논리] 06 loss = huber(cdist(pts, vv_전신).min()).mean() 는 부위 무관 최근접.
       발목 observability gate v2 실측:
         KNEE  Δcm  +0.113 (+5°)  vs  +0.343 (+10°)  — 뾰족 (대조군 ✓)
         ANKLE Δcm  +0.078 (+5°)  vs  +0.402 (+10°)  — ±5° 평평, 비대칭 4.6배

  ① ±5° 평평          → 데이터 미결정. 제약(flat-foot ∥ belt) 정당
  ② 비대칭 4.6배       → one-sided data→model 이 모델의 "허공 튀기" 를 안 벌함 → E_model 추가
  ③ 프레임별 최적 δ std 5.4° = 랜덤 → 시간 평활

[안전]
  · hip/knee 는 06 값 freeze → 무릎 168/168 회귀 위험 0
  · 06/07 파일 안 건드림. 새 stage. 새 npz.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import walkcfg  # [colab_kit] 경로/세션 설정
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(walkcfg.SESS)  # [colab_kit]
A0_NPZ   = Path(walkcfg.A0_NPZ)  # [colab_kit]
NEAR_NPZ = Path(walkcfg.NEAR_F3)   # λ=0 near fit  # [colab_kit]
OUT_DIR  = Path(walkcfg.ANKLE_DIR)  # [colab_kit]

SMPL_MODEL = walkcfg.SMPL_MODEL  # [colab_kit]
CAM_INTR   = walkcfg.CAM_INTR  # [colab_kit]
FROZEN_NPZ = walkcfg.FROZEN_NPZ  # [colab_kit]
DEPTH_SEQ  = walkcfg.DEPTH_SEQ  # [colab_kit]
FOURDH_SEQ = walkcfg.FOURDH_SEQ  # [colab_kit]
SEG_JSON   = walkcfg.SEG_JSON  # [colab_kit]

BP_RHIP, BP_RKNEE, BP_RANKLE = 1, 4, 7
J_RKNEE, J_RANKLE, J_RFOOT = 5, 8, 11
# ★ a0_align_v2.py:55 관례
SIDE_UPPER, SIDE_FAR, SIDE_NEAR = 0, 1, 2

# ---- 하이퍼 ----
HUBER_M      = 0.03
W_DATA       = 1.0
W_MODEL      = 1.0
W_FLAT       = 30.0
W_SMOOTH     = 5.0
W_PRIOR      = 0.0
ITERS, LR    = 400, 0.02

CONTACT_CM      = 3.0
CONTACT_MIN_PTS = 200
CONTACT_LEN_CM  = 12.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from a0_align_v2 import side_from_zorder, backproject  # noqa: E402

import smplx  # noqa: E402
_smpl = smplx.create(SMPL_MODEL, model_type="smpl", gender="neutral", num_betas=10).to(DEVICE)
_FACES_T = torch.tensor(_smpl.faces.astype(np.int64), device=DEVICE)
_FACES_NP = _smpl.faces.astype(np.int64)

with open(CAM_INTR) as _f:
    _K = json.load(_f)
    if "_default" in _K:
        _K = _K["_default"]


def smpl_forward_t(betas_np, go, tr, bp_23x3_t):
    """torch SMPL forward. bp_23x3_t 는 grad 흐르는 torch tensor. return verts(6890,3), joints(24,3) torch."""
    bt = torch.tensor(np.asarray(betas_np).reshape(1, 10), device=DEVICE, dtype=torch.float32)
    gt = torch.tensor(np.asarray(go).reshape(1, 3), device=DEVICE, dtype=torch.float32)
    trt = torch.tensor(np.asarray(tr).reshape(1, 3), device=DEVICE, dtype=torch.float32)
    bpt = bp_23x3_t.reshape(1, 69).to(torch.float32)
    out = _smpl(betas=bt, body_pose=bpt, global_orient=gt, transl=trt)
    return out.vertices[0], out.joints[0]


def load_depth_points(fid):
    fp = os.path.join(DEPTH_SEQ, f"WALK_side_{fid:05d}_person_metric.npy")
    depth = np.load(fp)
    pts, _ = backproject(depth, _K)
    return pts


def foot_vertex_ids():
    seg = json.loads(Path(SEG_JSON).read_text())
    ids = list(seg["rightFoot"]) + list(seg.get("rightToeBase", []))
    return np.unique(np.asarray(ids, np.int64))


def vertex_normals_np(verts, faces):
    n = np.zeros_like(verts)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-8, None)


def vertex_normals_t(verts, faces):
    """torch 버전 (미분 가능). verts (V,3), faces (F,3) long."""
    tri = verts[faces]                              # (F,3,3)
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    n = torch.zeros_like(verts)
    for k in range(3):
        n.index_add_(0, faces[:, k], fn)
    return n / (n.norm(dim=1, keepdim=True) + 1e-8)


def detect_contact(foot_pts, n_belt, d_belt):
    if len(foot_pts) < 50:
        return False, 0, 0.0
    h = foot_pts @ n_belt + d_belt
    low = foot_pts[np.abs(h) < CONTACT_CM / 100.0]
    if len(low) < CONTACT_MIN_PTS:
        return False, len(low), 0.0
    P = low - (low @ n_belt + d_belt)[:, None] * n_belt[None, :]
    P = P - P.mean(0)
    if len(P) < 3:
        return False, len(low), 0.0
    ev = np.linalg.eigh(P.T @ P)[1][:, -1]
    t = P @ ev
    L = float(np.percentile(t, 97.5) - np.percentile(t, 2.5)) * 100
    return L >= CONTACT_LEN_CM, len(low), L


def sole_vertex_ids(verts_rest, faces, foot_ids):
    """rest pose 에서 normal 이 '아래'(SMPL local -Y) 를 향하는 발 vertex = 밑창."""
    vn = vertex_normals_np(verts_rest, faces)
    m = np.zeros(len(verts_rest), bool); m[foot_ids] = True
    sole = np.where(m & (vn[:, 1] < -0.6))[0]
    if len(sole) < 5:
        sole = np.where(m & (vn[:, 1] > 0.6))[0]
    return sole


def sole_normal_t(verts, sole_ids):
    P = verts[sole_ids].to(torch.float64)
    P = P - P.mean(0, keepdim=True)
    C = P.T @ P
    w, V = torch.linalg.eigh(C)
    return V[:, 0]


def huber_t(a, delta=HUBER_M):
    return torch.where(a < delta, 0.5 * a ** 2, delta * (a - 0.5 * delta))


def chamfer_one_way(A, B):
    return torch.cdist(A, B).min(1).values


def estimate_belt(F):
    """belt = 수평 (n=[0,1,0]). Y_belt = 발 Y 의 95 percentile (아래끝)."""
    ys = []
    for f in F:
        if len(f["fp"]) >= 100:
            ys.append(np.percentile(f["fp"][:, 1], 95))
    Y = float(np.median(ys))
    n_belt = np.array([0.0, 1.0, 0.0])
    d_belt = -Y
    return n_belt, d_belt


def main(a):
    dev = DEVICE
    a0 = np.load(A0_NPZ)
    a0_fid = a0["frame"].astype(int)
    a0_go, a0_tr = a0["global_orient"], a0["transl"]

    fb = np.load(FROZEN_NPZ)
    betas = fb["betas"].astype(np.float32)[:10]

    fdh = np.load(FOURDH_SEQ, allow_pickle=False)
    fdh_fid = fdh["fids"].astype(int)
    fdh_bp = fdh["body_pose_aa"].astype(np.float64)
    fdh_pos = {int(f): i for i, f in enumerate(fdh_fid)}

    dn = np.load(NEAR_NPZ)
    fidn = dn["fid"].astype(int)
    th06 = dn["theta_fit"].astype(np.float64)
    o = np.argsort(fidn); fidn, th06 = fidn[o], th06[o]

    foot_ids = foot_vertex_ids()
    foot_ids_t = torch.tensor(foot_ids, device=dev, dtype=torch.long)

    # rest pose 로 밑창 vertex 정의
    with torch.no_grad():
        bp_rest = torch.zeros(23, 3, device=dev, dtype=torch.float32)
        v_rest, _ = smpl_forward_t(betas, np.zeros(3), np.zeros(3), bp_rest)
    sole_ids = sole_vertex_ids(v_rest.cpu().numpy(), _FACES_NP, foot_ids)
    sole_ids_t = torch.tensor(sole_ids, device=dev, dtype=torch.long)
    print(f"[sole] 밑창 vertex {len(sole_ids)} / 발 {len(foot_ids)}")

    # ---- 프레임별 준비 ----
    F = []
    for i, fid in enumerate(fidn):
        if int(fid) not in fdh_pos:
            continue
        k0_arr = np.where(a0_fid == fid)[0]
        if len(k0_arr) == 0:
            continue
        k0 = int(k0_arr[0])
        pts = load_depth_points(fid)
        near = pts[side_from_zorder(pts) == SIDE_NEAR]
        bp23 = fdh_bp[fdh_pos[int(fid)]].reshape(23, 3).copy()
        bp23[BP_RHIP], bp23[BP_RKNEE], bp23[BP_RANKLE] = th06[i, 0:3], th06[i, 3:6], th06[i, 6:9]
        with torch.no_grad():
            bpt = torch.tensor(bp23, device=dev, dtype=torch.float32)
            v0, J0 = smpl_forward_t(betas, a0_go[k0], a0_tr[k0], bpt)
        J0 = J0.cpu().numpy()
        ay = float(J0[J_RANKLE, 1])
        fp = near[near[:, 1] >= ay]                          # 발 depth
        F.append(dict(fid=int(fid), k0=k0, bp=bp23, fp=fp, i06=i))

    n_belt, d_belt = estimate_belt(F)
    n_belt_t = torch.tensor(n_belt, device=dev, dtype=torch.float64)
    print(f"[belt] n = {n_belt}  d = {d_belt:.4f}  (Y_belt = {-d_belt:.3f})")

    # 접촉 판정
    for f in F:
        con, nlow, L = detect_contact(f["fp"], n_belt, d_belt)
        f["contact"], f["nlow"], f["patch"] = con, nlow, L

    for i, f in enumerate(F):
        if i % 20 == 0:
            print(f"  ..{f['fid']}  발 {len(f['fp']):6d}점  접촉점 {f['nlow']:5d}  "
                  f"패치 {f['patch']:5.1f}cm  {'FLAT' if f['contact'] else '-'}")

    nC = sum(f["contact"] for f in F)
    print(f"\n[contact] flat-foot {nC}/{len(F)} ({100*nC/len(F):.0f}%)  "
          f"— 보행 주기의 ~40% 예상")

    # ---- 최적화 준비: 발목 3-DOF × 전 프레임 ----
    T = len(F)
    th_a_init = np.stack([f["bp"][BP_RANKLE] for f in F], 0)
    th_a = torch.tensor(th_a_init, dtype=torch.float32, device=dev, requires_grad=True)

    # 프레임별 constant tensor 미리 준비 (반복 오버헤드 제거)
    for f in F:
        f["bp_t"] = torch.tensor(f["bp"], dtype=torch.float32, device=dev)  # (23,3)
        f["pts_t"] = torch.tensor(f["fp"], dtype=torch.float32, device=dev) if len(f["fp"]) > 0 else None

    opt = torch.optim.Adam([th_a], lr=LR)
    print(f"[opt] iters={ITERS} lr={LR}  frames={T}")

    for it in range(ITERS):
        opt.zero_grad()
        L_d = torch.zeros((), device=dev, dtype=torch.float64)
        L_m = torch.zeros((), device=dev, dtype=torch.float64)
        L_f = torch.zeros((), device=dev, dtype=torch.float64)
        n_use = 0
        for i, f in enumerate(F):
            if f["pts_t"] is None or f["pts_t"].shape[0] < 100:
                continue
            # bp 조립: BP_RANKLE 만 th_a[i] 로 교체
            bp_use = torch.cat([f["bp_t"][:BP_RANKLE], th_a[i:i+1], f["bp_t"][BP_RANKLE+1:]], 0)
            v, _ = smpl_forward_t(betas, a0_go[f["k0"]], a0_tr[f["k0"]], bp_use)
            v64 = v.to(torch.float64)
            P = f["pts_t"].to(torch.float64)
            fv = v64[foot_ids_t]

            # ① 발 depth → 발 vertex
            L_d = L_d + huber_t(chamfer_one_way(P, fv)).mean()

            # ② 발 vertex → 발 depth (visible 만)
            vn = vertex_normals_t(v64, _FACES_T)
            vis_mask = vn[foot_ids_t, 2] < 0.0
            fv_vis = fv[vis_mask]
            if fv_vis.shape[0] > 3:
                L_m = L_m + huber_t(chamfer_one_way(fv_vis, P)).mean()

            # ③ flat-foot 이면 밑창 ∥ 벨트
            if f["contact"]:
                ns = sole_normal_t(v64, sole_ids_t)
                L_f = L_f + (1.0 - (ns @ n_belt_t) ** 2)

            n_use += 1

        # ④ 시간 평활
        L_s = ((th_a[1:] - th_a[:-1]) ** 2).sum().to(torch.float64)

        loss = W_DATA * L_d + W_MODEL * L_m + W_FLAT * L_f + W_SMOOTH * L_s
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == ITERS - 1:
            print(f"  it{it:4d}  data {float(L_d):.4f}  model {float(L_m):.4f}  "
                  f"flat {float(L_f):.4f}  smooth {float(L_s):.4f}  loss {float(loss):.4f}")

    th_new = th_a.detach().cpu().numpy().astype(np.float64)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    th_out = th06.copy()
    for i, f in enumerate(F):
        th_out[f["i06"], 6:9] = th_new[i]

    fid_out = np.array([f["fid"] for f in F], int)
    np.savez(OUT_DIR / "depthfit_ankle_a1.npz",
             fid=fid_out, theta_fit=th_out, theta_ankle_before=th06[:, 6:9].astype(np.float32),
             theta_ankle_after=th_new.astype(np.float32),
             contact=np.array([f["contact"] for f in F]),
             contact_patch_cm=np.array([f["patch"] for f in F]),
             foot_npts=np.array([len(f["fp"]) for f in F]),
             belt_n=n_belt, belt_d=d_belt, sole_ids=sole_ids,
             weights=np.array([W_DATA, W_MODEL, W_FLAT, W_SMOOTH, W_PRIOR]))

    from scipy.spatial.transform import Rotation as Rot
    moved = np.degrees((Rot.from_rotvec(th_new)
                        * Rot.from_rotvec(np.stack([th06[f["i06"], 6:9] for f in F]))
                        .inv()).magnitude())
    print("=" * 84)
    print(f"[결과] 발목 이동 (06 → 08): median {np.median(moved):.2f}°  "
          f"p95 {np.percentile(moved,95):.2f}°  max {moved.max():.2f}°")
    print(f"[saved] {OUT_DIR / 'depthfit_ankle_a1.npz'}")
    print()
    print("★ 다음:")
    print("  1) ankle_observability_gate_v2.py 를 이 npz 로 재실행 — basin 대칭·뾰족 여부")
    print("  2) ankle_angle_gate.py 로 발목 ROM 재계산 (e_flex 기준, 전 15.1°)")
    print("  3) 무릎 불변 확인 (hip/knee freeze 라 보장이지만)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=ITERS)
    args = p.parse_args()
    if args.iters != ITERS:
        ITERS = args.iters
    main(args)
