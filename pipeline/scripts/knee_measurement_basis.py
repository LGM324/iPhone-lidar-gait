#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
knee_measurement_basis.py
=========================
각 프레임 무릎각(near/far)에 "그 각을 만든 표면 측정 근거"를 태그한다.
- 프레임별 thigh / shank 각각의 surface residual(중앙값 cm) + 설명한 점 수(npts)
- fit 이 이미 낸 pose 를 그대로 forward → residual 만 "읽음" (재fit/offset/재정의 금지)
- near = 측정 근거 태그, far = 측정/추정(confidence) 라벨 (n_far ≥ 300 = MEASURED)

배치: _pipeline_consolidated/D_walking_analysis/[Script]D_experiments/
산출: _pipeline_consolidated/D_walking_analysis/[Result]D_experiments/knee_basis_{cycle,full}.{npz,csv,png}

불변:
- depth backproject / side 라벨 = Stage05/06 (a0_align_v2) 그대로 재사용
- SMPL forward = β(T-pose freeze) + root(A0) + near θ(f3) + far θ(far2)
- residual = 관측점 → 최근접 visible(vertex.normal_z<0) 동일-leg SMPL vertex, unidirectional, raw median cm
- 무릎각 = 기존 npz(knee_fit / far_knee_fit) 그대로 (재정의 X)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import walkcfg  # [colab_kit] 경로/세션 설정
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# a0_align_v2 의 backproject, side_from_zorder, SIDE_* 재사용 (재유도 금지)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from a0_align_v2 import backproject, side_from_zorder, SIDE_UPPER, SIDE_FAR, SIDE_NEAR


# =============================================================
# 0. 경로 (canonical: Walking_stand\new_pipeline\out\)
# =============================================================
BASE       = Path(walkcfg.SESS)  # [colab_kit]
DEPTH_DIR  = Path(walkcfg.DEPTH_SEQ)  # [colab_kit]
A0_NPZ     = Path(walkcfg.A0_NPZ)  # [colab_kit]
NEAR_NPZ   = Path(walkcfg.NEAR_F3)   # λ=0 near fit (f3)  # [colab_kit]
FAR_NPZ    = Path(walkcfg.FAR_F2)    # λ=0 far fit (far2)  # [colab_kit]
BETA_NPZ   = walkcfg.FROZEN_NPZ  # [colab_kit]
BP_NPZ     = walkcfg.FOURDH_SEQ  # [colab_kit]
SMPL_MODEL = walkcfg.SMPL_MODEL  # [colab_kit]
CAM_INTR   = walkcfg.CAM_INTR  # [colab_kit]
PART_SEG   = walkcfg.SEG_JSON  # [colab_kit]

OUT_DIR    = Path(walkcfg.KNEE_DIR)  # [colab_kit]


# =============================================================
# 1. 상수 (fit 과 동일)
# =============================================================
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
VIS_NZ         = 0.0     # visible = vertex normal.z < 0  (Stage05/06 와 동일)
NOISE_FLOOR_CM = 1.43    # prior_ablation lam0 near residual median (그림 참조선)
MIN_FAR_PTS    = 300     # far MEASURED/INFERRED 경계 (Stage07 동일)

# body_pose(23,3) index (SMPL joint - 1). Stage06/07 상수와 일치.
BP_R_HIP, BP_R_KNEE, BP_R_ANKLE = 1, 4, 7   # near = 오른다리
BP_L_HIP, BP_L_KNEE, BP_L_ANKLE = 0, 3, 6   # far  = 왼다리


# =============================================================
# 2. HOOK 구현
# =============================================================
def make_smpl():
    """SMPL 로드 (Stage05/06/07 과 동일: neutral, num_betas=10)."""
    import smplx
    smpl = walkcfg.make_smpl(SMPL_MODEL).to(DEVICE)
    faces = smpl.faces.astype(np.int64)
    return smpl, faces


def lift_pointcloud(fid, K):
    """metric depth npy → pts (N,3). Stage06 backproject 그대로."""
    fp = DEPTH_DIR / f"WALK_side_{fid:05d}_person_metric.npy"
    if not fp.exists():
        return None
    depth = np.load(fp)
    r = backproject(depth, K)
    pts = r[0] if isinstance(r, tuple) else r
    return pts.astype(np.float32)


def side_masks(pts):
    """A0/A1 side 라벨. near=오른다리, far=왼다리."""
    side = side_from_zorder(pts)
    return (side == SIDE_NEAR), (side == SIDE_FAR)


def load_part_masks(V=6890):
    """SMPL vertex 마스크: rightUpLeg=R_thigh, rightLeg=R_shank, leftUpLeg=L_thigh, leftLeg=L_shank."""
    if not os.path.exists(PART_SEG):
        raise FileNotFoundError(f"part segmentation JSON 없음: {PART_SEG}")
    with open(PART_SEG) as f:
        seg = json.load(f)

    def m(name):
        a = np.zeros(V, bool)
        a[seg[name]] = True
        return a

    return {
        "R_thigh": m("rightUpLeg"), "R_shank": m("rightLeg"),
        "L_thigh": m("leftUpLeg"),  "L_shank": m("leftLeg"),
    }


# =============================================================
# 3. 유틸
# =============================================================
def vertex_normals_np(verts, faces):
    nrm = np.zeros_like(verts)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(nrm, faces[:, k], fn)
    n = np.linalg.norm(nrm, axis=1, keepdims=True)
    return nrm / np.clip(n, 1e-8, None)


def assemble_body_pose(base_bp_frame, near_theta9, far_theta9, far_is_fit):
    """
    base 4DH body_pose(23,3) 에 near/far 다리 3관절씩 덮어씀.
    theta 순서 = [hip(3), knee(3), ankle(3)]  (Stage06/07 저장 순서 확인됨).
    far_is_fit=False (INFERRED) 면 far θ 는 안 덮어씀 (4DH 그대로).
    """
    bp = base_bp_frame.copy().reshape(23, 3).astype(np.float32)
    nt = np.asarray(near_theta9).reshape(3, 3)
    bp[BP_R_HIP], bp[BP_R_KNEE], bp[BP_R_ANKLE] = nt[0], nt[1], nt[2]
    if far_is_fit and not np.any(np.isnan(far_theta9)):
        ft = np.asarray(far_theta9).reshape(3, 3)
        bp[BP_L_HIP], bp[BP_L_KNEE], bp[BP_L_ANKLE] = ft[0], ft[1], ft[2]
    return bp.reshape(-1)


def smpl_forward(smpl, betas_t, go_t, tr_t, bp_t):
    """SMPL forward → verts (V,3) numpy."""
    with torch.no_grad():
        out = smpl(betas=betas_t, body_pose=bp_t.reshape(1, 69),
                   global_orient=go_t.reshape(1, 3), transl=tr_t.reshape(1, 3))
    return out.vertices[0].detach().cpu().numpy()


def part_residual(obs_pts, verts, vnrm, thigh_mask, shank_mask):
    """
    관측점 → 최근접 visible 동일-leg vertex (unidirectional, point-to-vertex).
    각 점을 최근접 vertex 의 part 로 배정 → part 별 (중앙값 cm, npts).
    """
    leg_mask = (thigh_mask | shank_mask) & (vnrm[:, 2] < VIS_NZ)
    idx_leg = np.where(leg_mask)[0]
    if obs_pts.shape[0] == 0 or idx_leg.size == 0:
        return dict(thigh_cm=np.nan, thigh_n=0, shank_cm=np.nan, shank_n=0)
    tree = cKDTree(verts[idx_leg])
    dist, j = tree.query(obs_pts, k=1)
    nearest_v = idx_leg[j]
    is_thigh = thigh_mask[nearest_v]
    dist_cm = dist * 100.0
    dt = dist_cm[is_thigh]
    ds = dist_cm[~is_thigh]
    return dict(
        thigh_cm=float(np.median(dt)) if dt.size else np.nan,
        thigh_n=int(dt.size),
        shank_cm=float(np.median(ds)) if ds.size else np.nan,
        shank_n=int(ds.size),
    )


# =============================================================
# 4. 메인
# =============================================================
def main(cycle=False):
    from time import time as _t

    print("[load] intrinsics, A0, near, far, β, 4DH, SMPL, part masks ...")
    K = json.load(open(CAM_INTR))
    if "_default" in K:
        K = K["_default"]

    a0   = np.load(A0_NPZ)
    near = np.load(NEAR_NPZ)
    far  = np.load(FAR_NPZ, allow_pickle=True)
    betas_np = np.load(BETA_NPZ)["betas"][:10].astype(np.float32)
    fdh = np.load(BP_NPZ, allow_pickle=True)
    fdh_bp = np.asarray(fdh["body_pose_aa"]).astype(np.float32)
    if fdh_bp.ndim == 3:
        fdh_bp = fdh_bp.reshape(fdh_bp.shape[0], -1)
    fdh_row = {int(f): i for i, f in enumerate(np.asarray(fdh["fids"]).astype(int))}

    smpl, faces = make_smpl()
    pmask = load_part_masks(V=6890)
    betas_t = torch.tensor(betas_np.reshape(1, 10), device=DEVICE)

    # fid 필터
    fids_all = a0["frame"].astype(int)
    if cycle:
        fids = fids_all[(fids_all >= 1018) & (fids_all <= 1059)]
    else:
        fids = fids_all

    def rowmap(npz, key):
        return {int(f): i for i, f in enumerate(npz[key].astype(int))}
    r_a0 = rowmap(a0, "frame")
    r_nr = rowmap(near, "fid")
    r_fr = rowmap(far, "fid")

    print(f"[run] n_frames = {len(fids)}  (fid {int(fids[0])}..{int(fids[-1])})")

    rows = []
    t0 = _t()
    for k_i, fid in enumerate(fids):
        fid_int = int(fid)
        if fid_int not in r_nr or fid_int not in r_fr or fid_int not in r_a0:
            continue
        if fid_int not in fdh_row:
            continue

        # ---- pose 재구성 ----
        go = a0["global_orient"][r_a0[fid_int]].astype(np.float32)
        tr = a0["transl"][r_a0[fid_int]].astype(np.float32)
        n_theta = near["theta_fit"][r_nr[fid_int]]
        f_theta = far["far_theta_fit"][r_fr[fid_int]]
        far_is_fit = bool(far["is_fit"][r_fr[fid_int]])
        bp = assemble_body_pose(fdh_bp[fdh_row[fid_int]], n_theta, f_theta, far_is_fit)

        go_t = torch.tensor(go, device=DEVICE)
        tr_t = torch.tensor(tr, device=DEVICE)
        bp_t = torch.tensor(bp, device=DEVICE)
        verts = smpl_forward(smpl, betas_t, go_t, tr_t, bp_t)
        vnrm = vertex_normals_np(verts, faces)

        # ---- 관측점 + side ----
        pts = lift_pointcloud(fid_int, K)
        if pts is None:
            continue
        near_m, far_m = side_masks(pts)

        # ---- 부위별 residual ----
        rn = part_residual(pts[near_m], verts, vnrm, pmask["R_thigh"], pmask["R_shank"])
        rf = part_residual(pts[far_m],  verts, vnrm, pmask["L_thigh"], pmask["L_shank"])
        n_far_total = rf["thigh_n"] + rf["shank_n"]

        rows.append(dict(
            fid=fid_int,
            knee_near=float(near["knee_fit"][r_nr[fid_int]]),
            thigh_near_resid_cm=rn["thigh_cm"], thigh_near_npts=rn["thigh_n"],
            shank_near_resid_cm=rn["shank_cm"], shank_near_npts=rn["shank_n"],
            knee_far=float(far["far_knee_fit"][r_fr[fid_int]]),
            thigh_far_resid_cm=rf["thigh_cm"],
            # far_spike_refit.py 가 reject 한 프레임(재피팅해도 튐) → 관측 점 0 으로 = 가림과 같이 보간
            thigh_far_npts=(0 if ("reject" in far.files and bool(far["reject"][r_fr[fid_int]])) else rf["thigh_n"]),
            shank_far_resid_cm=rf["shank_cm"], shank_far_npts=rf["shank_n"],
            far_meas=bool(n_far_total >= MIN_FAR_PTS),
        ))

        if (k_i + 1) % 20 == 0:
            print(f"  processed {k_i+1}/{len(fids)}  ({_t()-t0:.1f}s)")

    if not rows:
        print("[ERROR] no rows processed"); return

    # ---- 자기검증: npts 합 vs 기존 n_{near,far}_leg ----
    f0 = rows[0]["fid"]
    got_n = rows[0]["thigh_near_npts"] + rows[0]["shank_near_npts"]
    exp_n = int(near["n_near_leg"][r_nr[f0]]) if "n_near_leg" in near.files else -1
    got_f = rows[0]["thigh_far_npts"] + rows[0]["shank_far_npts"]
    exp_f = int(far["n_far_leg"][r_fr[f0]]) if "n_far_leg" in far.files else -1
    print(f"\n[self-check] fid {f0}:")
    print(f"  near part npts 합 = {got_n:>5d}   vs   n_near_leg = {exp_n:>5d}  (Δ={got_n-exp_n:+d})")
    print(f"  far  part npts 합 = {got_f:>5d}   vs   n_far_leg  = {exp_f:>5d}  (Δ={got_f-exp_f:+d})")
    print("  (thigh+shank 만 취급하고 foot Y-band 은 near/far leg 원본 마스크에 포함될 수 있어 Δ 는 소량 예상)")

    # ---- 저장 ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "cycle" if cycle else "full"

    keys = list(rows[0].keys())
    save_dict = {k: np.array([r[k] for r in rows]) for k in keys}
    npz_out = OUT_DIR / f"knee_basis_{tag}.npz"
    np.savez(str(npz_out), **save_dict)

    csv_out = OUT_DIR / f"knee_basis_{tag}.csv"
    with open(csv_out, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    png_out = OUT_DIR / f"knee_basis_{tag}.png"
    plot(rows, png_out, tag)

    print(f"\n[saved] {npz_out}")
    print(f"[saved] {csv_out}")
    print(f"[saved] {png_out}")
    print(f"[elapsed] {_t()-t0:.1f}s")


# =============================================================
# 5. 그림 (prior_ablation 스타일: near 실선 / far 점선)
# =============================================================
def plot(rows, path, tag):
    fid = np.array([r["fid"] for r in rows])

    def col(k):
        return np.array([r[k] for r in rows], float)

    fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # (1) 무릎각 + 측정근거 태그 (마커 크기 ∝ npts, 진하기 ∝ residual)
    for leg, ls, base_c in [("near", "-", "tab:blue"), ("far", "--", "tab:red")]:
        knee = col(f"knee_{leg}")
        npts = col(f"thigh_{leg}_npts") + col(f"shank_{leg}_npts")
        rcm  = np.nanmean(np.stack([col(f"thigh_{leg}_resid_cm"),
                                     col(f"shank_{leg}_resid_cm")]), axis=0)
        ax[0].plot(fid, knee, ls, color=base_c, lw=1, alpha=0.5, label=f"{leg} knee")
        s = 20 + 180 * (npts / (np.nanmax(npts) + 1e-9))
        c = np.clip(rcm / 3.0, 0, 1)
        ax[0].scatter(fid, knee, s=s, c=c, cmap="viridis_r",
                      edgecolor=base_c, linewidth=0.6, zorder=3)
    ax[0].set_ylabel("knee [deg]")
    ax[0].set_title(f"knee angle tagged by measurement basis "
                    f"(marker size ∝ npts, darkness ∝ residual)  [{tag}]")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.3)

    # (2) residual cm
    for leg, ls in [("near", "-"), ("far", "--")]:
        ax[1].plot(fid, col(f"thigh_{leg}_resid_cm"), ls, color="tab:green",
                   label=f"{leg} thigh")
        ax[1].plot(fid, col(f"shank_{leg}_resid_cm"), ls, color="tab:orange",
                   label=f"{leg} shank")
    ax[1].axhline(NOISE_FLOOR_CM, color="k", ls=":", lw=1,
                  label=f"depth noise floor ~{NOISE_FLOOR_CM}cm")
    ax[1].set_ylabel("surface residual [cm]")
    ax[1].legend(loc="upper right", ncol=2, fontsize=8)
    ax[1].grid(alpha=0.3)

    # (3) npts + far INFERRED 배경
    for leg, ls in [("near", "-"), ("far", "--")]:
        ax[2].plot(fid, col(f"thigh_{leg}_npts"), ls, color="tab:green",
                   label=f"{leg} thigh n")
        ax[2].plot(fid, col(f"shank_{leg}_npts"), ls, color="tab:orange",
                   label=f"{leg} shank n")
    inf = ~col("far_meas").astype(bool)
    for x in fid[inf]:
        ax[2].axvspan(x - 0.5, x + 0.5, color="red", alpha=0.06)
    ax[2].set_ylabel("npts explained")
    ax[2].set_xlabel("fid")
    ax[2].legend(loc="upper right", ncol=2, fontsize=8)
    ax[2].grid(alpha=0.3)
    ax[2].text(0.01, 0.95, "red band = far INFERRED (n_far<300)",
               transform=ax[2].transAxes, fontsize=8, va="top", color="red")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", action="store_true",
                    help="42f (fid 1018~1059) 만. 미지정시 full 200f (fid 1018~1217).")
    args = ap.parse_args()
    main(cycle=args.cycle)
