"""
depthfit_far_leg.py
-------------------
far(왼) 다리 9-DOF (L_hip/knee/ankle body_pose) 를 depth 표면에 refine.
near depthfit 완료 상태에서 실행. betas/root/near-θ 전부 고정.

near 와 다른 3 안전장치:
  1. MEASURED / INFERRED 프레임 구분
     - MEASURED: far 점 >= MIN_FAR_PTS -> fit
     - INFERRED: 그 외 -> fit 스킵, θ_far = 4D 유지, 라벨만
  2. NEAR θ = A2 결과 주입 후 near residual 재확인 (far 가 near 안 건드리는지 검증)
  3. 곡선: MEASURED 실선, INFERRED 4D 점선 (측정/추정 시각적 분리)

불변:
  - betas STAND freeze
  - root (go, tr) = A0
  - near θ = A2 depthfit 결과 (전 프레임)
  - side = A0 side_from_zorder (FAR 라벨 점만 데이터항)
  - visible normal.z<0, unidirectional Huber δ=3cm

게이트:
  far1  : 첫 MEASURED 1 프레임 (residual before/after)
  far2  : NEAR fit 있는 전 프레임 (기본)
  far3  : F3 near fit 필요 (--near-npz 지정)

산출:
  out/depthfit_far_leg_{stage}.npz
  out/depthfit_far_leg_{stage}.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import walkcfg  # [colab_kit] 경로/세션 설정
from leg_data_term import leg_data_energy
CLOTH_MASK = None   # main 에서 walkcfg.cloth_mask() 로 설정 (하의 한쪽 항)
import os, sys, json, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- A0 원본 ----
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from a0_align_v2 import (side_from_zorder, backproject,
                         SIDE_UPPER, SIDE_FAR, SIDE_NEAR,
                         vertex_normals, HUBER_DELTA_M, huber)

# ---- 경로 ----
DEPTH_SEQ  = walkcfg.DEPTH_SEQ  # [colab_kit]
FROZEN_NPZ = walkcfg.FROZEN_NPZ  # [colab_kit]
FOURDH_SEQ = walkcfg.FOURDH_SEQ  # [colab_kit]
A0_NPZ     = walkcfg.A0_NPZ  # [colab_kit]
SMPL_MODEL = walkcfg.SMPL_MODEL  # [colab_kit]
CAM_INTR   = walkcfg.CAM_INTR  # [colab_kit]
NEAR_F2    = walkcfg.NEAR_F2  # [colab_kit]
NEAR_F3    = walkcfg.NEAR_F3  # [colab_kit]
OUT_DIR    = walkcfg.FAR_DIR  # [colab_kit]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 최적화 hyper ----
LR         = 0.02
N_ITERS    = 100
LAM_PRIOR  = 0.0           # ★ 09 prior_ablation 결론 반영 (원래 0.02)
MIN_FAR_PTS = 300          # MEASURED 최소 점수 (하드코딩 X — 분포 print 후 조정)
LEG_Y_MARGIN_UP   = 0.02
LEG_Y_MARGIN_DOWN = 0.10

# SMPL 인덱스 (far = L 다리)
J_LHIP, J_LKNEE, J_LANKLE = 1, 4, 7
BP_LHIP, BP_LKNEE, BP_LANKLE = 0, 3, 6        # joint 1/4/7 -> bp 0/3/6
# near (L 무관, but NEAR θ 주입용)
J_RHIP, J_RKNEE, J_RANKLE = 2, 5, 8
BP_RHIP, BP_RKNEE, BP_RANKLE = 1, 4, 7


def leg_flexion_deg(joints_np, hip, knee, ankle):
    t = joints_np[knee] - joints_np[hip]
    s = joints_np[ankle] - joints_np[knee]
    cosa = float(np.dot(t, s) / (np.linalg.norm(t) * np.linalg.norm(s) + 1e-9))
    return float(np.degrees(np.arccos(np.clip(cosa, -1, 1))))


def prep_far_leg_points(pts, side, joints):
    """FAR + Y-band (foot spillover 배제)."""
    far = pts[side == SIDE_FAR]
    if len(far) < 30:
        return far
    hip_y   = float(joints[J_LHIP, 1])
    ankle_y = float(joints[J_LANKLE, 1])
    y_lo = min(hip_y, ankle_y) - LEG_Y_MARGIN_UP
    y_hi = max(hip_y, ankle_y) + LEG_Y_MARGIN_DOWN
    mask = (far[:, 1] >= y_lo) & (far[:, 1] <= y_hi)
    return far[mask]


def compute_residual_cm(smpl, bt, bp, go, tr, pts_t, faces):
    """현재 pose 에서 점 -> visible SMPL 최단거리 median cm."""
    with torch.no_grad():
        out = smpl(betas=bt, body_pose=bp, global_orient=go, transl=tr)
        v = out.vertices[0]
        vn = vertex_normals(v, faces)
        vv = v[vn[:, 2] < 0.0]
        d_list = []
        for i in range(0, pts_t.shape[0], 4096):
            d_list.append(torch.cdist(pts_t[i:i + 4096], vv).min(1).values)
        d = torch.cat(d_list)
    return float(d.median().item() * 100.0)


def fit_far_frame(smpl, bt, bp_locked, go, tr, far_pts, faces,
                  lam_prior=LAM_PRIOR, iters=N_ITERS, lr=LR):
    """far 9-DOF Adam fit. bp_locked = 4D 에 near-A2 주입한 것.
    반환: theta_fit(9,), joints_fit(24,3), resid_before_cm, resid_after_cm."""
    bp_r = bp_locked.reshape(1, 23, 3).clone()
    theta = torch.tensor(
        np.concatenate([bp_r[0, i].detach().cpu().numpy()
                        for i in (BP_LHIP, BP_LKNEE, BP_LANKLE)], axis=0),
        device=DEVICE, dtype=torch.float32
    ).clone().requires_grad_(True)
    theta0 = theta.detach().clone()

    pts_t = torch.tensor(far_pts, device=DEVICE, dtype=torch.float32)

    resid_before_cm = compute_residual_cm(smpl, bt, bp_locked, go, tr, pts_t, faces)

    opt = torch.optim.Adam([theta], lr=lr)
    for it in range(iters):
        opt.zero_grad()
        bp_cur = bp_r.clone()
        bp_cur[0, BP_LHIP]   = theta[0:3]
        bp_cur[0, BP_LKNEE]  = theta[3:6]
        bp_cur[0, BP_LANKLE] = theta[6:9]
        out = smpl(betas=bt, body_pose=bp_cur.reshape(1, 69),
                   global_orient=go, transl=tr)
        v = out.vertices[0]
        vn = vertex_normals(v, faces)
        E_data  = leg_data_energy(pts_t, v, vn, HUBER_DELTA_M, CLOTH_MASK, walkcfg.CLOTH_TOL_M)
        E_prior = ((theta - theta0) ** 2).mean()
        (E_data + lam_prior * E_prior).backward()
        opt.step()

    with torch.no_grad():
        bp_cur = bp_r.clone()
        bp_cur[0, BP_LHIP]   = theta[0:3]
        bp_cur[0, BP_LKNEE]  = theta[3:6]
        bp_cur[0, BP_LANKLE] = theta[6:9]
        bp_flat = bp_cur.reshape(1, 69)
        out = smpl(betas=bt, body_pose=bp_flat, global_orient=go, transl=tr)
        joints_fit = out.joints[0].detach().cpu().numpy()
        v = out.vertices[0]
        vn = vertex_normals(v, faces)
        vv = v[vn[:, 2] < 0.0]
        d_list = []
        for i in range(0, pts_t.shape[0], 4096):
            d_list.append(torch.cdist(pts_t[i:i + 4096], vv).min(1).values)
        d = torch.cat(d_list)
        resid_after_cm = float(d.median().item() * 100.0)

    return theta.detach().cpu().numpy(), joints_fit, resid_before_cm, resid_after_cm, bp_flat


def check_near_residual_cm(smpl, bt, bp_flat_after_far, go, tr, near_pts_t, faces):
    """far fit 후 bp 로 near 점 residual 재측정 (far 가 near 로 새지 않았는지)."""
    return compute_residual_cm(smpl, bt, bp_flat_after_far, go, tr, near_pts_t, faces)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["far1", "far2", "far3"], default="far2",
                    help="far1=첫 MEASURED 1프레임  far2=NEAR fit 있는 전 프레임")
    ap.add_argument("--near-npz", default=None,
                    help="depthfit_near_leg_{f2,f3}.npz. 미지정시 F3 -> F2 자동")
    ap.add_argument("--min-far-pts", type=int, default=MIN_FAR_PTS,
                    help=f"MEASURED 최소 far 점수 (기본 {MIN_FAR_PTS})")
    args = ap.parse_args()

    # NEAR npz 선택
    near_npz = args.near_npz
    if near_npz is None:
        near_npz = NEAR_F3 if os.path.exists(NEAR_F3) else NEAR_F2
    if not os.path.exists(near_npz):
        raise SystemExit(f"near npz 없음: {near_npz}")
    print("=" * 62)
    print(f"[stage] {args.stage}")
    print(f"[near fit] {near_npz}")
    print(f"[MIN_FAR_PTS] {args.min_far_pts}   (분포 보고 조정 가능)")
    print(f"[hyper] LR={LR} iters={N_ITERS} λ_prior={LAM_PRIOR} Huber δ={HUBER_DELTA_M}m")
    print("=" * 62)

    K = json.load(open(CAM_INTR))
    if "_default" in K: K = K["_default"]

    fb = np.load(FROZEN_NPZ)
    bt = torch.tensor(fb["betas"].astype(np.float32).reshape(1, 10), device=DEVICE)

    A0 = np.load(A0_NPZ)
    a0_fids = A0["frame"].astype(int)
    a0_row = {int(f): i for i, f in enumerate(a0_fids)}
    a0_go, a0_tr = A0["global_orient"], A0["transl"]

    fdh = np.load(FOURDH_SEQ, allow_pickle=True)
    fdh_bp = np.asarray(fdh["body_pose_aa"])
    if fdh_bp.ndim == 3: fdh_bp = fdh_bp.reshape(fdh_bp.shape[0], -1)
    fdh_row = {int(f): i for i, f in enumerate(np.asarray(fdh["fids"]).astype(int))}

    near = np.load(near_npz)
    near_fids = near["fid"].astype(int)
    near_theta_all = near["theta_fit"]                # (F, 9)
    near_row = {int(f): i for i, f in enumerate(near_fids)}
    print(f"[near fit fids] n={len(near_fids)} [{near_fids.min()}~{near_fids.max()}]")

    import smplx
    smpl = walkcfg.make_smpl(SMPL_MODEL).to(DEVICE)
    global CLOTH_MASK
    CLOTH_MASK = walkcfg.cloth_mask(DEVICE)
    faces = torch.tensor(smpl.faces.astype(np.int64), device=DEVICE)
    print(f"[SMPL] loaded on {DEVICE}")

    # 처리 대상 fids
    target_fids = [int(f) for f in near_fids]
    print(f"[frames] target n={len(target_fids)} [{min(target_fids)}~{max(target_fids)}]")

    rows = []
    import time
    t_start = time.time()
    first_measured_found = False
    for i_fr, fid in enumerate(target_fids):
        r = fdh_row.get(int(fid))
        ai = a0_row.get(int(fid))
        ni = near_row.get(int(fid))
        if r is None or ai is None or ni is None: continue
        fp = os.path.join(DEPTH_SEQ, f"WALK_side_{fid:05d}_person_metric.npy")
        if not os.path.exists(fp): continue

        depth = np.load(fp)
        pts = backproject(depth, K)
        if isinstance(pts, tuple): pts = pts[0]
        side = side_from_zorder(pts)

        # bp_locked = 4D 에 near-A2 주입 (near θ 고정)
        bp4d = torch.tensor(fdh_bp[r], device=DEVICE, dtype=torch.float32).reshape(23, 3)
        near_theta = near_theta_all[ni]                 # (9,)
        bp4d[BP_RHIP]   = torch.tensor(near_theta[0:3], device=DEVICE, dtype=torch.float32)
        bp4d[BP_RKNEE]  = torch.tensor(near_theta[3:6], device=DEVICE, dtype=torch.float32)
        bp4d[BP_RANKLE] = torch.tensor(near_theta[6:9], device=DEVICE, dtype=torch.float32)
        bp_locked = bp4d.reshape(1, 69)                  # near 고정, far = 4D
        go = torch.tensor(a0_go[ai], device=DEVICE, dtype=torch.float32).reshape(1, 3)
        tr = torch.tensor(a0_tr[ai], device=DEVICE, dtype=torch.float32).reshape(1, 3)

        # 초기 forward 로 joints + 4D far 무릎각
        with torch.no_grad():
            out0 = smpl(betas=bt, body_pose=bp_locked, global_orient=go, transl=tr)
            j0 = out0.joints[0].detach().cpu().numpy()
        far_knee_4d = leg_flexion_deg(j0, J_LHIP, J_LKNEE, J_LANKLE)

        # far observability
        far_leg = prep_far_leg_points(pts, side, j0)
        n_far = len(far_leg)
        # 참고: shin/thigh 각각 (band별)
        Y = far_leg[:, 1] if len(far_leg) else np.array([])
        knee_y = float(j0[J_LKNEE, 1])
        hip_y  = float(j0[J_LHIP, 1])
        ankle_y = float(j0[J_LANKLE, 1])
        shin_n  = int(((Y >= min(knee_y, ankle_y)) & (Y < max(knee_y, ankle_y))).sum())
        thigh_n = int(((Y >= min(hip_y, knee_y)) & (Y < max(hip_y, knee_y))).sum())

        label = "MEASURED" if n_far >= args.min_far_pts else "INFERRED"

        rec = dict(
            fid=int(fid), label=label,
            n_far_leg=int(n_far), shin_n=shin_n, thigh_n=thigh_n,
            far_knee_4d=far_knee_4d,
            is_fit=False, far_theta_fit=np.zeros(9, np.float32),
            far_knee_fit=far_knee_4d,
            far_resid_before_cm=np.nan, far_resid_after_cm=np.nan,
            near_resid_before_cm=np.nan, near_resid_after_cm=np.nan,
        )

        if label == "MEASURED":
            # far fit
            theta_fit, joints_fit, r_before, r_after, bp_after = fit_far_frame(
                smpl, bt, bp_locked, go, tr, far_leg, faces,
                lam_prior=LAM_PRIOR, iters=N_ITERS, lr=LR)
            rec["is_fit"]           = True
            rec["far_theta_fit"]    = theta_fit
            rec["far_knee_fit"]     = leg_flexion_deg(joints_fit, J_LHIP, J_LKNEE, J_LANKLE)
            rec["far_resid_before_cm"] = r_before
            rec["far_resid_after_cm"]  = r_after

            # near residual 재확인 (far 가 near 안 건드리는지)
            near_leg_all = pts[side == SIDE_NEAR]
            if len(near_leg_all) >= 30:
                near_pts_t = torch.tensor(near_leg_all, device=DEVICE, dtype=torch.float32)
                rec["near_resid_before_cm"] = compute_residual_cm(smpl, bt, bp_locked, go, tr, near_pts_t, faces)
                rec["near_resid_after_cm"]  = compute_residual_cm(smpl, bt, bp_after,  go, tr, near_pts_t, faces)

            if args.stage == "far1" and not first_measured_found:
                first_measured_found = True
                rows.append(rec)
                break

        rows.append(rec)

        if args.stage != "far1" and (i_fr + 1) % 5 == 0:
            print(f"  processed {i_fr+1}/{len(target_fids)}  "
                  f"(fid {fid} {label} n_far={n_far} shin={shin_n} thigh={thigh_n})", flush=True)

    dt = time.time() - t_start
    print(f"\n[elapsed] {dt:.1f}s")

    if not rows:
        print("[ERROR] no frames processed"); return

    fids  = np.array([r["fid"] for r in rows], int)
    labs  = np.array([r["label"] for r in rows])
    n_far = np.array([r["n_far_leg"] for r in rows], int)
    shin_n= np.array([r["shin_n"] for r in rows], int)
    thigh_n=np.array([r["thigh_n"] for r in rows], int)
    is_fit= np.array([r["is_fit"] for r in rows], bool)
    far_k4 = np.array([r["far_knee_4d"] for r in rows])
    far_kf = np.array([r["far_knee_fit"] for r in rows])
    far_rb = np.array([r["far_resid_before_cm"] for r in rows])
    far_ra = np.array([r["far_resid_after_cm"] for r in rows])
    near_rb= np.array([r["near_resid_before_cm"] for r in rows])
    near_ra= np.array([r["near_resid_after_cm"] for r in rows])
    thetas = np.stack([r["far_theta_fit"] for r in rows])

    n_meas = int(is_fit.sum())
    print()
    print("=" * 62)
    print(f"[GATE-{args.stage.upper()} 요약]  n={len(rows)}   MEASURED {n_meas}/{len(rows)} ({100*n_meas/max(len(rows),1):.1f}%)")
    print("-" * 62)
    # far band 점수 분포 (MIN_FAR_PTS 조정 참고용)
    print(f"[far pts 분포] all: p10/p50/p90 = "
          f"{np.percentile(n_far,10):.0f}/{np.median(n_far):.0f}/{np.percentile(n_far,90):.0f}")
    print(f"               shin: p10/p50/p90 = "
          f"{np.percentile(shin_n,10):.0f}/{np.median(shin_n):.0f}/{np.percentile(shin_n,90):.0f}")
    print(f"               thigh: p10/p50/p90 = "
          f"{np.percentile(thigh_n,10):.0f}/{np.median(thigh_n):.0f}/{np.percentile(thigh_n,90):.0f}")
    print(f"[MIN_FAR_PTS 참고] 현재 {args.min_far_pts}   "
          f"(median 대비 {'낮음' if args.min_far_pts < np.median(n_far) else '높음'})")

    if n_meas > 0:
        m = is_fit
        print()
        print(f"[FAR2 - far residual (MEASURED만)]")
        print(f"   before  median = {np.nanmedian(far_rb[m]):.2f} cm")
        print(f"   after   median = {np.nanmedian(far_ra[m]):.2f} cm")
        print(f"   감소량           = {np.nanmedian(far_rb[m])-np.nanmedian(far_ra[m]):+.2f} cm")
        print(f"[far knee]  4D ROM = {far_k4[m].max()-far_k4[m].min():.1f}°  "
              f"fit ROM = {far_kf[m].max()-far_kf[m].min():.1f}°")
        dk = far_kf[m] - far_k4[m]
        print(f"[|Δfar knee|] median = {np.median(np.abs(dk)):.2f}°  max = {np.max(np.abs(dk)):.2f}°")

        print()
        print(f"[near residual 재확인 (far 가 near 안 건드리는지)]")
        ok = np.isfinite(near_rb) & np.isfinite(near_ra) & is_fit
        if ok.any():
            nrb = near_rb[ok]; nra = near_ra[ok]
            print(f"   before  median = {np.median(nrb):.2f} cm")
            print(f"   after   median = {np.median(nra):.2f} cm")
            diff = np.median(nra) - np.median(nrb)
            hint = "악화 없음 (설계대로 far 는 near 격리됨)" if abs(diff) < 0.05 else \
                   ("악화 - far 가 near 로 새는 신호" if diff > 0.05 else "미세 개선 (수치 노이즈)")
            print(f"   diff             = {diff:+.3f} cm   -> {hint}")

    # per-band MEASURED %
    shin_meas_pct  = 100.0 * float((shin_n >= args.min_far_pts).mean())
    thigh_meas_pct = 100.0 * float((thigh_n >= args.min_far_pts).mean())
    print()
    print(f"[FAR3 - band별 MEASURED % (기준: n >= MIN_FAR_PTS={args.min_far_pts})]")
    print(f"   shin  MEASURED = {shin_meas_pct:.1f}%")
    print(f"   thigh MEASURED = {thigh_meas_pct:.1f}%")
    print("=" * 62)

    # 저장
    out_npz = os.path.join(OUT_DIR, f"depthfit_far_leg_lam0_{args.stage}.npz")
    np.savez(out_npz,
             fid=fids, label=labs, is_fit=is_fit,
             n_far_leg=n_far, shin_n=shin_n, thigh_n=thigh_n,
             far_knee_4d=far_k4, far_knee_fit=far_kf,
             far_theta_fit=thetas,
             far_resid_before_cm=far_rb, far_resid_after_cm=far_ra,
             near_resid_before_cm=near_rb, near_resid_after_cm=near_ra)
    print(f"[saved] {out_npz}")

    # png
    if args.stage != "far1":
        fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                gridspec_kw={"height_ratios": [3, 2, 1]})
        # 위: far 무릎각
        ax[0].plot(fids, far_k4, '--', color='lightgray', label="4D far (baseline)")
        km = far_kf.copy()
        km_meas = km.copy(); km_meas[~is_fit] = np.nan
        km_infer= km.copy(); km_infer[is_fit] = np.nan
        ax[0].plot(fids, km_meas, 'o-', color='crimson', markersize=4,
                   label=f"MEASURED depth-fit  (n={n_meas})")
        ax[0].plot(fids, km_infer, 'x', color='gray', alpha=0.6,
                   label=f"INFERRED = 4D  (n={len(rows)-n_meas})")
        ax[0].set_ylabel("L(far) knee flexion [deg]")
        ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
        ax[0].set_title(f"far depth fit  |  MEASURED {100*n_meas/max(len(rows),1):.0f}%  "
                        f"| far resid: {np.nanmedian(far_rb[is_fit]) if n_meas else float('nan'):.2f}"
                        f"->{np.nanmedian(far_ra[is_fit]) if n_meas else float('nan'):.2f}cm")

        # 중: far residual (MEASURED만) + near residual check
        m = is_fit
        ax[1].plot(fids[m], far_rb[m], 'o-', color='tab:red', alpha=0.6,
                   label=f"far resid before (med {np.nanmedian(far_rb[m]):.2f})")
        ax[1].plot(fids[m], far_ra[m], 's-', color='tab:red',
                   label=f"far resid after (med {np.nanmedian(far_ra[m]):.2f})")
        ax[1].plot(fids[m], near_rb[m], 'o-', color='tab:blue', alpha=0.4,
                   label=f"near resid before (med {np.nanmedian(near_rb[m]) if m.any() else float('nan'):.2f})")
        ax[1].plot(fids[m], near_ra[m], 's-', color='tab:blue', alpha=0.8,
                   label=f"near resid after (med {np.nanmedian(near_ra[m]) if m.any() else float('nan'):.2f})")
        ax[1].set_ylabel("residual [cm]  (MEASURED)")
        ax[1].legend(fontsize=8, ncol=2); ax[1].grid(alpha=0.3)

        # 아래: observability timeline (색 띠)
        colors = np.array([[0.85, 0.15, 0.15] if f else [0.55, 0.55, 0.55]
                           for f in is_fit])
        ax[2].scatter(fids, np.zeros(len(fids)), c=colors, marker='s', s=50)
        ax[2].set_yticks([]); ax[2].set_xlabel("fid")
        ax[2].set_title("observability  red=MEASURED  gray=INFERRED", fontsize=9)

        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, f"depthfit_far_leg_lam0_{args.stage}.png")
        plt.savefig(out_png, dpi=120)
        print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
