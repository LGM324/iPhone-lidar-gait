"""
depthfit_near_leg.py
--------------------
near(오른) 다리 9-DOF (hip/knee/ankle body_pose) 를 depth 표면에 refine.

불변:
  - betas STAND freeze / segment scale 안 씀 (A4 보류)
  - root(go, tr) = A0 값 고정
  - side = A0 side_from_zorder (import)
  - visible = normal.z<0 (A0 방식)
  - Huber δ=3cm, unidirectional data->model (A0 그대로)
  - torso 가중 유지는 여기선 무관 (다리만 대상, torso 점 안 씀)

변수: R_hip(bp1), R_knee(bp4), R_ankle(bp7) 3관절 × 3axis = 9 DOF
      초기값 = 4D body_pose. Prior = ||θ - θ_4D||² (λ_prior 약)

게이트:
  F1 (1 frame) : residual before/after 확인 -> 통과시 F2
  F2 (42 frames, 1주기) : 무릎각 곡선 4D vs fit, ROM 변화, |Δknee| median
  F3 (200 frames)      : F2 통과시만 실행 (--stage f3 옵션)

산출:
  out/depthfit_near_leg_{stage}.npz  (fid, theta_fit(9), knee_4d, knee_fit,
                                       resid_before/after, rom_shift)
  out/depthfit_near_leg_{stage}.png  (4D vs fit 곡선 + residual)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import walkcfg  # [colab_kit] 경로/세션 설정
import os, sys, json, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- A0 원본 import ----
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
OUT_DIR    = walkcfg.NEAR_DIR  # [colab_kit]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 최적화 hyper
LR         = 0.02          # A0 와 동일
N_ITERS    = 100
LAM_PRIOR  = 0.0           # ★ 09 prior_ablation 결론 반영 (원래 0.01)
LEG_Y_MARGIN_UP   = 0.02   # near 다리 mask 상단 margin (hip_y 위쪽 여유)
LEG_Y_MARGIN_DOWN = 0.10   # near 다리 mask 하단 margin (ankle_y 아래쪽 여유)

# SMPL joint / body_pose 인덱스 (body_pose = joints 1..23 axis-angle stacked)
J_RHIP, J_RKNEE, J_RANKLE = 2, 5, 8
BP_RHIP, BP_RKNEE, BP_RANKLE = 1, 4, 7    # joint 2/5/8 -> bp idx 1/4/7


def leg_flexion_deg(joints_np):
    """seg 무릎각 (deg). R다리."""
    t = joints_np[J_RKNEE] - joints_np[J_RHIP]
    s = joints_np[J_RANKLE] - joints_np[J_RKNEE]
    cosa = float(np.dot(t, s) / (np.linalg.norm(t) * np.linalg.norm(s) + 1e-9))
    return float(np.degrees(np.arccos(np.clip(cosa, -1, 1))))


def compute_residual_cm(smpl, bt, bp, go, tr, near_pts_t, faces):
    """현재 pose 에서 near 다리 점 -> visible SMPL 최단거리 median cm (진단용)."""
    with torch.no_grad():
        out = smpl(betas=bt, body_pose=bp, global_orient=go, transl=tr)
        v = out.vertices[0]
        vn = vertex_normals(v, faces)
        vv = v[vn[:, 2] < 0.0]
        d_list = []
        for i in range(0, near_pts_t.shape[0], 4096):
            d_list.append(torch.cdist(near_pts_t[i:i + 4096], vv).min(1).values)
        d = torch.cat(d_list)
    return float(d.median().item() * 100.0)


def fit_frame(smpl, bt, bp_4d, go, tr, near_pts, faces,
              lam_prior=LAM_PRIOR, iters=N_ITERS, lr=LR, verbose=False):
    """near 다리 9-DOF Adam fit. 나머지 bp = 4D 고정.
    반환: theta_fit(9,), joints_fit(24,3), resid_before_cm, resid_after_cm."""
    bp_reshaped = bp_4d.reshape(1, 23, 3).clone()          # (1, 23, 3)
    # near 다리 3관절 axis-angle 만 leaf tensor 로 분리
    theta = torch.tensor(
        np.concatenate([bp_reshaped[0, i].detach().cpu().numpy()
                        for i in (BP_RHIP, BP_RKNEE, BP_RANKLE)], axis=0),
        device=DEVICE, dtype=torch.float32
    ).clone().requires_grad_(True)                          # (9,)
    theta0 = theta.detach().clone()

    pts_t = torch.tensor(near_pts, device=DEVICE, dtype=torch.float32)

    # 진단: fit 전 residual
    resid_before_cm = compute_residual_cm(smpl, bt, bp_4d, go, tr, pts_t, faces)

    opt = torch.optim.Adam([theta], lr=lr)
    for it in range(iters):
        opt.zero_grad()
        # theta -> 전신 bp 재조합
        bp_cur = bp_reshaped.clone()
        bp_cur[0, BP_RHIP]   = theta[0:3]
        bp_cur[0, BP_RKNEE]  = theta[3:6]
        bp_cur[0, BP_RANKLE] = theta[6:9]
        bp_flat = bp_cur.reshape(1, 69)
        out = smpl(betas=bt, body_pose=bp_flat, global_orient=go, transl=tr)
        v = out.vertices[0]
        vn = vertex_normals(v, faces)
        vv = v[vn[:, 2] < 0.0]
        d_list = []
        for i in range(0, pts_t.shape[0], 4096):
            d_list.append(torch.cdist(pts_t[i:i + 4096], vv).min(1).values)
        dist = torch.cat(d_list)
        E_data  = huber(dist, HUBER_DELTA_M).mean()
        E_prior = ((theta - theta0) ** 2).mean()
        loss = E_data + lam_prior * E_prior
        loss.backward()
        opt.step()
        if verbose and (it + 1) % 20 == 0:
            print(f"    iter {it+1:3d}  E_data={E_data.item():.5f}  "
                  f"E_prior={E_prior.item():.5f}  |dθ|={float((theta-theta0).norm()):.4f}",
                  flush=True)

    # 최종 forward + residual
    with torch.no_grad():
        bp_cur = bp_reshaped.clone()
        bp_cur[0, BP_RHIP]   = theta[0:3]
        bp_cur[0, BP_RKNEE]  = theta[3:6]
        bp_cur[0, BP_RANKLE] = theta[6:9]
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

    return theta.detach().cpu().numpy(), joints_fit, resid_before_cm, resid_after_cm


def prep_near_leg_points(pts, side, joints_init):
    """near + Y-band (foot spillover 배제)."""
    near = pts[side == SIDE_NEAR]
    if len(near) < 30:
        return near
    hip_y   = float(joints_init[J_RHIP, 1])
    ankle_y = float(joints_init[J_RANKLE, 1])
    y_lo = min(hip_y, ankle_y) - LEG_Y_MARGIN_UP
    y_hi = max(hip_y, ankle_y) + LEG_Y_MARGIN_DOWN
    mask = (near[:, 1] >= y_lo) & (near[:, 1] <= y_hi)
    return near[mask]


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["f1", "f2", "f3"], default="f2",
                    help="f1=1frame  f2=42frames(1 gait cycle)  f3=200frames")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("=" * 62)
    print(f"[stage] {args.stage}")
    print("[paths]")
    for k, v in dict(DEPTH_SEQ=DEPTH_SEQ, FROZEN_NPZ=FROZEN_NPZ, FOURDH_SEQ=FOURDH_SEQ,
                     A0_NPZ=A0_NPZ, SMPL_MODEL=SMPL_MODEL, CAM_INTR=CAM_INTR).items():
        print(f"  {k}: exists={os.path.exists(v)}")
    print(f"[hyper] LR={LR} iters={N_ITERS} λ_prior={LAM_PRIOR} Huber δ={HUBER_DELTA_M}m")
    print("=" * 62)

    K = json.load(open(CAM_INTR))
    if "_default" in K: K = K["_default"]

    fb = np.load(FROZEN_NPZ)
    bt = torch.tensor(fb["betas"].astype(np.float32).reshape(1, 10), device=DEVICE)

    A0 = np.load(A0_NPZ)
    a0_fids = A0["frame"].astype(int)
    a0_go, a0_tr = A0["global_orient"], A0["transl"]
    a0_row = {int(f): i for i, f in enumerate(a0_fids)}

    fdh = np.load(FOURDH_SEQ, allow_pickle=True)
    fdh_fids = np.asarray(fdh["fids"]).astype(int)
    fdh_bp   = np.asarray(fdh["body_pose_aa"])
    if fdh_bp.ndim == 3: fdh_bp = fdh_bp.reshape(fdh_bp.shape[0], -1)
    fdh_row = {int(f): i for i, f in enumerate(fdh_fids)}

    import smplx
    smpl = smplx.create(SMPL_MODEL, model_type="smpl", gender="neutral", num_betas=10).to(DEVICE)
    faces = torch.tensor(smpl.faces.astype(np.int64), device=DEVICE)
    print(f"[SMPL] loaded on {DEVICE}")

    if args.stage == "f1":
        sel = list(a0_fids[:1])
    elif args.stage == "f2":
        sel = list(a0_fids[:42])
    else:
        sel = list(a0_fids)
    print(f"[frames] n={len(sel)}  [{sel[0]}~{sel[-1]}]")
    print()

    rows = []
    import time
    t_start = time.time()
    for i_fr, fid in enumerate(sel):
        r = fdh_row.get(int(fid))
        ai = a0_row.get(int(fid))
        if r is None or ai is None: continue
        fp = os.path.join(DEPTH_SEQ, f"WALK_side_{fid:05d}_person_metric.npy")
        if not os.path.exists(fp): continue

        depth = np.load(fp)
        pts = backproject(depth, K)
        # backproject 리턴 스키마: A0 는 (pts, valid_idx) 튜플
        if isinstance(pts, tuple):
            pts = pts[0]
        side = side_from_zorder(pts)

        bp4d = torch.tensor(fdh_bp[r], device=DEVICE, dtype=torch.float32).reshape(1, -1)
        go   = torch.tensor(a0_go[ai], device=DEVICE, dtype=torch.float32).reshape(1, 3)
        tr   = torch.tensor(a0_tr[ai], device=DEVICE, dtype=torch.float32).reshape(1, 3)

        # 초기 joints 로 near leg band 결정
        with torch.no_grad():
            out0 = smpl(betas=bt, body_pose=bp4d, global_orient=go, transl=tr)
            joints0 = out0.joints[0].detach().cpu().numpy()
        knee_4d = leg_flexion_deg(joints0)

        near_leg = prep_near_leg_points(pts, side, joints0)
        if len(near_leg) < 30:
            print(f"  [skip] fid {fid}: near leg pts too few ({len(near_leg)})")
            continue

        theta_fit, joints_fit, r_before, r_after = fit_frame(
            smpl, bt, bp4d, go, tr, near_leg, faces,
            lam_prior=LAM_PRIOR, iters=N_ITERS, lr=LR,
            verbose=(args.verbose and i_fr == 0))
        knee_fit = leg_flexion_deg(joints_fit)

        rec = dict(fid=int(fid),
                   knee_4d=knee_4d, knee_fit=knee_fit, dknee=knee_fit - knee_4d,
                   resid_before_cm=r_before, resid_after_cm=r_after,
                   n_near_leg=int(len(near_leg)),
                   theta_fit=theta_fit)
        rows.append(rec)

        if args.stage == "f1":
            print(f"  [F1 fid {fid}]  n_near_leg={len(near_leg)}  "
                  f"resid before={r_before:.2f}cm  after={r_after:.2f}cm  "
                  f"knee 4D={knee_4d:.2f}° fit={knee_fit:.2f}°  Δ={rec['dknee']:+.2f}°",
                  flush=True)
        elif (i_fr + 1) % 5 == 0:
            print(f"  processed {i_fr+1}/{len(sel)}  "
                  f"(fid {fid} Δknee {rec['dknee']:+.2f}° resid {r_before:.2f}->{r_after:.2f}cm)",
                  flush=True)

    dt = time.time() - t_start
    print(f"\n[elapsed] {dt:.1f}s  ({dt/max(len(rows),1):.2f}s/frame)")

    if not rows:
        print("[ERROR] no frames processed"); return

    fids  = np.array([r["fid"] for r in rows], int)
    k4    = np.array([r["knee_4d"] for r in rows])
    kf    = np.array([r["knee_fit"] for r in rows])
    dk    = kf - k4
    rb    = np.array([r["resid_before_cm"] for r in rows])
    ra    = np.array([r["resid_after_cm"] for r in rows])
    thetas= np.stack([r["theta_fit"] for r in rows])
    nleg  = np.array([r["n_near_leg"] for r in rows], int)

    print()
    print("=" * 62)
    print(f"[GATE-{args.stage.upper()} 요약]  n={len(rows)}")
    print("-" * 62)
    print(f"[residual median]  before {np.median(rb):.2f}cm -> after {np.median(ra):.2f}cm  "
          f"(감소 {np.median(rb)-np.median(ra):+.2f}cm)")
    print(f"                   mean   before {rb.mean():.2f}cm -> after {ra.mean():.2f}cm")
    print(f"[knee angle 4D]   ROM {k4.max()-k4.min():.1f}°  med {np.median(k4):.1f}°  "
          f"range [{k4.min():.1f}, {k4.max():.1f}]")
    print(f"[knee angle fit]  ROM {kf.max()-kf.min():.1f}°  med {np.median(kf):.1f}°  "
          f"range [{kf.min():.1f}, {kf.max():.1f}]")
    print(f"[|Δknee|]         median {np.median(np.abs(dk)):.2f}°  "
          f"mean {np.mean(np.abs(dk)):.2f}°  max {np.max(np.abs(dk)):.2f}°  "
          f"p95 {np.percentile(np.abs(dk),95):.2f}°")
    if args.stage in ("f2", "f3"):
        # jitter (프레임간 |Δfit|)
        j_med = float(np.median(np.abs(np.diff(kf))))
        j_p95 = float(np.percentile(np.abs(np.diff(kf)), 95))
        print(f"[jitter fit]      median {j_med:.2f}°  p95 {j_p95:.2f}°   (4D jitter med 1.53°)")
    print("=" * 62)

    # 저장
    out_npz = os.path.join(OUT_DIR, f"depthfit_near_leg_lam0_{args.stage}.npz")
    np.savez(out_npz,
             fid=fids, knee_4d=k4, knee_fit=kf, dknee=dk,
             resid_before_cm=rb, resid_after_cm=ra,
             theta_fit=thetas, n_near_leg=nleg)
    print(f"[saved] {out_npz}")

    # png
    if args.stage != "f1":
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax[0].plot(fids, k4, 'b-o', label=f"4D (before)  ROM {k4.max()-k4.min():.1f}°", markersize=4)
        ax[0].plot(fids, kf, 'r-s', label=f"depth-fit    ROM {kf.max()-kf.min():.1f}°", markersize=4)
        ax[0].set_ylabel("R knee flexion [deg]")
        ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
        ax[0].set_title(f"depth fit near leg  |Δknee| med {np.median(np.abs(dk)):.2f}° "
                        f"max {np.max(np.abs(dk)):.2f}°  "
                        f"resid {np.median(rb):.2f}->{np.median(ra):.2f}cm")

        ax[1].plot(fids, rb, 'gray', marker='o', markersize=3, label=f"resid before (med {np.median(rb):.2f})")
        ax[1].plot(fids, ra, 'k',    marker='s', markersize=3, label=f"resid after  (med {np.median(ra):.2f})")
        ax[1].set_ylabel("near leg residual [cm]")
        ax[1].set_xlabel("fid")
        ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, f"depthfit_near_leg_lam0_{args.stage}.png")
        plt.savefig(out_png, dpi=120)
        print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
