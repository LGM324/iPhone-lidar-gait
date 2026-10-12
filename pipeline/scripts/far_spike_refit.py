"""far 다리 튐 프레임 재피팅 (NB5 far 직후, NB6 전에 실행).

튐 판정 (IMU 를 보지 않고 피팅 결과만으로, 보행의 물리적 범위 기준):
  ① 무릎 굴곡 > MAX_KNEE_DEG (트레드밀 보행 최대 굴곡 60~70°)
  ② 앞뒤 2프레임(정상 피팅만) 중앙값과 차이 > JUMP_DEG (30 fps 에서 한 프레임에 30° 이상 튀었다 돌아오는 것은 불가능)
처리:
  - 앞뒤 정상 프레임의 far θ(고관절·무릎·발목 9개)를 fid 비율로 선형 보간 → 초기값으로 그 프레임만 다시 피팅
  - 다시 판정 통과 → label REFIT (측정값으로 사용)
  - 실패 → reject=True (NB6 knee_measurement_basis 가 관측 점 0 으로 처리 → 가림과 같이 보간)
출력: FAR_F2 를 갱신 (원본은 *_orig.npz 로 한 번만 백업), refit/reject/knee_before_refit 필드 추가.
"""
import os, sys, json, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walkcfg
import depthfit_far_leg_lam0 as F
from a0_align_v2 import side_from_zorder, backproject

MAX_KNEE_DEG = 85.0
JUMP_DEG = 30.0


def spike_mask(fid, knee, ok):
    """ok = 판정에 쓸 정상 피팅 프레임. 반환: 튐 여부 (ok 인 프레임만 판정)."""
    n = len(knee); bad = np.zeros(n, bool)
    for i in range(n):
        if not ok[i]:
            continue
        if knee[i] > MAX_KNEE_DEG:
            bad[i] = True; continue
        nb = [knee[j] for j in range(max(0, i - 2), min(n, i + 3)) if j != i and ok[j] and knee[j] <= MAX_KNEE_DEG]
        if len(nb) >= 2 and abs(knee[i] - np.median(nb)) > JUMP_DEG:
            bad[i] = True
    return bad


def main():
    far_p = walkcfg.FAR_F2
    orig = far_p.replace(".npz", "_orig.npz")
    if not os.path.exists(orig):
        shutil.copy(far_p, orig)
    far = dict(np.load(orig, allow_pickle=True))
    fid = far["fid"].astype(int); knee = far["far_knee_fit"].astype(float).copy()
    is_fit = far["is_fit"].astype(bool)
    bad = spike_mask(fid, knee, is_fit)
    print(f"[spike] 판정 {is_fit.sum()} 프레임 중 튐 {bad.sum()}: {fid[bad].tolist()}")
    refit = np.zeros(len(fid), bool); reject = np.zeros(len(fid), bool)
    knee_before = knee.copy()
    if bad.any():
        dev = F.DEVICE
        K = json.load(open(walkcfg.CAM_INTR)); K = K.get("_default", K)
        fb = np.load(walkcfg.FROZEN_NPZ)
        bt = torch.tensor(fb["betas"].astype(np.float32).reshape(1, 10), device=dev)
        A0 = np.load(walkcfg.A0_NPZ); a0_row = {int(f): i for i, f in enumerate(A0["frame"])}
        fdh = np.load(walkcfg.FOURDH_SEQ, allow_pickle=True)
        fdh_bp = np.asarray(fdh["body_pose_aa"]).reshape(len(fdh["fids"]), -1)
        fdh_row = {int(f): i for i, f in enumerate(fdh["fids"])}
        near = np.load(walkcfg.NEAR_F3); near_row = {int(f): i for i, f in enumerate(near["fid"])}
        smpl = walkcfg.make_smpl(walkcfg.SMPL_MODEL).to(dev)
        F.CLOTH_MASK = walkcfg.cloth_mask(dev)                 # far 피팅과 같은 데이터 항
        faces = torch.tensor(smpl.faces.astype(np.int64), device=dev)
        good = is_fit & ~bad
        theta_all = far["far_theta_fit"].astype(np.float32).copy()
        for i in np.where(bad)[0]:
            f = int(fid[i])
            prv = [j for j in range(i - 1, -1, -1) if good[j]][:1]
            nxt = [j for j in range(i + 1, len(fid)) if good[j]][:1]
            if not prv and not nxt:
                reject[i] = True; continue
            if prv and nxt:
                a, b = prv[0], nxt[0]; w = (fid[i] - fid[a]) / (fid[b] - fid[a])
                th0 = (1 - w) * theta_all[a] + w * theta_all[b]
            else:
                th0 = theta_all[(prv or nxt)[0]]
            depth = np.load(os.path.join(walkcfg.DEPTH_SEQ, f"WALK_side_{f:05d}_person_metric.npy"))
            pts = backproject(depth, K); pts = pts[0] if isinstance(pts, tuple) else pts
            side = side_from_zorder(pts)
            bp = torch.tensor(fdh_bp[fdh_row[f]], device=dev, dtype=torch.float32).reshape(23, 3)
            nt = near["theta_fit"][near_row[f]]
            for k, idx in enumerate((F.BP_RHIP, F.BP_RKNEE, F.BP_RANKLE)):
                bp[idx] = torch.tensor(nt[3 * k:3 * k + 3], device=dev, dtype=torch.float32)
            for k, idx in enumerate((F.BP_LHIP, F.BP_LKNEE, F.BP_LANKLE)):      # far 초기값 = 앞뒤 보간
                bp[idx] = torch.tensor(th0[3 * k:3 * k + 3], device=dev, dtype=torch.float32)
            bp = bp.reshape(1, 69)
            go = torch.tensor(A0["global_orient"][a0_row[f]], device=dev, dtype=torch.float32).reshape(1, 3)
            tr = torch.tensor(A0["transl"][a0_row[f]], device=dev, dtype=torch.float32).reshape(1, 3)
            with torch.no_grad():
                j0 = smpl(betas=bt, body_pose=bp, global_orient=go, transl=tr).joints[0].cpu().numpy()
            far_leg = F.prep_far_leg_points(pts, side, j0)
            th, jf, rb, ra, _ = F.fit_far_frame(smpl, bt, bp, go, tr, far_leg, faces,
                                                lam_prior=F.LAM_PRIOR, iters=F.N_ITERS, lr=F.LR)
            k_new = F.leg_flexion_deg(jf, F.J_LHIP, F.J_LKNEE, F.J_LANKLE)
            knee_try = knee.copy(); knee_try[i] = k_new
            ok_try = good.copy(); ok_try[i] = True
            still = spike_mask(fid, knee_try, ok_try)[i]
            print(f"  fid {f}: {knee[i]:.1f}° → 재피팅 {k_new:.1f}° (초기 보간 {F.leg_flexion_deg(j0, F.J_LHIP, F.J_LKNEE, F.J_LANKLE):.1f}°, "
                  f"잔차 {ra:.2f} cm) → {'REFIT' if not still else 'REJECT'}", flush=True)
            if still:
                reject[i] = True
            else:
                refit[i] = True; knee[i] = k_new; theta_all[i] = th
                far["far_resid_after_cm"][i] = ra; far["label"][i] = "REFIT"
        far["far_theta_fit"] = theta_all
    far["far_knee_fit"] = knee; far["refit"] = refit; far["reject"] = reject; far["knee_before_refit"] = knee_before
    np.savez(far_p, **far)
    print(f"[saved] {far_p}  REFIT {refit.sum()}  REJECT {reject.sum()}  (원본 {os.path.basename(orig)})")


if __name__ == "__main__":
    main()
