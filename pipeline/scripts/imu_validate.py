"""
imu_validate.py — Xsens MTw 무릎각 vs SMPL depth-fit 무릎각 검증 (NB7)

무릎각 정의 (SMPL 쪽 knee_fit 과 동일):
  thigh 장축과 shank 장축 사이 3D 각 (곧게 펴면 0°, unsigned).
  SMPL: J[hip]→J[knee] vs J[knee]→J[ankle] 벡터 사이각 (depthfit_*_lam0.py leg_flexion_deg).
  IMU : 정지 기립 구간에서 각 센서 좌표계의 '위쪽(중력 반대)' 방향 = 분절 장축으로 캘리브레이션
        → 보행 중 R_sensor(t)·u 로 전역 장축을 얻어 두 장축 사이각 계산.
  (보조) hinge flexion: 상대회전의 지배축 투영 (signed) — 진단용.

동기화:
  1) coarse  : 영상 전체 motion envelope(NB1) vs IMU shank 각속도 envelope 상호상관 (전 구간 탐색)
  2) fine    : coarse ±FINE_WIN 안에서 무릎각 곡선(좌+우) 상호상관 → 최종 offset
  (옵션) jump : --jump-fid 주면 영상 motion 최대 ↔ IMU |FreeAcc| 최대(착지) 로 앵커 → fine 은 앵커 ±0.3s
  3) 판정     : 점프 앵커 있으면 |fine-앵커| < 0.25s (창 경계 아님), 없으면 |coarse-fine| < 0.5 보행주기

매핑 (DeviceId → 분절):
  00B4BB52 R_FOOT / 00B4BA83 R_THIGH / 00B4BAF3 R_SHANK
  00B4BB4F L_FOOT / 00B4BB4D L_THIGH / 00B4BB4E L_SHANK
near(카메라쪽) = 오른다리, far = 왼다리 (A0 v2.1 규약)

사용:
  python imu_validate.py --imu-prefix MT_01200295_002_60fps_walking [--imu-only]
  --imu-only : 영상 없이 IMU 무릎각/정지구간/샘플링만 점검 (설정 확인용)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walkcfg  # noqa: E402

SENSORS = {"00B4BB52": "R_FOOT", "00B4BA83": "R_THIGH", "00B4BAF3": "R_SHANK",
           "00B4BB4F": "L_FOOT", "00B4BB4D": "L_THIGH", "00B4BB4E": "L_SHANK"}
FS_IMU = 100.0          # 실측: 샘플 수 / (mtb 시작→기록종료 시각) = 100 Hz (파일명 '60fps' 와 무관)
STATIC_WIN_S = 2.0      # 캘리브레이션 정지 구간 길이
STATIC_SEARCH_S = 60.0  # 처음 N 초 안에서 정지 구간 탐색
FINE_WIN_S = 1.5        # fine sync 탐색 반폭 (coarse 기준)
FINE_WIN_JUMP_S = 0.3   # 점프 앵커가 있을 때 fine 탐색 반폭 (< 반 보행주기 → 주기 모호성 없음)
JUMP_SEARCH_S = 5.0     # coarse 예측 위치 ±N초 에서 IMU 착지 피크 탐색
THRESH_DEG = 3.0        # thesis 목표


# ---------------------------------------------------------------- IMU 로드
def load_mt_txt(path):
    hdr, rows = None, []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("//"):
                continue
            p = line.rstrip("\n").split("\t")
            if hdr is None:
                hdr = [h.strip() for h in p]
                continue
            if p and p[0].strip():
                rows.append(p)
    ix = {h: i for i, h in enumerate(hdr)}

    def col(name):
        j = ix[name]
        return np.array([float(r[j]) if j < len(r) and r[j].strip() else np.nan for r in rows])

    pc = np.array([int(r[0]) for r in rows], np.int64)
    pc = np.concatenate([[0], np.cumsum(np.where(np.diff(pc) < 0, np.diff(pc) + 65536, np.diff(pc)))])
    q = np.stack([col(f"Quat_q{i}") for i in range(4)], 1)            # (w,x,y,z) sensor→ENU
    gyr = np.stack([col(f"Gyr_{a}") for a in "XYZ"], 1)
    facc = (np.stack([col(f"FreeAcc_{a}") for a in "ENU"], 1) if "FreeAcc_E" in ix
            else np.full((len(rows), 3), np.nan))                       # 중력 제거 가속도 (점프 검출)
    return pc, q, gyr, facc


def load_imu_session(imu_dir, prefix):
    out = {}
    for sid, seg in SENSORS.items():
        fs = glob.glob(os.path.join(imu_dir, f"{prefix}-000_{sid}.txt"))
        if not fs:
            raise SystemExit(f"IMU txt 없음: {prefix}-000_{sid}.txt in {imu_dir}")
        out[seg] = load_mt_txt(fs[0])
    n = min(len(v[0]) for v in out.values())
    pcs = [v[0][:n] for v in out.values()]
    if not all(np.array_equal(pcs[0], p) for p in pcs):
        raise SystemExit("센서 간 PacketCounter 불일치 — 동기 내보내기 확인")
    gaps = int((np.diff(pcs[0]) != 1).sum())
    t = pcs[0] / FS_IMU
    data = {seg: dict(q=v[1][:n], gyr=v[2][:n], facc=v[3][:n]) for seg, v in out.items()}
    return t, data, gaps


def qrot(q, v):
    """quaternion (w,x,y,z) 배열로 벡터 v 회전 (sensor→global)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    return np.einsum("nij,nj->ni", R, v if v.ndim == 2 else np.broadcast_to(v, (len(q), 3)))


def find_static(t, data, search_s=STATIC_SEARCH_S, win_s=STATIC_WIN_S):
    gm = sum(np.linalg.norm(data[s]["gyr"], axis=1) for s in ("R_THIGH", "R_SHANK", "L_THIGH", "L_SHANK"))
    n_search = int(min(len(t), search_s * FS_IMU))
    w = int(win_s * FS_IMU)
    c = np.convolve(gm[:n_search], np.ones(w) / w, mode="valid")
    i0 = int(np.argmin(c))
    return slice(i0, i0 + w), float(c[i0] / 4)


def imu_knee_angles(t, data, static):
    """정지 기립 캘리브레이션 → 무릎 장축각(unsigned) + hinge signed flexion."""
    up = np.array([0.0, 0.0, 1.0])     # ENU 위쪽
    res = {}
    for side in ("R", "L"):
        ax = {}
        for seg in ("THIGH", "SHANK"):
            q = data[f"{side}_{seg}"]["q"]
            qs = q[static]
            # 정지 구간: 센서 좌표계에서 본 '위쪽' = R^T·up (분절 장축, 근위 방향)
            Rt_up = qrot(qs * np.array([1.0, -1.0, -1.0, -1.0]), up)   # 켤레 = global→sensor
            u = Rt_up.mean(0); u /= np.linalg.norm(u)
            ax[seg] = qrot(q, u)                               # 전역 장축 (N,3)
        cosv = np.clip(np.einsum("ij,ij->i", ax["THIGH"], ax["SHANK"]), -1, 1)
        knee = np.degrees(np.arccos(cosv))
        res[side] = dict(knee=knee, static_knee=float(knee[static].mean()))
    return res


def envelope(x, fs, win_s=2.0, out_fs=10.0):
    w = max(1, int(win_s * fs))
    e = np.convolve(np.abs(x), np.ones(w) / w, mode="same")
    tt = np.arange(0, len(x) / fs, 1 / out_fs)
    return tt, np.interp(tt, np.arange(len(x)) / fs, e)


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu-prefix", required=True, help="예: MT_01200295_002_60fps_walking")
    ap.add_argument("--imu-dir", default=f"{walkcfg.ROOT}/raw/imu")
    ap.add_argument("--knee-npz", default=f"{walkcfg.KNEE_DIR}/far_knee_final.npz")
    ap.add_argument("--imu-only", action="store_true")
    ap.add_argument("--jump-fid", type=int, default=None,
                    help="영상에서 동기화 점프 근처 fid (예: 3164). 점프 동작의 최대 순간을 영상(움직임)/IMU(가속도) 양쪽에서 찾아 앵커로 사용")
    ap.add_argument("--offset-s", type=float, default=None,
                    help="수동 offset (영상 t + offset = IMU t). 지정 시 sync 생략")
    args = ap.parse_args()
    os.makedirs(walkcfg.IMU_OUT, exist_ok=True)

    t, data, gaps = load_imu_session(args.imu_dir, args.imu_prefix)
    static, static_gyr = find_static(t, data)
    ka = imu_knee_angles(t, data, static)
    print(f"[IMU] {args.imu_prefix}: {len(t)} samples, {t[-1]:.1f}s @ {FS_IMU:.0f}Hz, gaps={gaps}")
    print(f"[IMU] 정지 캘리브레이션 구간 t={t[static.start]:.1f}~{t[static.stop - 1]:.1f}s "
          f"(평균 |gyro| {static_gyr:.3f} rad/s)")
    for s in ("R", "L"):
        k = ka[s]["knee"]
        print(f"[IMU] {s} knee: static={ka[s]['static_knee']:.2f}°  p5={np.percentile(k, 5):.1f}°  "
              f"p95={np.percentile(k, 95):.1f}°  max={k.max():.1f}°")

    gyr_env = sum(np.linalg.norm(data[s]["gyr"], axis=1) for s in ("R_SHANK", "L_SHANK"))

    if args.imu_only:
        fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        ax[0].plot(t, gyr_env, lw=.5); ax[0].set_ylabel("|gyro| shank R+L (rad/s)")
        ax[0].axvspan(t[static.start], t[static.stop - 1], color="g", alpha=.3, label="static calib")
        ax[0].legend()
        ax[1].plot(t, ka["R"]["knee"], lw=.6, label="R knee"); ax[1].plot(t, ka["L"]["knee"], lw=.6, label="L knee")
        ax[1].set_ylabel("knee (deg)"); ax[1].set_xlabel("IMU t (s)"); ax[1].legend()
        fig.tight_layout(); fp = f"{walkcfg.IMU_OUT}/imu_only_{args.imu_prefix}.png"
        fig.savefig(fp, dpi=110); print("[saved]", fp)
        np.savez(f"{walkcfg.IMU_OUT}/imu_knee_{args.imu_prefix}.npz", t=t,
                 knee_R=ka["R"]["knee"], knee_L=ka["L"]["knee"], gyr_env=gyr_env,
                 static=np.array([static.start, static.stop]))
        return

    # ---- 영상 쪽 ----
    meta = json.load(open(f"{walkcfg.EXTRACT_DIR}/meta.json", encoding="utf-8"))
    env = np.load(f"{walkcfg.EXTRACT_DIR}/motion_envelope.npz")
    kn = np.load(args.knee_npz, allow_pickle=True)
    fid = kn["fid"].astype(int)
    t_frame = np.interp(fid, env["fid"], env["t"])         # 프레임 시각 (영상 기준 초)
    smpl = {"R": kn["knee_near"].astype(float), "L": kn["knee_far_final"].astype(float)}
    label = kn["label"].astype(str) if "label" in kn.files else np.array(["MEASURED"] * len(fid))
    # 보행주기(stride): near 무릎각 자기상관 (한쪽 다리 → stride 주기)
    x = smpl["R"] - smpl["R"].mean(); ac = np.correlate(x, x, "full")[len(x) - 1:]; ac /= ac[0]
    fps_eff = 1.0 / float(np.median(np.diff(t_frame)))      # 실제 추출 간격 (step=2 면 60fps 영상도 30)
    lo, hi = int(0.7 * fps_eff), min(len(ac) - 1, int(2.0 * fps_eff))
    period_s = (lo + int(np.argmax(ac[lo:hi]))) / fps_eff
    print(f"[gait] stride period = {period_s:.2f}s (near knee autocorr)")

    # 1) coarse: 전체 envelope 상관 (10 Hz)
    if args.offset_s is None:
        def coarse_sync(win_s):
            tv, ev = envelope(env["motion"], meta["fps"], win_s=win_s, out_fs=10.0)
            tv = tv + env["t"][0]
            ti, ei = envelope(gyr_env, FS_IMU, win_s=win_s, out_fs=10.0)
            big_t = np.arange(-tv[-1], ti[-1] + tv[-1], 0.1)
            A = np.interp(big_t, ti, ei, left=np.nan, right=np.nan)
            lg, cr = [], []
            for L in np.arange(-tv[-1], ti[-1], 0.1):          # 영상 t=0 이 IMU t=L 에 해당
                B = np.interp(big_t, tv + L, ev, left=np.nan, right=np.nan)
                m = np.isfinite(A) & np.isfinite(B)
                if m.sum() < 50:
                    continue
                a, b = A[m], B[m]
                lg.append(L)
                cr.append(np.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std() + 1e-9))
            lg, cr = np.array(lg), np.array(cr)
            return lg, cr, float(lg[np.nanargmax(cr)]), float(np.nanmax(cr))
        lags, cc, coarse, r_coarse = coarse_sync(2.0)
        print(f"[sync coarse] offset={coarse:.2f}s (영상 t + offset = IMU t), r={r_coarse:.3f}")
        # 강건성: 평활 폭을 바꿔도 같은 위치인가 (점프 없는 녹화에서 특히 중요)
        alt = {w: coarse_sync(w)[2:] for w in (1.0, 4.0)}
        spread = max(abs(v[0] - coarse) for v in alt.values())
        print("[sync coarse 강건성] " + "  ".join(f"win {w:.0f}s → {v[0]:.2f}s (r {v[1]:.2f})" for w, v in alt.items())
              + f"  | 최대 차이 {spread:.2f}s")

        # 1-b) 점프 앵커: 영상 motion 최대(jump_fid ±1.5s) ↔ IMU 6센서 |FreeAcc| 합 최대(coarse 예측 ±5s)
        #      둘 다 착지가 아니라 공중에서 다리를 접는 순간에 최대가 나옴(0729_full: 영상 3191 ↔ IMU 95.36s).
        #      도약(3185↔95.17)·실제 착지(3199↔95.645)로 맞춰도 offset 차 ≤ 0.03s 로 일치 확인 (2026-10-08).
        # 점프가 없으면 fine 탐색을 보행주기의 30% 이내로 제한 → 한 걸음 어긋난 위치를 고를 수 없음
        #   (001 실측: 점프 없이 coarse 만으로 착지 기준과 0.07s 차)
        center, half, anchor = coarse, min(FINE_WIN_S, 0.3 * period_s), None
        if args.jump_fid is not None:
            fps = meta["fps"]
            wv = np.abs(env["fid"] - args.jump_fid) <= int(1.5 * fps)
            iv = np.where(wv)[0][np.argmax(env["motion"][wv])]
            t_vj = float(env["t"][iv])
            acc = sum(np.linalg.norm(data[s]["facc"], axis=1) for s in data)
            if np.all(np.isnan(acc)):
                raise SystemExit("FreeAcc 열 없음 — MT Manager 내보내기에 Free Acceleration 체크")
            pred = t_vj + coarse
            wi = np.abs(t - pred) <= JUMP_SEARCH_S
            if not wi.any():
                wi = np.ones(len(t), bool)
            ii = np.where(wi)[0][np.nanargmax(acc[wi])]
            t_ij = float(t[ii])
            anchor = t_ij - t_vj
            center, half = anchor, FINE_WIN_JUMP_S
            print(f"[sync jump: 점프 동작 최대]   영상 fid {int(env['fid'][iv])} (t={t_vj:.2f}s) ↔ IMU t={t_ij:.2f}s "
                  f"(|FreeAcc| {acc[ii]:.0f}, 주변 중앙값 {np.nanmedian(acc[wi]):.0f}) → offset={anchor:.3f}s "
                  f"| coarse 와 차 {anchor - coarse:+.2f}s")

        # 2) fine: 무릎각 곡선 상관 (100 Hz 격자), center ±half
        best = None
        for off in np.arange(center - half, center + half + 1e-9, 0.01):
            rs = []
            for s in ("R", "L"):
                ok = (label == "MEASURED") if s == "L" else np.ones(len(fid), bool)
                imu_at = np.interp(t_frame + off, t, ka[s]["knee"], left=np.nan, right=np.nan)
                m = ok & np.isfinite(imu_at)
                if m.sum() > 20:
                    rs.append(np.corrcoef(imu_at[m], smpl[s][m])[0, 1])
            r = float(np.mean(rs)) if rs else np.nan
            if best is None or (np.isfinite(r) and r > best[1]):
                best = (off, r)
        offset, r_fine = best
        # 모호성: ±1 주기 위치의 상관
        def r_at(off):
            imu_at = np.interp(t_frame + off, t, ka["R"]["knee"], left=np.nan, right=np.nan)
            m = np.isfinite(imu_at)
            return np.corrcoef(imu_at[m], smpl["R"][m])[0, 1] if m.sum() > 20 else np.nan
        r_m, r_p = r_at(offset - period_s), r_at(offset + period_s)
        print(f"[sync fine]   offset={offset:.3f}s  r={r_fine:.3f}  "
              f"| ±1주기 r={r_m:.3f}/{r_p:.3f}  | coarse-fine={offset - coarse:+.2f}s")
        if anchor is not None:
            # 점프 앵커 역할 = 보행주기 단위 모호성 제거. 영상 motion 최대 ≠ 정확한 착지 프레임이라
            # 앵커 자체 오차 ~0.1s 는 정상. fine 이 탐색창 경계(±0.3s)에 붙으면 앵커 의심.
            sync_ok = abs(offset - anchor) < FINE_WIN_JUMP_S - 0.05
            print(f"[sync 판정] {'PASS' if sync_ok else 'CHECK'} (|jump-fine| = {abs(offset - anchor):.3f}s "
                  f"< {FINE_WIN_JUMP_S - 0.05:.2f}s, 반 주기 {0.5 * period_s:.2f}s 보다 작음)")
        else:
            edge = abs(offset - coarse) >= half - 0.02          # 탐색창 경계에 붙음 = 의심
            sync_ok = (not edge) and spread < 0.3
            print(f"[sync 판정] {'PASS' if sync_ok else 'CHECK'} (점프 없음: fine 탐색 ±{half:.2f}s = 보행주기 30%, "
                  f"|coarse-fine| {abs(offset - coarse):.2f}s {'경계' if edge else '안쪽'}, coarse 강건성 {spread:.2f}s < 0.3s)")
    else:
        offset, r_fine, coarse, sync_ok = args.offset_s, np.nan, np.nan, True
        lags = cc = None

    # 3) 오차
    rows, curves = [], {}
    for s, name in (("R", "near(R)"), ("L", "far(L)")):
        imu_at = np.interp(t_frame + offset, t, ka[s]["knee"], left=np.nan, right=np.nan)
        curves[s] = imu_at
        for subset, m0 in (("ALL", np.ones(len(fid), bool)),
                           ("MEASURED", label == "MEASURED"),
                           ("INTERP", label != "MEASURED")):
            if s == "R" and subset != "ALL":
                continue
            m = m0 & np.isfinite(imu_at)
            if m.sum() < 5:
                continue
            e = smpl[s][m] - imu_at[m]
            bias = float(e.mean())
            rows.append(dict(leg=name, subset=subset, n=int(m.sum()),
                             rmse=float(np.sqrt(np.mean(e ** 2))),
                             rmse_debiased=float(np.sqrt(np.mean((e - bias) ** 2))),
                             bias=bias, mae=float(np.mean(np.abs(e))),
                             max_abs=float(np.max(np.abs(e))),
                             r=float(np.corrcoef(smpl[s][m], imu_at[m])[0, 1]),
                             rom_smpl=float(np.ptp(smpl[s][m])), rom_imu=float(np.ptp(imu_at[m])),
                             peak_smpl=float(np.max(smpl[s][m])), peak_imu=float(np.max(imu_at[m]))))
    print("\n leg      subset    n   RMSE  RMSE(bias제거)  bias    MAE    max    r     ROM smpl/imu   판정(<3°)")
    for r in rows:
        print(f" {r['leg']:8s} {r['subset']:8s} {r['n']:4d} {r['rmse']:6.2f} {r['rmse_debiased']:10.2f}   "
              f"{r['bias']:+6.2f} {r['mae']:6.2f} {r['max_abs']:6.2f} {r['r']:5.3f}  "
              f"{r['rom_smpl']:5.1f}/{r['rom_imu']:5.1f}   "
              f"{'PASS' if r['rmse'] < THRESH_DEG else 'FAIL'}")

    out = dict(imu_prefix=args.imu_prefix, session=walkcfg.SESSION, fs_imu=FS_IMU,
               offset_s=offset, coarse_offset_s=coarse,
               jump_offset_s=(anchor if args.offset_s is None else None), jump_fid=args.jump_fid, r_fine=r_fine, sync_ok=bool(sync_ok),
               static_window_s=[float(t[static.start]), float(t[static.stop - 1])],
               static_knee_deg={s: ka[s]["static_knee"] for s in "RL"},
               threshold_deg=THRESH_DEG, metrics=rows)
    json.dump(out, open(f"{walkcfg.IMU_OUT}/imu_validation.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    np.savez(f"{walkcfg.IMU_OUT}/imu_validation.npz", fid=fid, t_frame=t_frame,
             smpl_R=smpl["R"], smpl_L=smpl["L"], imu_R=curves["R"], imu_L=curves["L"],
             label=label, offset_s=offset)

    fig, ax = plt.subplots(3 if lags is not None else 2, 1, figsize=(13, 10))
    for i, s in enumerate(("R", "L")):
        ax[i].plot(fid, curves[s], "k-", lw=1.4, label="IMU (Xsens)")
        ax[i].plot(fid, smpl[s], "r-", lw=1.0, label="SMPL depth-fit")
        if s == "L":
            bad = label != "MEASURED"
            ax[i].plot(fid[bad], smpl[s][bad], "o", ms=3, mfc="none", c="orange", label="INTERPOLATED")
        rr = [r for r in rows if r["leg"].startswith("near" if s == "R" else "far") and r["subset"] == "ALL"]
        ttl = f"{'near(R)' if s == 'R' else 'far(L)'} knee"
        if rr:
            ttl += f"  RMSE {rr[0]['rmse']:.2f}°  bias {rr[0]['bias']:+.2f}°  r {rr[0]['r']:.3f}"
        ax[i].set_title(ttl); ax[i].set_ylabel("deg"); ax[i].legend(fontsize=8)
    if lags is not None:
        ax[2].plot(lags, cc); ax[2].axvline(offset, c="r", ls="--")
        ax[2].set_title(f"coarse sync (envelope xcorr)  offset={offset:.3f}s")
        ax[2].set_xlabel("offset (s)")
    fig.tight_layout(); fp = f"{walkcfg.IMU_OUT}/imu_validation.png"
    fig.savefig(fp, dpi=110)
    print("[saved]", fp)


if __name__ == "__main__":
    main()
