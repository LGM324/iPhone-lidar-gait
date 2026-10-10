#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
far_thigh_npts_gate.py

far thigh 의 "튀는 구간" 을 관측량(npts) 으로 산정 → B(시간보간) vs A(prior) 필요성 판정.

원칙:
- 판정 신호 = far_thigh_npts 만 (관측 조건). residual/무릎각/IMU 안 씀 → 순환 없음.
- 임계 N_low = npts 분포의 골에서 (log10 npts 에 1D 2-means, v2.3 라벨링과 동일 방식).
- 각도(far knee) 는 flank 리스크 지표로만 (측정된 양옆 anchor 간 각차). 붕괴 프레임 각은 판정에 안 씀.

선행: knee_measurement_basis.py 를 --cycle 없이 실행해 knee_basis_full.npz 부터 생성.
     (cycle 42f 는 far 유리 window 라 붕괴 phase 거의 없음 → 이 게이트는 full 필수.)

배치: _pipeline_consolidated/D_walking_analysis/[Script]D_experiments/
산출: _pipeline_consolidated/D_walking_analysis/[Result]D_experiments/far_thigh_gate_{cycle,full}.{npz,csv,png}
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import walkcfg  # [colab_kit] 경로/세션 설정
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- 경로 (consolidated: knee_measurement_basis 와 짝) ----
RESULT_DIR = Path(walkcfg.KNEE_DIR)  # [colab_kit]
SRC_DEFAULT = RESULT_DIR / "knee_basis_full.npz"
OUT_DIR = RESULT_DIR

GAIT_LEN = int(round(walkcfg.GAIT_FRAMES))   # 4/20=42  # [colab_kit]
MAX_GAPS = [2, 3, 4, 5]


# ---------------------------------------------------------------
# 1D 2-means (log space) — v2.3 near/far 라벨링과 동일 사고
# ---------------------------------------------------------------
def kmeans_1d(x, iters=100):
    lo, hi = np.percentile(x, 20), np.percentile(x, 80)
    c = np.array([lo, hi], float)
    for _ in range(iters):
        d = np.abs(x[:, None] - c[None, :])
        lab = d.argmin(1)
        newc = np.array([x[lab == k].mean() if np.any(lab == k) else c[k] for k in (0, 1)])
        if np.allclose(newc, c):
            break
        c = newc
    order = np.argsort(c)
    c = c[order]
    lab = (x >= (c[0] + c[1]) / 2).astype(int)   # 0=low(붕괴) 1=high(가시)
    return c, lab


def valley_from_hist(x, nbins=30):
    """히스토그램 최소밀도 bin (교차확인용)."""
    h, edges = np.histogram(x, bins=nbins)
    lo_pk = np.argmax(h[: nbins // 2])
    hi_pk = nbins // 2 + np.argmax(h[nbins // 2:])
    if hi_pk <= lo_pk:
        return None
    v = lo_pk + np.argmin(h[lo_pk: hi_pk + 1])
    return 0.5 * (edges[v] + edges[v + 1])


# ---------------------------------------------------------------
# 붕괴 run 추출
# ---------------------------------------------------------------
def runs_of_true(mask):
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))   # [i, j)
            i = j
        else:
            i += 1
    return runs


def main(src):
    src_path = Path(src)
    if not src_path.exists():
        raise SystemExit(
            f"[ERROR] not found: {src_path}\n"
            f"  선행 필요: knee_measurement_basis.py 를 --cycle 없이 실행해 knee_basis_full.npz 부터 만들 것.")
    d = np.load(src_path)
    fid = d["fid"].astype(int)
    order = np.argsort(fid)
    fid = fid[order]
    npts = d["thigh_far_npts"][order].astype(float)
    knee_far = d["knee_far"][order].astype(float)
    resid = d["thigh_far_resid_cm"][order].astype(float)  # 참고 출력용(판정 X)

    is_cycle = len(fid) <= 60
    tag = "cycle" if is_cycle else "full"
    if is_cycle:
        print("[WARN] cycle(≤60f) 로 보임 — far 유리 window 라 붕괴 과소평가. full 200f 권장.\n")

    # --- 1. 임계 N_low ---
    logn = np.log10(np.clip(npts, 1, None))
    cent_log, lab = kmeans_1d(logn)
    N_low_km = 10 ** ((cent_log[0] + cent_log[1]) / 2)
    v_log = valley_from_hist(logn)
    N_low_hist = 10 ** v_log if v_log is not None else np.nan
    lo_frac = float(np.mean(lab == 0))
    separation = (cent_log[1] - cent_log[0]) / (logn.std() + 1e-9)

    N_low = N_low_km
    print("=" * 60)
    print(f"[분포] N_low (2-means)      = {N_low_km:,.0f} pts")
    print(f"       N_low (hist valley)  = {N_low_hist:,.0f} pts (교차확인)")
    print(f"       centroids(pts)       = {10**cent_log[0]:,.0f} / {10**cent_log[1]:,.0f}")
    print(f"       low(붕괴) 클러스터 비율 = {lo_frac:.1%},  분리도 = {separation:.2f}")
    if lo_frac < 0.05 or separation < 1.0:
        print("       [!] bimodality 약함 → clean 붕괴 임계 불확실. 임계 신뢰 낮음.")
    print("=" * 60)

    # --- 2. collapsed + run-length ---
    collapsed = npts < N_low
    runs = runs_of_true(collapsed)
    run_len = np.array([j - i for i, j in runs])
    n_frames = len(fid)
    print(f"[붕괴] 프레임 {collapsed.sum()}/{n_frames} ({collapsed.mean():.1%})")
    print(f"       run 개수 = {len(runs)}   (gait cycle 수 ≈ {n_frames/GAIT_LEN:.1f} — phase-locked 면 근접해야)")
    if len(run_len):
        print(f"       run-length 중앙={np.median(run_len):.0f}  최대={run_len.max()}  분포={np.bincount(run_len).tolist()}")
    print("-" * 60)

    # --- 3. run 별 flank + 각차 ---
    rec = []
    for (i, j) in runs:
        fb = npts[i - 1] if i - 1 >= 0 else np.nan
        fa = npts[j] if j < n_frames else np.nan
        vb = (i - 1 >= 0) and (not collapsed[i - 1])
        va = (j < n_frames) and (not collapsed[j])
        ang_b = knee_far[i - 1] if i - 1 >= 0 else np.nan
        ang_a = knee_far[j] if j < n_frames else np.nan
        dang = abs(ang_a - ang_b) if (vb and va) else np.nan
        rec.append(dict(start_fid=int(fid[i]), length=int(j - i),
                        flank_before_npts=float(fb), flank_after_npts=float(fa),
                        flank_visible=bool(vb and va), flank_angle_delta=float(dang)))
        if not np.isnan(dang):
            print(f"  run @fid {fid[i]:>5} len={j-i}  flank_npts {fb:>7,.0f}/{fa:>7,.0f}"
                  f"  visible={vb and va}  angle Δ={dang:5.1f}°")
        else:
            print(f"  run @fid {fid[i]:>5} len={j-i}  flank_npts {fb:>7,.0f}/{fa:>7,.0f}  visible={vb and va}")

    # --- 4. MAX_GAP 민감도: B vs A ---
    print("-" * 60)
    print("[B(시간보간) vs A(prior)]  MAX_GAP 별 (B = 짧은붕괴 + 양옆가시)")
    for g in MAX_GAPS:
        nB = sum(1 for r in rec if r["flank_visible"] and r["length"] <= g)
        nA = len(rec) - nB
        fB = sum(r["length"] for r in rec if r["flank_visible"] and r["length"] <= g)
        fA = int(collapsed.sum()) - fB
        print(f"   MAX_GAP={g}:  run B/A = {nB}/{nA}   frame B/A = {fB}/{fA}")
    print("=" * 60)

    # --- 저장 ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if rec:
        keys = list(rec[0].keys())
        np.savez(str(OUT_DIR / f"far_thigh_gate_{tag}.npz"),
                 N_low=N_low, **{k: np.array([r[k] for r in rec]) for k in keys})
        with open(OUT_DIR / f"far_thigh_gate_{tag}.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rec:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    plot(fid, npts, collapsed, N_low, logn, cent_log, run_len, rec,
         OUT_DIR / f"far_thigh_gate_{tag}.png", tag)
    print(f"[saved] far_thigh_gate_{tag}.png / .npz / .csv  →  {OUT_DIR}")


def plot(fid, npts, collapsed, N_low, logn, cent_log, run_len, rec, path, tag):
    fig, ax = plt.subplots(3, 1, figsize=(13, 10))

    # (1) npts timeline + N_low + 붕괴 음영
    ax[0].plot(fid, npts, "-", color="tab:green", lw=1)
    ax[0].axhline(N_low, color="k", ls=":", label=f"N_low={N_low:,.0f}")
    for i in np.where(collapsed)[0]:
        ax[0].axvspan(fid[i] - 0.5, fid[i] + 0.5, color="red", alpha=0.12)
    ax[0].set_ylabel("far_thigh_npts")
    ax[0].set_xlabel("fid")
    ax[0].set_title(f"far thigh npts — collapsed (red)  [{tag}]")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # (2) histogram (log) + centroid + N_low
    ax[1].hist(logn, bins=30, color="tab:green", alpha=0.6)
    for c in cent_log:
        ax[1].axvline(c, color="tab:blue", ls="--")
    ax[1].axvline(np.log10(N_low), color="k", ls=":", label="N_low")
    ax[1].set_xlabel("log10(npts)")
    ax[1].set_ylabel("frames")
    ax[1].set_title("npts distribution (2-means centroid=blue, N_low=black)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    # (3) run-length bars
    if len(run_len):
        vals, cnts = np.unique(run_len, return_counts=True)
        ax[2].bar(vals, cnts, color="tab:orange")
        ax[2].set_xlabel("collapsed run length (frames)")
        ax[2].set_ylabel("# runs")
    txt = "B/A runs per MAX_GAP:\n" + "\n".join(
        f"  g={g}: B={sum(1 for r in rec if r['flank_visible'] and r['length']<=g)}"
        f"/A={len(rec)-sum(1 for r in rec if r['flank_visible'] and r['length']<=g)}"
        for g in MAX_GAPS)
    ax[2].text(0.98, 0.95, txt, transform=ax[2].transAxes, va="top", ha="right",
               fontsize=9, family="monospace",
               bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax[2].set_title("collapsed run length distribution")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_DEFAULT),
                    help="knee_measurement_basis.py 산출 npz (full 권장)")
    args = ap.parse_args()
    main(args.src)
