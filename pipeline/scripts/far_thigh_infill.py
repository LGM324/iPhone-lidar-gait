#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
far_thigh_infill.py

far 무릎각 최종 시계열 생성.
붕괴 gap 3분기 (선형 / cubic / cubic-peak) 로 채우고 프레임마다 [값 / conf / σ / label] 부여.

설계 근거:
- Δ 가 1차 분기 (tangent 부호는 평탄 구간에서 노이즈로 뒤집힘 — 1124/1168 실측).
  Δ 분리도: trough 0.5~2.2° vs peak 12.1~20.3° → 완벽 분리.
- tangent = quad4 (2차 fit 4프레임, 끝점 미분). lin3 (창 중앙 기울기) 는 버그.
  near 모의가림 실측: lin3 bias +1.24 → quad4 +0.40, p95 2.71 → 2.64.
- bias 보정 안 함: 1022 는 overshoot +0.05 뿐이라 −0.40 하면 measured anchor (59.48) 밑으로
  내려감 → 측정값 위반. σ 에 포함하고 문서화.
- 3층 분리: 값 / conf (관측량만, VisDB) / σ (캘리브레이션). conf 를 값 품질로 올리면 순환.

배치: _pipeline_consolidated/D_walking_analysis/[Script]D_experiments/
산출: _pipeline_consolidated/D_walking_analysis/[Result]D_experiments/far_knee_final.{npz,csv,png}

선행:
- knee_basis_full.npz          (knee_measurement_basis.py)
- far_thigh_gate_full.npz      (far_thigh_npts_gate.py) → N_low
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import walkcfg  # [colab_kit] 경로/세션 설정
import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows cp949 로 특수문자 (em-dash, √, σ 등) 출력 시 크래시 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---- 경로 (consolidated) ----
RESULT_DIR = Path(walkcfg.KNEE_DIR)  # [colab_kit]
BASIS_DEFAULT = RESULT_DIR / "knee_basis_full.npz"
GATE_DEFAULT  = RESULT_DIR / "far_thigh_gate_full.npz"
OUT_DIR = RESULT_DIR

DELTA_LIN = 3.0     # Δ < 이면 선형 (관측: trough runs 0.5~2.2°)
DELTA_CUB = 12.0    # Δ > 이면 cubic (관측: peak runs 12.1~20.3°)
EPS_V     = 0.3
TAN_DEG, TAN_K = 2, 4    # quad4
LENGTHS   = [2, 3, 4]


# ---------------------------------------------------------------
def tangent(fids, vals, x0, deg=TAN_DEG):
    if len(vals) <= deg:
        deg = max(1, len(vals) - 1)
    p = np.polyfit(np.asarray(fids, float), np.asarray(vals, float), deg)
    return float(np.polyval(np.polyder(p), float(x0)))


def hermite(p0, p1, m0, m1, N, ks):
    M0, M1 = m0 * N, m1 * N

    def h(t):
        return ((2 * t**3 - 3 * t**2 + 1) * p0 + (t**3 - 2 * t**2 + t) * M0 +
                (-2 * t**3 + 3 * t**2) * p1 + (t**3 - t**2) * M1)

    vals = np.array([h(k / N) for k in ks])
    a = 6 * p0 + 3 * M0 - 6 * p1 + 3 * M1
    b = -6 * p0 - 4 * M0 + 6 * p1 - 2 * M1
    c = M0
    ext = None
    if abs(a) > 1e-9:
        disc = b * b - 4 * a * c
        if disc >= 0:
            for t in ((-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)):
                if 0 < t < 1:
                    ext = float(h(t))
    return vals, ext


def linear_fill(p0, p1, L):
    return np.array([p0 + (p1 - p0) * k / (L + 1) for k in range(1, L + 1)])


def classify_tan(vb, va, L):
    sb = 0 if abs(vb) < EPS_V else np.sign(vb)
    sa = 0 if abs(va) < EPS_V else np.sign(va)
    if sb * sa >= 0:
        return "CUB_OK"
    return "CUB_OK" if L <= 2 else "CUB_PEAK"


def branch(knee, a0, a1, L, fid):
    """far/near 공통 분기. return (regime, vals, ext, delta, vb, va)"""
    p0, p1 = float(knee[a0]), float(knee[a1])
    delta = abs(p1 - p0)
    if delta < DELTA_LIN:
        return "LIN", linear_fill(p0, p1, L), None, delta, np.nan, np.nan
    bi = list(range(a0 - TAN_K + 1, a0 + 1))
    ai = list(range(a1, a1 + TAN_K))
    vb = tangent(fid[bi], knee[bi], fid[a0])
    va = tangent(fid[ai], knee[ai], fid[a1])
    vals, ext = hermite(p0, p1, vb, va, L + 1, list(range(1, L + 1)))
    reg = classify_tan(vb, va, L)
    if DELTA_LIN <= delta <= DELTA_CUB:
        reg = "CUB_PEAK"          # 모호대 → 보수적
    return reg, vals, ext, delta, vb, va


def runs_of_true(mask):
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    return runs


def kmeans_1d(x, iters=100):
    c = np.array([np.percentile(x, 20), np.percentile(x, 80)], float)
    for _ in range(iters):
        lab = (np.abs(x[:, None] - c[None, :])).argmin(1)
        nc = np.array([x[lab == k].mean() if np.any(lab == k) else c[k] for k in (0, 1)])
        if np.allclose(nc, c):
            break
        c = nc
    return np.sort(c)


# =============================================================
# 1. σ 캘리브레이션 — near(측정 완결) 에 far 와 동일 분기 적용
# =============================================================
def calibrate(fid, knee_near):
    n = len(fid)
    samp = defaultdict(list)
    for L in LENGTHS:
        for i in range(TAN_K, n - L - TAN_K):
            a0, a1 = i - 1, i + L
            reg, vals, ext, delta, vb, va = branch(knee_near, a0, a1, L, fid)
            truth = knee_near[i:i + L]
            err = vals - truth
            true_max = float(knee_near[a0:a1 + 1].max())
            samp[(reg, L)].append(dict(
                max_abs=float(np.max(np.abs(err))),
                med_abs=float(np.median(np.abs(err))),
                bias=(float(ext) - true_max) if ext is not None else np.nan))

    print("=" * 82)
    print("[캘리브레이션] near (측정 완결) 에 far 와 동일 분기로 인공 gap → regime 별 오차 분포")
    print(f"{'regime':>9} {'L':>2} {'n':>5} {'med|e|':>8} {'p90':>7} {'p95':>7} {'max':>7} {'bias':>8}")
    sigma = {}
    for (reg, L) in sorted(samp.keys()):
        rs = samp[(reg, L)]
        mx = np.array([r["max_abs"] for r in rs])
        me = np.median([r["med_abs"] for r in rs])
        b  = np.nanmean([r["bias"] for r in rs])
        p95 = float(np.percentile(mx, 95))
        sigma[(reg, L)] = p95
        print(f"{reg:>9} {L:>2} {len(rs):>5} {me:>8.2f} {np.percentile(mx,90):>7.2f} "
              f"{p95:>7.2f} {mx.max():>7.2f} {b:>+8.2f}")
    print("  ※ sliding window 라 표본 독립 아님 (겹침). p95 를 σ 로 채택.")
    print("  ※ near knee 자체 jitter (~1.3°) 포함 → 보간오차 + 측정노이즈 합, 보수적 (과대).")
    print("=" * 82)
    return sigma


# =============================================================
# 2. far infill
# =============================================================
def main(basis, gate):
    for label, p in [("basis", basis), ("gate", gate)]:
        if not Path(p).exists():
            raise SystemExit(f"[ERROR] {label} not found: {p}")

    d = np.load(basis)
    o = np.argsort(d["fid"])
    fid    = d["fid"][o].astype(int)
    knee_n = d["knee_near"][o].astype(float)
    knee_f = d["knee_far"][o].astype(float)
    npts   = d["thigh_far_npts"][o].astype(float)
    N_low  = float(np.load(gate)["N_low"])
    n = len(fid)

    sigma = calibrate(fid, knee_n)

    # --- conf 스케일: N_full = 2-means 상위 centroid (재계산, 하드코딩 X) ---
    cent = kmeans_1d(np.log10(np.clip(npts, 1, None)))
    N_full = float(10 ** cent[1])
    print(f"[conf] N_low={N_low:,.0f}  N_full (2-means upper centroid)={N_full:,.0f}")
    obs = np.clip((npts - N_low) / (N_full - N_low), 0.0, 1.0)

    collapsed = npts < N_low
    runs = runs_of_true(collapsed)

    knee_out = knee_f.copy()
    label = np.array(["MEASURED"] * n, dtype=object)
    source = np.array(["MEASURED"] * n, dtype=object)
    conf = obs.copy()
    sig = np.zeros(n)

    print("-" * 82)
    print(f"{'fid':>5} {'L':>2} {'Δ':>7} {'regime':>9} {'label':>24} {'σ':>6}")
    rec = []
    for (i, j) in runs:
        L = j - i
        a0, a1 = i - 1, j
        if a0 - TAN_K + 1 < 0 or a1 + TAN_K - 1 >= n:
            print(f"{fid[i]:>5} {L:>2}  [skip] anchor 부족 — 시퀀스 경계")
            continue
        reg, vals, ext, delta, vb, va = branch(knee_f, a0, a1, L, fid)
        if DELTA_LIN <= delta <= DELTA_CUB:
            print(f"  [!] fid {fid[i]}: Δ={delta:.1f}° 가 모호대 ({DELTA_LIN}~{DELTA_CUB}) "
                  f"→ 보수적으로 CUB_PEAK 처리")
        lab = "INTERPOLATED_HIGH_CURV" if reg == "CUB_PEAK" else "INTERPOLATED"
        src = {"LIN": "LINEAR", "CUB_OK": "CUBIC", "CUB_PEAK": "CUBIC_PEAK"}[reg]
        s = sigma.get((reg, L), np.nan)

        knee_out[i:j] = vals
        label[i:j] = lab
        source[i:j] = src
        sig[i:j] = s
        # conf: anchor 관측도를 물려받고 gap 깊이로 감쇠
        ca = min(obs[a0], obs[a1])
        for k in range(1, L + 1):
            dnear = min(k, L + 1 - k)
            conf[i + k - 1] = ca / dnear

        print(f"{fid[i]:>5} {L:>2} {delta:>7.1f} {reg:>9} {lab:>24} {s:>6.2f}")
        rec.append(dict(start_fid=int(fid[i]), L=L, delta=delta, regime=reg,
                        v_before=vb, v_after=va,
                        cubic_ext=(float(ext) if ext is not None else np.nan),
                        exit_measured=float(knee_f[a1]), sigma=s))

    # --- 회계 ---
    print("-" * 82)
    cnt = Counter(label.tolist())
    print("[프레임 회계]")
    for k in ["MEASURED", "INTERPOLATED", "INTERPOLATED_HIGH_CURV"]:
        c = cnt.get(k, 0)
        print(f"  {k:>24}: {c:>4} ({c/n:.1%})")
    print(f"  {'합':>24}: {sum(cnt.values()):>4}   (self-check: =200 이어야)")
    print(f"  {'붕괴 프레임':>24}: {int(collapsed.sum()):>4}   (self-check: =32 이어야)")

    hc = label == "INTERPOLATED_HIGH_CURV"
    ip = label == "INTERPOLATED"
    contrib = np.sqrt(np.mean(np.where(hc | ip, sig, 0.0) ** 2))
    print(f"[예산] 보간 프레임의 RMSE 기여 = √(mean σ²) = {contrib:.2f}°  (목표 <3°)")
    print("  ※ measured 프레임의 측정노이즈는 별도 항 — 여기 σ_interp 에 미포함 (Phase 6 소관).")
    print("=" * 82)

    # --- 저장 ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(str(OUT_DIR / "far_knee_final.npz"),
             fid=fid, knee_near=knee_n, knee_far_final=knee_out,
             knee_far_raw=knee_f, label=label, conf=conf, sigma_interp=sig,
             source=source, thigh_far_npts=npts, N_low=N_low, N_full=N_full)
    with open(OUT_DIR / "far_knee_final.csv", "w") as f:
        f.write("fid,knee_near,knee_far_final,label,conf,sigma_interp,source,thigh_far_npts\n")
        for k in range(n):
            f.write(f"{fid[k]},{knee_n[k]:.3f},{knee_out[k]:.3f},{label[k]},"
                    f"{conf[k]:.3f},{sig[k]:.3f},{source[k]},{npts[k]:.0f}\n")

    plot(fid, knee_f, knee_out, label, conf, sig, collapsed, sigma,
         OUT_DIR / "far_knee_final.png")
    print(f"[saved] far_knee_final.npz / .csv / .png → {OUT_DIR}")


def plot(fid, raw, out, label, conf, sig, collapsed, sigma, path):
    col = {"MEASURED": "tab:blue", "INTERPOLATED": "tab:green",
           "INTERPOLATED_HIGH_CURV": "tab:red"}
    fig, ax = plt.subplots(3, 1, figsize=(14, 11))

    ax[0].plot(fid[collapsed], raw[collapsed], "x", color="0.8", ms=6,
               label="discarded collapsed fit (prior-driven)")
    ax[0].plot(fid, out, "-", color="0.5", lw=0.8, zorder=1)
    for lb in ["MEASURED", "INTERPOLATED", "INTERPOLATED_HIGH_CURV"]:
        m = label == lb
        if not m.any():
            continue
        ax[0].errorbar(fid[m], out[m], yerr=sig[m], fmt="none", ecolor=col[lb],
                       alpha=0.5, capsize=2)
        ax[0].scatter(fid[m], out[m], s=15 + 60 * conf[m], c=col[lb],
                      edgecolor="k", linewidth=0.3, zorder=3, label=lb)
    ax[0].set_ylabel("far knee [deg]")
    ax[0].set_title("far knee final — color=label, size ∝ conf, errorbar=σ")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].grid(alpha=0.3)

    ax[1].plot(fid, conf, "-", color="tab:purple", lw=1.2)
    ax[1].fill_between(fid, 0, conf, color="tab:purple", alpha=0.15)
    ax[1].set_ylabel("conf")
    ax[1].set_xlabel("fid")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("conf (observation-based — npts + anchor distance. NOT value quality)")
    ax[1].grid(alpha=0.3)

    ks = sorted(sigma.keys())
    ax[2].bar(range(len(ks)), [sigma[k] for k in ks],
              color=["tab:green" if k[0] != "CUB_PEAK" else "tab:red" for k in ks])
    ax[2].set_xticks(range(len(ks)))
    ax[2].set_xticklabels([f"{r}\nL{L}" for (r, L) in ks], fontsize=8)
    ax[2].axhline(3.0, color="k", ls=":", label="3° budget")
    ax[2].set_ylabel("σ (p95) [deg]")
    ax[2].legend(fontsize=8)
    ax[2].set_title("calibration σ — near synthetic gap, regime × L")
    ax[2].grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default=str(BASIS_DEFAULT))
    ap.add_argument("--gate",  default=str(GATE_DEFAULT))
    a = ap.parse_args()
    main(a.basis, a.gate)
