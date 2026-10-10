#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
far_thigh_infill_v2.py

far 무릎각 최종 시계열. v1 대비 **값은 동일**, σ 만 Δ 조건화로 재계산.

[v1 문제] σ = (regime, L) 별로 near 의 Δ>12 전 구간 sliding window pool → p95.
  near 급경사는 8~10°/frame → L=4 window 의 Δ 가 40~50° 까지 감.
  far 실제 run 은 Δ 12.1~20.3° → far 가 겪지 않는 어려운 window 가 p95 를 부풀림.
  증거: far 기하 일치 게이트 p95 2.64 vs v1 CUB_OK L4 = 5.49.
  방증: CUB_PEAK (정점, 완만) σ < CUB_OK (급경사 중간) σ  (L3: 2.97 vs 4.89).

[v2 수정] far run 마다 같은 (regime, L) 이면서 **Δ 최근접 K 개** window 로 σ 산출.
  Δ 는 anchor 에서 나오는 측정량 (분기에도 이미 사용) → 순환 없음.
  단, σ(Δ) 가 Δ 에 단조 증가하는지 먼저 확인 — 아니면 Δ 만으론 부족하다는 뜻.

배치: _pipeline_consolidated/D_walking_analysis/[Script]D_experiments/
산출: [Result]D_experiments/far_knee_final.{npz,csv} (v1 덮어씀, 값 동일)
      [Result]D_experiments/far_knee_final_v2.png (신규)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import walkcfg  # [colab_kit] 경로/세션 설정
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows cp949 로 특수문자 크래시 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---- 경로 (consolidated) ----
RESULT_DIR = Path(walkcfg.KNEE_DIR)  # [colab_kit]
BASIS_DEFAULT = RESULT_DIR / "knee_basis_full.npz"
GATE_DEFAULT  = RESULT_DIR / "far_thigh_gate_full.npz"
PREV_DEFAULT  = RESULT_DIR / "far_knee_final.npz"
OUT_DIR = RESULT_DIR

DELTA_LIN = 3.0
DELTA_CUB = 12.0
EPS_V     = 0.3
TAN_DEG, TAN_K = 2, 4      # quad4
LENGTHS   = [2, 3, 4]
K_MATCH   = 40             # Δ 최근접 표본 수
GAIT_T    = walkcfg.GAIT_FRAMES   # 4/20=41.5  # [colab_kit]


# --------------------------- 공통 함수 (v1 과 동일) ---------------------------
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
        reg = "CUB_PEAK"
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
# 1. near 캘리브레이션 표본 수집 (조건 라벨 포함)
# =============================================================
def build_calib(fid, knee_near):
    n = len(fid)
    S = []
    for L in LENGTHS:
        for i in range(TAN_K, n - L - TAN_K):
            a0, a1 = i - 1, i + L
            reg, vals, ext, delta, vb, va = branch(knee_near, a0, a1, L, fid)
            truth = knee_near[i:i + L]
            err = vals - truth
            true_max = float(knee_near[a0:a1 + 1].max())
            S.append(dict(regime=reg, L=L, delta=delta, pos=int(fid[i]),
                          max_abs=float(np.max(np.abs(err))),
                          med_abs=float(np.median(np.abs(err))),
                          bias=(float(ext) - true_max) if ext is not None else np.nan))
    return S


# =============================================================
# 2. 진단 — σ(Δ) 가 Δ 에 단조인가?
# =============================================================
def sigma_vs_delta(S, far_deltas):
    print("=" * 88)
    print("[진단] σ(Δ) — Δ 가 오차를 설명하는가?  (단조 증가여야 Δ 매칭이 유효)")
    ok_all = True
    for reg in ["LIN", "CUB_OK", "CUB_PEAK"]:
        rs = [s for s in S if s["regime"] == reg]
        if len(rs) < 12:
            print(f"  {reg}: 표본 {len(rs)} 개 — bin 진단 생략")
            continue
        dv = np.array([s["delta"] for s in rs])
        mx = np.array([s["max_abs"] for s in rs])
        qs = np.quantile(dv, np.linspace(0, 1, 6))     # 5 bin
        print(f"  --- {reg} (n={len(rs)}) ---")
        print(f"      {'Δ bin':>16} {'n':>5} {'med|e|':>8} {'p90':>7} {'p95':>7}")
        p95s = []
        for b in range(5):
            m = (dv >= qs[b]) & (dv <= qs[b + 1] if b == 4 else dv < qs[b + 1])
            if m.sum() < 3:
                continue
            p95 = float(np.percentile(mx[m], 95))
            p95s.append(p95)
            print(f"      {qs[b]:6.1f}~{qs[b+1]:6.1f}° {m.sum():>5} "
                  f"{np.median(mx[m]):>8.2f} {np.percentile(mx[m],90):>7.2f} {p95:>7.2f}")
        if len(p95s) >= 3:
            mono = all(p95s[k] <= p95s[k + 1] + 0.5 for k in range(len(p95s) - 1))
            print(f"      → 단조 증가: {'OK' if mono else '**아님**'}")
            ok_all &= mono
    print(f"  far run 들의 Δ = {np.round(sorted(far_deltas),1).tolist()}")
    if not ok_all:
        print("  [!] 일부 regime 에서 σ(Δ) 가 단조 아님 → Δ 만으론 오차를 다 설명 못 함.")
        print("      Δ 매칭 결과를 보수적으로 볼 것 (곡률 등 다른 조건이 필요할 수 있음).")
    print("=" * 88)
    return ok_all


# =============================================================
# 3. Δ 최근접 매칭으로 σ 산출
# =============================================================
def sigma_matched(S, regime, L, delta, K=K_MATCH):
    cand = [s for s in S if s["regime"] == regime and s["L"] == L]
    pooled = False
    if len(cand) < 5:
        cand = [s for s in S if s["regime"] == regime and abs(s["L"] - L) <= 1]
        pooled = True
    if not cand:
        return np.nan, 0, (np.nan, np.nan), 0, True
    cand = sorted(cand, key=lambda s: abs(s["delta"] - delta))[:min(K, len(cand))]
    mx = np.array([s["max_abs"] for s in cand])
    dsp = (min(s["delta"] for s in cand), max(s["delta"] for s in cand))
    ncyc = len({int(s["pos"] // GAIT_T) for s in cand})   # 서로 다른 gait 주기 수
    return float(np.percentile(mx, 95)), len(cand), dsp, ncyc, pooled


# =============================================================
def main(basis, gate, prev):
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

    S = build_calib(fid, knee_n)

    cent = kmeans_1d(np.log10(np.clip(npts, 1, None)))
    N_full = float(10 ** cent[1])
    obs = np.clip((npts - N_low) / (N_full - N_low), 0.0, 1.0)
    collapsed = npts < N_low
    runs = runs_of_true(collapsed)

    # far run 조건 먼저 수집 (진단용)
    pre = []
    for (i, j) in runs:
        L = j - i
        a0, a1 = i - 1, j
        if a0 - TAN_K + 1 < 0 or a1 + TAN_K - 1 >= n:
            continue
        reg, vals, ext, delta, vb, va = branch(knee_f, a0, a1, L, fid)
        pre.append((i, j, L, reg, vals, delta))
    sigma_vs_delta(S, [p[5] for p in pre])

    knee_out = knee_f.copy()
    label  = np.array(["MEASURED"] * n, dtype=object)
    source = np.array(["MEASURED"] * n, dtype=object)
    conf   = obs.copy()
    sig    = np.zeros(n)

    print(f"{'fid':>5} {'L':>2} {'Δ':>6} {'regime':>9} {'K':>3} {'Δ range(K)':>16} "
          f"{'ncyc':>5} {'σ_new':>6}")
    rec = []
    for (i, j, L, reg, vals, delta) in pre:
        s_new, K, dsp, ncyc, pooled = sigma_matched(S, reg, L, delta)
        lab = "INTERPOLATED_HIGH_CURV" if reg == "CUB_PEAK" else "INTERPOLATED"
        src = {"LIN": "LINEAR", "CUB_OK": "CUBIC", "CUB_PEAK": "CUBIC_PEAK"}[reg]

        knee_out[i:j] = vals
        label[i:j], source[i:j], sig[i:j] = lab, src, s_new
        ca = min(obs[i - 1], obs[j])
        for k in range(1, L + 1):
            conf[i + k - 1] = ca / min(k, L + 1 - k)

        warn = ""
        if pooled:
            warn += " [L pooled]"
        if ncyc <= 1:
            warn += " [!same-cycle only — samples not independent]"
        print(f"{fid[i]:>5} {L:>2} {delta:>6.1f} {reg:>9} {K:>3} "
              f"{dsp[0]:>6.1f}~{dsp[1]:>6.1f}° {ncyc:>5} {s_new:>6.2f}{warn}")
        rec.append(dict(start_fid=int(fid[i]), L=L, delta=delta, regime=reg,
                        sigma=s_new, K=K, d_lo=dsp[0], d_hi=dsp[1], n_cycles=ncyc))

    # --- self-check: 값 불변 ---
    print("-" * 88)
    if prev and Path(prev).exists():
        pv = np.load(prev, allow_pickle=True)
        try:
            same = np.allclose(pv["knee_far_final"], knee_out, atol=1e-6)
        except Exception:
            same = False
        print(f"[self-check] knee_far_final v1 과 동일: {'OK' if same else '**불일치 — 값이 바뀜!**'}")
        if not same:
            print("             σ 만 바뀌어야 하는데 값이 변함 → 분기 로직 확인 필요.")

    cnt = Counter(label.tolist())
    print("[프레임 회계]")
    for k in ["MEASURED", "INTERPOLATED", "INTERPOLATED_HIGH_CURV"]:
        c = cnt.get(k, 0)
        print(f"  {k:>24}: {c:>4} ({c/n:.1%})")
    print(f"  {'합':>24}: {sum(cnt.values()):>4}  (=200 이어야)   붕괴 {int(collapsed.sum())} (=32)")
    print(f"[예산] RMSE 기여 √(mean σ² over 200f) = {np.sqrt(np.mean(sig**2)):.2f}°  (목표 <3°)")
    print("  ※ measured 프레임 측정노이즈는 미포함 (Phase 6 소관).")
    print("=" * 88)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(str(OUT_DIR / "far_knee_final.npz"),
             fid=fid, knee_near=knee_n, knee_far_final=knee_out, knee_far_raw=knee_f,
             label=label, conf=conf, sigma_interp=sig, source=source,
             thigh_far_npts=npts, N_low=N_low, N_full=N_full)
    with open(OUT_DIR / "far_knee_final.csv", "w", encoding="utf-8") as f:
        f.write("# conf 는 far 자체 스케일 (N_full=%.0f) 정규화 — **다리 간 비교 불가**.\n" % N_full)
        f.write("# OpenSim cross-leg 가중치로 쓰지 말 것. 같은 다리 안 프레임 비교만.\n")
        f.write("# sigma_interp = 보간 불확실도 (near 인공 gap, Δ 매칭 캘리브레이션). "
                "measured 프레임의 측정노이즈 미포함.\n")
        f.write("fid,knee_near,knee_far_final,label,conf,sigma_interp,source,thigh_far_npts\n")
        for k in range(n):
            f.write(f"{fid[k]},{knee_n[k]:.3f},{knee_out[k]:.3f},{label[k]},"
                    f"{conf[k]:.3f},{sig[k]:.3f},{source[k]},{npts[k]:.0f}\n")

    plot(fid, knee_f, knee_out, label, conf, sig, collapsed, S, rec,
         OUT_DIR / "far_knee_final_v2.png")
    print(f"[saved] far_knee_final.npz / .csv / far_knee_final_v2.png → {OUT_DIR}")


def plot(fid, raw, out, label, conf, sig, collapsed, S, rec, path):
    col = {"MEASURED": "tab:blue", "INTERPOLATED": "tab:green",
           "INTERPOLATED_HIGH_CURV": "tab:red"}
    rcol = {"LIN": "tab:gray", "CUB_OK": "tab:green", "CUB_PEAK": "tab:red"}
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))

    # (1) key diagnostic: σ vs Δ
    for reg in ["LIN", "CUB_OK", "CUB_PEAK"]:
        rs = [s for s in S if s["regime"] == reg]
        if not rs:
            continue
        ax[0].scatter([s["delta"] for s in rs], [s["max_abs"] for s in rs],
                      s=8, alpha=0.35, color=rcol[reg], label=f"{reg} (near window)")
    for r in rec:
        ax[0].axvline(r["delta"], color=rcol[r["regime"]], ls="--", lw=1, alpha=0.8)
        ax[0].plot(r["delta"], r["sigma"], "*", color="k", ms=14, zorder=5)
    ax[0].plot([], [], "*", color="k", ms=10, label="far run Δ + adopted σ")
    ax[0].axhline(3.0, color="k", ls=":", lw=1)
    ax[0].set_xlabel("Δ = |knee difference between anchors| [deg]")
    ax[0].set_ylabel("max |reconstructed − truth| [deg]")
    ax[0].set_title("σ vs Δ — where do far runs (vertical lines) sit in near sample distribution? "
                    "(v1 pooled all Δ>12 → inflated)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # (2) final curve
    ax[1].plot(fid[collapsed], raw[collapsed], "x", color="0.8", ms=6,
               label="discarded collapsed fit")
    ax[1].plot(fid, out, "-", color="0.5", lw=0.8, zorder=1)
    for lb in ["MEASURED", "INTERPOLATED", "INTERPOLATED_HIGH_CURV"]:
        m = label == lb
        if not m.any():
            continue
        ax[1].errorbar(fid[m], out[m], yerr=sig[m], fmt="none", ecolor=col[lb],
                       alpha=0.5, capsize=2)
        ax[1].scatter(fid[m], out[m], s=15 + 60 * conf[m], c=col[lb],
                      edgecolor="k", linewidth=0.3, zorder=3, label=lb)
    ax[1].set_xlabel("fid")
    ax[1].set_ylabel("far knee [deg]")
    ax[1].set_title("far knee final (values same as v1, σ updated via Δ-matching)")
    ax[1].legend(fontsize=8, ncol=2)
    ax[1].grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default=str(BASIS_DEFAULT))
    ap.add_argument("--gate",  default=str(GATE_DEFAULT))
    ap.add_argument("--prev",  default=str(PREV_DEFAULT),
                    help="v1 산출 — 값 불변 self-check 용 (덮어쓰기 전에 읽음)")
    a = ap.parse_args()
    main(a.basis, a.gate, a.prev)
