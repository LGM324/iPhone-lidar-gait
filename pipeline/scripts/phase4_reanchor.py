#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase4_reanchor.py
==================
ANCHOR_ARTIFACT 수정 — anchor slope a 의 프레임별 flicker로 생긴 "압축(extent 붕괴)"을,
pristine s_relative 에서 global_a 로 재-anchor 하여 시간축 extent 일관성을 회복한다.
z_med(centroid)는 정확히 보존 -> 앞서 검증한 z_med 안정성(0.61cm) 그대로 유지.

[원리·논문 근거]
  - 직선 z=a·s+b 는 centroid (s̄,z̄) 를 지남 => z_med≈z̄ 는 slope 무관, extent=a·(s-s̄)는 a 비례.
    압축 = 저-a 프레임. corr(두께,a)=0.867 로 실증됨.
  - 수정: z_new = global_a·s_relative + b_new,  b_new = z_med - global_a·s̄
    -> median(z_new)=z_med 정확 보존(centroid 성질), extent 는 global_a 로 통일.
    -> pristine s_relative 사용이라 노이즈 증폭 없음(최종 zmetric rescaling 대비 우월).
  - 근거: PromptDA(Lin 2026) — iPhone LiDAR temporal flicker -> anchor a 출렁(수정 대상 한계).
          RANSAC+MAD 정합(C.5)로 얻은 centroid 는 robust하므로 보존, 노이즈 많은 slope만 교체.
  - global_a 절대값: 본 자료만으론 유일 확정 불가(저-a=적은앵커/degenerate, 고-a=소표본).
    시간축 일관성은 보장(=SMPL fitting 필수), 절대 스케일은 SMPL β가 흡수 + 정적 SMPL로 교차검증(A0).
  - tail 제거(grazing 75°)+Poisson refine 은 검증 노트북 코드 그대로(스케일 바뀌었으니 재실행).

[사용]  python phase4_reanchor.py --dry     # 후보 표 + before/after만 출력, 파일 안 씀
        python phase4_reanchor.py --apply   # 재-anchor+tail+refine -> 새 npy 저장(.bak 백업)
[입력]  VIDEO_DIR/*.npz (s_relative, mask_crop, normal, a_raw, zmetric_crop, ibox)
        OLD_METRIC_DIR/*_person_metric.npy (H,W 기준 + 백업 대상)
        INTRINSICS_JSON (RGB 기준 fx,fy,cx,cy)
[출력]  OUT_METRIC_DIR/*_person_metric.npy (재-anchor) + phase4_report.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import walkcfg  # [colab_kit] 경로/세션 설정
import os, re, glob, json, math, argparse, shutil
import numpy as np
import cv2
import torch

# GPU 사용 가능 시 자동. refine 만 cuda 로 오프로드 (알고리즘 identical).
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 경로 (Claude Code가 여기만 맞추면 됨) ----
BASE = walkcfg.DEPTH_BASE  # [colab_kit]
VIDEO_DIR       = os.path.join(BASE, "work", "video")
OLD_METRIC_DIR  = os.path.join(BASE, "work", "out")
OUT_METRIC_DIR  = os.path.join(BASE, "work", "out")          # 같은 폴더에 덮되 .bak 백업
INTRINSICS_JSON = os.path.join(BASE, "inputs", "intrinsics.json")
OUT_REPORT      = os.path.join(BASE, "work", "analysis", "phase4_report.json")
KEY_PREFIX = "WALK_side_"

# ---- 재-anchor / refine 파라미터 (검증본, 변경 금지) ----
GLOBAL_A_METHOD = "manual"      # {"median","trimmed","p75","manual"}
GLOBAL_A_MANUAL = walkcfg.global_a()   # 세션별 extract_pool_fit 결과 (4/20=0.486)  # [colab_kit]
TRIM_FRAC       = 0.25          # trimmed-mean 양끝 절단
TORSO_BAND      = (0.15, 0.45)
GRAZE_DEG, DISC_REL, ERODE_PX = 75, 0.03, 2   # tail 제거(검증본)
REFINE_LAM, REFINE_ITERS      = 0.1, 800       # Poisson refine(검증본)
DO_REFINE = True                 # False면 tail 제거까지만


# ============================================================
# 유틸 + 검증 노트북 이식 함수 (clean_flying / refine / to_full)
# ============================================================
def load_intr():
    with open(INTRINSICS_JSON, encoding="utf-8") as f:
        return json.load(f)

def get_intr(intr, key):
    d = intr.get(key, intr.get("_default"))
    return float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"])

def clean_flying(z, mask, fx, fy, cx, cy, rel=DISC_REL, erode_px=ERODE_PX,
                 normal=None, graze_deg=GRAZE_DEG):
    M = mask & (z > 0); keep = M.copy()
    if erode_px > 0:
        k = 2 * erode_px + 1
        keep &= cv2.erode(M.astype(np.uint8), np.ones((k, k), np.uint8), 1).astype(bool)
    jump = np.zeros_like(z)
    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        sh = np.roll(np.roll(z, dy, 0), dx, 1)
        vd = np.roll(np.roll(M, dy, 0), dx, 1)
        jump = np.maximum(jump, np.where(vd, np.abs(z - sh), 0))
    keep &= jump < rel * np.clip(z, 1e-3, None)
    if normal is not None:
        h, w = z.shape
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        r = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(z)], -1)
        r /= np.linalg.norm(r, axis=-1, keepdims=True)
        keep &= np.abs((normal * r).sum(-1)) > np.cos(np.deg2rad(graze_deg))
    z2 = z.copy(); z2[~keep] = 0.0
    return keep, z2

def refine(z, n, mask, fx, fy, cx, cy, lam=REFINE_LAM, iters=REFINE_ITERS):
    """GPU 오프로드 버전. 알고리즘 identical (numpy 원본과 numerical 동일)."""
    h, w = z.shape
    # cv2.erode 는 CPU (fast, 한 번만)
    M_np = mask & (z > 0)
    er_np = cv2.erode(M_np.astype(np.uint8), np.ones((5, 5), np.uint8), 1).astype(bool)

    dev = _DEVICE
    # torch tensor 로 이동 (fp32)
    z_t = torch.from_numpy(z).float().to(dev)
    n_t = torch.from_numpy(n).float().to(dev)
    M = torch.from_numpy(M_np).to(dev)                  # bool
    er = torch.from_numpy(er_np).to(dev)                # bool
    mask_t = torch.from_numpy(mask).to(dev)             # bool

    uu, vv = torch.meshgrid(
        torch.arange(w, device=dev, dtype=torch.float32),
        torch.arange(h, device=dev, dtype=torch.float32),
        indexing="xy")
    a = (uu - cx) / fx; c = (vv - cy) / fy
    nx, ny, nz = n_t[..., 0], n_t[..., 1], n_t[..., 2]
    D = nx * a + ny * c + nz
    D = torch.where(D.abs() < 1e-3, torch.sign(D) * 1e-3 + 1e-9, D)
    Zu = -z_t * nx / (fx * D); Zv = -z_t * ny / (fy * D)
    reliable = nz.abs() > 0.3

    wd = torch.where(M & ~er, torch.tensor(lam * 6, device=dev),
                     torch.tensor(lam, device=dev))
    z0 = z_t.clone(); d = z_t.clone()

    zero_scalar = torch.zeros((), device=dev)

    for _ in range(iters):
        num = torch.zeros_like(d); den = torch.zeros_like(d)
        # horizontal neighbors
        lk = M[:, :-1] & M[:, 1:] & reliable[:, :-1]
        lk_f = lk.float()
        num[:, :-1] += torch.where(lk, d[:, 1:] - Zu[:, :-1], zero_scalar)
        den[:, :-1] += lk_f
        num[:, 1:]  += torch.where(lk, d[:, :-1] + Zu[:, :-1], zero_scalar)
        den[:, 1:]  += lk_f
        # vertical neighbors
        lv = M[:-1, :] & M[1:, :] & reliable[:-1, :]
        lv_f = lv.float()
        num[:-1, :] += torch.where(lv, d[1:, :] - Zv[:-1, :], zero_scalar)
        den[:-1, :] += lv_f
        num[1:, :]  += torch.where(lv, d[:-1, :] + Zv[:-1, :], zero_scalar)
        den[1:, :]  += lv_f
        d = torch.where(M, (num + wd * z0) / (den + wd + 1e-9), z0)

    d = torch.where(mask_t, d, zero_scalar)
    return d.detach().cpu().numpy().astype(np.float32)

def to_full(zc, ib, H, W):
    x0, y0, x1, y1 = ib
    full = np.zeros((H, W), np.float32)
    sx0, sy0 = max(0, x0), max(0, y0); sx1, sy1 = min(W, x1), min(H, y1)
    dx, dy = sx0 - x0, sy0 - y0
    if sx1 > sx0 and sy1 > sy0:
        full[sy0:sy1, sx0:sx1] = zc[dy:dy + (sy1 - sy0), dx:dx + (sx1 - sx0)]
    return full

def torso_extent(z, m):
    v, u = np.where(m & (z > 0) & np.isfinite(z))
    if len(v) < 50:
        return np.nan
    top, bot = v.min(), v.max()
    sel = (v >= top + TORSO_BAND[0] * (bot - top)) & (v <= top + TORSO_BAND[1] * (bot - top))
    if sel.sum() < 20:
        return np.nan
    zt = z[v[sel], u[sel]]
    return float(np.percentile(zt, 95) - np.percentile(zt, 5))

def torso_srange(s, m):
    v, u = np.where(m)
    if len(v) < 50:
        return np.nan
    top, bot = v.min(), v.max()
    sel = (v >= top + TORSO_BAND[0] * (bot - top)) & (v <= top + TORSO_BAND[1] * (bot - top))
    if sel.sum() < 20:
        return np.nan
    st = s[v[sel], u[sel]]
    return float(np.percentile(st, 95) - np.percentile(st, 5))


# ============================================================
def collect():
    rows = []
    for fp in glob.glob(os.path.join(VIDEO_DIR, f"{KEY_PREFIX}*.npz")):
        m = re.search(rf"{re.escape(KEY_PREFIX)}(\d+)\.npz$", os.path.basename(fp))
        if m:
            rows.append((int(m.group(1)), f"{KEY_PREFIX}{m.group(1)}", fp))
    rows.sort(key=lambda r: r[0])
    if not rows:
        raise FileNotFoundError(f"npz 없음: {VIDEO_DIR}")
    return rows

def choose_global_a(a_raw, method):
    a = a_raw[np.isfinite(a_raw)]
    cands = dict(
        median=float(np.median(a)),
        trimmed=float(np.mean(np.sort(a)[int(len(a)*TRIM_FRAC):len(a)-int(len(a)*TRIM_FRAC)])),
        p75=float(np.percentile(a, 75)),
    )
    if method == "manual" and GLOBAL_A_MANUAL is not None:
        cands["manual"] = float(GLOBAL_A_MANUAL)
    return cands.get(method, cands["median"]), cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    dry = not args.apply
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    os.makedirs(OUT_METRIC_DIR, exist_ok=True)
    intr = load_intr()
    rows = collect()

    # 1패스: a_raw 수집 + s_range_torso 중앙값 (예상 extent 계산용)
    a_raw, sr_torso = [], []
    for fid, key, fp in rows:
        d = np.load(fp, allow_pickle=True)
        a_raw.append(float(d["a_raw"]) if "a_raw" in d else np.nan)
        if "s_relative" in d and "mask_crop" in d:
            sr_torso.append(torso_srange(np.array(d["s_relative"], float),
                                         np.array(d["mask_crop"], bool)))
        else:
            sr_torso.append(np.nan)
    a_raw = np.array(a_raw); sr_med = float(np.nanmedian(sr_torso))
    global_a, cands = choose_global_a(a_raw, GLOBAL_A_METHOD)

    print("=" * 68)
    print(f"[a_raw] median={np.nanmedian(a_raw):.3f} std={np.nanstd(a_raw):.3f} "
          f"min={np.nanmin(a_raw):.3f} max={np.nanmax(a_raw):.3f}")
    print(f"[torso s-range median] {sr_med:.3f}  (예상 두께 = a × 이 값)")
    print("-" * 68)
    print(f"{'global_a 후보':16s}{'a':>8s}{'예상 몸통두께':>16s}")
    for k, v in cands.items():
        mark = " <== 선택" if abs(v - global_a) < 1e-9 else ""
        print(f"{k:16s}{v:>8.3f}{v*sr_med*100:>14.2f}cm{mark}")
    print("-" * 68)
    print("정적 SMPL β 의 몸통 두께와 대조해 후보 선택 권장(절대스케일 교차검증, A0).")
    print(f"[MODE] {'DRY(파일 안 씀)' if dry else 'APPLY(재-anchor+tail+refine 저장)'}  DO_REFINE={DO_REFINE}")
    print("=" * 68)

    # 2패스: 프레임별 재-anchor
    report = []
    for fid, key, fp in rows:
        d = np.load(fp, allow_pickle=True)
        if not all(k in d for k in ("s_relative", "mask_crop", "zmetric_crop", "ibox")):
            print(f"{key}: 필수키 없음 skip"); continue
        s = np.array(d["s_relative"], float)
        m = np.array(d["mask_crop"], bool)
        z_old = np.array(d["zmetric_crop"], float)
        ib = tuple(int(v) for v in np.array(d["ibox"]))
        nrm = np.array(d["normal"], float) if "normal" in d else None

        zmask = z_old[m & (z_old > 0)]
        if len(zmask) < 50:
            print(f"{key}: 유효 depth 부족 skip"); continue
        z_med = float(np.median(zmask))
        s_med = float(np.median(s[m]))
        b_new = z_med - global_a * s_med

        z_new = (global_a * s + b_new).astype(np.float32)
        z_new[~m] = 0.0

        ext_before = torso_extent(z_old, m)
        ext_raw_after = torso_extent(z_new, m)   # tail/refine 전

        old_fp = os.path.join(OLD_METRIC_DIR, f"{key}_person_metric.npy")
        H, W = (np.load(old_fp).shape if os.path.exists(old_fp) else (1920, 1440))
        fx, fy, cx, cy = get_intr(intr, key)
        x0, y0 = ib[0], ib[1]; cxc, cyc = cx - x0, cy - y0

        if not dry:
            keep, z_clean = clean_flying(z_new, m, fx, fy, cxc, cyc, normal=nrm)
            z_ref = refine(z_clean, nrm, keep, fx, fy, cxc, cyc) if (DO_REFINE and nrm is not None) else z_clean
            ext_final = torso_extent(z_ref, keep)
            zfull = to_full(z_ref, ib, H, W)
            z_med_final = float(np.median(zfull[zfull > 0]))
            if os.path.exists(old_fp) and not os.path.exists(old_fp + ".bak"):
                shutil.copy2(old_fp, old_fp + ".bak")
            np.save(os.path.join(OUT_METRIC_DIR, f"{key}_person_metric.npy"), zfull)
        else:
            ext_final = ext_raw_after; z_med_final = z_med

        report.append(dict(key=key, a_old=float(d.get("a_raw", np.nan)), global_a=global_a,
                           z_med_before=z_med, z_med_after=z_med_final,
                           torso_before_cm=ext_before*100 if np.isfinite(ext_before) else None,
                           torso_after_cm=ext_final*100 if np.isfinite(ext_final) else None))

    # 요약
    tb = np.array([r["torso_before_cm"] for r in report if r["torso_before_cm"]], float)
    ta = np.array([r["torso_after_cm"] for r in report if r["torso_after_cm"]], float)
    zb = np.array([r["z_med_before"] for r in report], float)
    za = np.array([r["z_med_after"] for r in report], float)
    summary = dict(n=len(report), global_a=global_a, method=GLOBAL_A_METHOD, dry=dry,
                   torso_before=dict(median=float(np.median(tb)), std=float(np.std(tb)),
                                     min=float(tb.min()), max=float(tb.max())),
                   torso_after=dict(median=float(np.median(ta)), std=float(np.std(ta)),
                                    min=float(ta.min()), max=float(ta.max())),
                   z_med_preserved_max_drift_cm=float(np.max(np.abs(za - zb)) * 100))
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(dict(summary=summary, per_frame=report), f, indent=2, ensure_ascii=False)

    print("-" * 68)
    print(f"[torso 두께] before: med={np.median(tb):.2f} std={np.std(tb):.2f} "
          f"[{tb.min():.2f},{tb.max():.2f}]cm")
    print(f"[torso 두께] after : med={np.median(ta):.2f} std={np.std(ta):.2f} "
          f"[{ta.min():.2f},{ta.max():.2f}]cm   (std 급감 = flicker 제거 확인)")
    print(f"[z_med 보존] 최대 drift = {np.max(np.abs(za-zb))*100:.4f} cm  (0에 가까워야 정상)")
    print(f"[report] {OUT_REPORT}")
    print("=" * 68)


if __name__ == "__main__":
    main()
