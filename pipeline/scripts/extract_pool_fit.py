#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_pool_fit.py
===================
[선행연구 정석] 전역 상수 스케일 결정 — SMPL 매칭 폐기. LiDAR 점에 직접 적합.

방법론 근거:
  - RadarCam-Depth(Li 2024): mono-depth 를 range 센서에 전역 affine 정렬. 동적 시퀀스는
    프레임별(Var/LS) 대신 "Const"(전 프레임 공통 스케일) 사용 -> flicker 제거의 표준 해법.
  - PromptDA(Lin 2026) C.5: iPhone LiDAR 정합은 RANSAC + MAD 투표가 polyfit보다 robust(F +8~10%).
  - 단일뷰 human(HaP 2311.02892 / IF-Nets / OmniFit): 관측은 "보이는 한 겹 부분표면"으로 두고
    SMPL 을 거기 등록해 뒷면을 prior 로 채움. => depth 를 full 두께로 부풀리지 않는다(1.099 폐기).

이 스크립트가 하는 일:
  1) inputs 의 LiDAR depth/conf + 저장된 s_relative/mask/ibox 로 프레임별 anchor 점쌍 (s_i, z_i) 재추출
  2) [자기검증 게이트] 재추출 점으로 per-frame 재적합 a_i 를 저장된 a_raw 와 대조.
     -> 재현되면 좌표정합 OK(강한 증거). 아니면 폴백(robust a_raw = RadarCam Const 근사) + 경고.
  3) 검증 통과 시: 전 프레임 점 pool -> 단일 전역 a 를 RANSAC+MAD 적합 (RadarCam Const 정석)
  4) 전역 a 를 per-frame a_raw 분포/torso 예측과 대조 리포트. (적용은 phase4 GLOBAL_A_MANUAL 로)

[입력]  EXTRACT_DIR/depth_npy/{fid}.npy(256x192,m), conf_npy/{fid}.npy(256x192,uint8)
        VIDEO_DIR/WALK_side_{fid}.npz(s_relative,mask_crop,ibox,a_raw,b_raw)
        OLD_METRIC_DIR/WALK_side_{fid}_person_metric.npy(전체 H,W)
[출력]  work/analysis/global_const_fit.json  (global_a, per-frame b, 검증 리포트)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import walkcfg  # [colab_kit] 경로/세션 설정
import os, re, glob, json
import numpy as np
import cv2

BASE = walkcfg.DEPTH_BASE  # [colab_kit]
EXTRACT_DIR    = walkcfg.EXTRACT_DIR  # [colab_kit]
VIDEO_DIR      = os.path.join(BASE, "work", "video")
OLD_METRIC_DIR = os.path.join(BASE, "work", "out")
OUT_JSON       = os.path.join(BASE, "work", "analysis", "global_const_fit.json")
KEY_PREFIX = "WALK_side_"

MIN_ANCHOR   = 300      # conf==2 부족 시 conf>=1 완화 (파이프라인과 동일)
ERODE_K      = 5        # anchor mask erode (파이프라인과 동일)
ARAW_TOL     = 0.15     # per-frame 재현 허용 상대오차
GATE_RATE    = 0.70     # 이 비율 이상 프레임에서 a_raw 재현되면 좌표정합 OK
TORSO_BAND   = (0.15, 0.45)


def ransac_mad_fit(s, z, iters=300, seed=0):
    """z = a·s + b, RANSAC + MAD 투표 (PromptDA C.5 방식). 반환 a,b,inlier_frac,rel."""
    rng = np.random.default_rng(seed)
    n = len(s)
    if n < 10:
        A = np.stack([s, np.ones_like(s)], 1)
        a, b = np.linalg.lstsq(A, z, rcond=None)[0]
        return float(a), float(b), 1.0, float(np.median(np.abs(z-(a*s+b))/np.clip(z,1e-3,None)))
    best = None
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        if s[i] == s[j]:
            continue
        a = (z[i]-z[j])/(s[i]-s[j]); b = z[i]-a*s[i]
        r = np.abs(z-(a*s+b))
        thr = np.median(r) + 1.4826*np.median(np.abs(r-np.median(r)))  # median+MAD
        inl = r < max(thr, 1e-6)
        sc = inl.sum()
        if best is None or sc > best[0]:
            best = (sc, inl)
    inl = best[1]
    A = np.stack([s[inl], np.ones(inl.sum())], 1)
    a, b = np.linalg.lstsq(A, z[inl], rcond=None)[0]
    for _ in range(2):  # MAD 재가중 2회
        r = np.abs(z-(a*s+b)); thr = np.median(r)+1.4826*np.median(np.abs(r-np.median(r)))
        w = r < max(thr,1e-6)
        A = np.stack([s[w], np.ones(w.sum())], 1)
        a, b = np.linalg.lstsq(A, z[w], rcond=None)[0]
    rel = float(np.median(np.abs(z-(a*s+b))/np.clip(z,1e-3,None)))
    return float(a), float(b), float(w.mean()), rel


def upsample_crop(arr_lowres, full_hw, ib):
    """LiDAR 저해상 -> RGB 전체(nearest) -> ibox crop. 여러 방향 시도 후 shape 맞는 것 반환."""
    H, W = full_hw; x0, y0, x1, y1 = ib
    ch, cw = (y1-y0), (x1-x0)
    for a in (arr_lowres, arr_lowres.T, np.rot90(arr_lowres), np.rot90(arr_lowres,3)):
        up = cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        crop = up[y0:y1, x0:x1]
        if crop.shape == (ch, cw):
            return crop
    return None   # 정합 실패


def collect():
    rows = []
    for fp in glob.glob(os.path.join(VIDEO_DIR, f"{KEY_PREFIX}*.npz")):
        m = re.search(rf"{re.escape(KEY_PREFIX)}(\d+)\.npz$", os.path.basename(fp))
        if m:
            rows.append((m.group(1), f"{KEY_PREFIX}{m.group(1)}", fp))
    rows.sort(key=lambda r: int(r[0]))
    return rows


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    rows = collect()
    if not rows:
        raise FileNotFoundError(VIDEO_DIR)

    all_s, all_z = [], []       # pool
    all_sd, all_zd = [], []     # [colab_kit 2026-10-08] 프레임별 중앙값을 뺀 값 (fixed-effects)
    fr_s, fr_z = [], []         # 프레임 중앙값 (진단: 프레임 사이 관계)
    perframe = []               # 검증 + 진단
    a_raw_all = []
    n_ok = n_align_fail = 0
    for fid_str, key, fp in rows:
        d = np.load(fp, allow_pickle=True)
        if not all(k in d for k in ("s_relative","mask_crop","ibox","a_raw")):
            continue
        s = np.array(d["s_relative"], float); m = np.array(d["mask_crop"], bool)
        ib = tuple(int(v) for v in np.array(d["ibox"]))
        a_raw = float(d["a_raw"]); a_raw_all.append(a_raw)

        # [경로 수정] Walking\[Result]00_extract\{depth_npy,conf_npy}\{fid}.npy
        fid = key.replace(KEY_PREFIX, "")
        dep_fp = os.path.join(EXTRACT_DIR, "depth_npy", f"{fid}.npy")
        con_fp = os.path.join(EXTRACT_DIR, "conf_npy",  f"{fid}.npy")
        old_fp = os.path.join(OLD_METRIC_DIR, f"{key}_person_metric.npy")
        if not (os.path.exists(dep_fp) and os.path.exists(con_fp) and os.path.exists(old_fp)):
            perframe.append(dict(key=key, status="missing_inputs")); continue
        H, W = np.load(old_fp).shape
        depL = np.load(dep_fp).astype(np.float32); conL = np.load(con_fp)
        depC = upsample_crop(depL, (H,W), ib); conC = upsample_crop(conL, (H,W), ib)
        if depC is None or conC is None or depC.shape != m.shape:
            n_align_fail += 1; perframe.append(dict(key=key, status="align_fail")); continue

        er = cv2.erode(m.astype(np.uint8), np.ones((ERODE_K,ERODE_K),np.uint8),1).astype(bool)
        sel = er & (conC>=2) & (depC>1e-6) & np.isfinite(depC)
        if sel.sum() < MIN_ANCHOR:
            sel = er & (conC>=1) & (depC>1e-6) & np.isfinite(depC)  # 완화
        if sel.sum() < 50:
            perframe.append(dict(key=key, status="few_anchor", n=int(sel.sum()))); continue

        si, zi = s[sel], depC[sel]
        a_i, b_i, inl_i, rel_i = ransac_mad_fit(si, zi)
        reproduced = abs(a_i - a_raw)/max(abs(a_raw),1e-6) < ARAW_TOL
        n_ok += int(reproduced)
        all_s.append(si); all_z.append(zi)
        ms, mz = float(np.median(si)), float(np.median(zi))
        all_sd.append(si - ms); all_zd.append(zi - mz); fr_s.append(ms); fr_z.append(mz)
        perframe.append(dict(key=key, status="ok", n=int(sel.sum()),
                             a_refit=a_i, a_raw=a_raw, reproduced=bool(reproduced),
                             inlier=inl_i, rel=rel_i))

    n_used = sum(1 for p in perframe if p.get("status")=="ok")
    gate_rate = n_ok/max(n_used,1)
    a_raw_arr = np.array(a_raw_all)

    result = dict(n_frames=len(rows), n_anchor_frames=n_used, n_align_fail=n_align_fail,
                  araw_reproduce_rate=gate_rate,
                  a_raw_stats=dict(median=float(np.nanmedian(a_raw_arr)),
                                   trimmed=float(np.mean(np.sort(a_raw_arr[np.isfinite(a_raw_arr)])
                                        [int(0.25*np.isfinite(a_raw_arr).sum()):
                                         np.isfinite(a_raw_arr).sum()-int(0.25*np.isfinite(a_raw_arr).sum())])),
                                   min=float(np.nanmin(a_raw_arr)), max=float(np.nanmax(a_raw_arr))))

    if gate_rate >= GATE_RATE and all_s:
        S = np.concatenate(all_s); Z = np.concatenate(all_z)
        # 속도: RANSAC 은 최대 POOL_MAX 점으로 (프레임당 ~3000점이면 전체 점 결과와 같은 값 — 2026-10-08 확인)
        POOL_MAX = 2_000_000
        if len(S) > POOL_MAX:
            _k = np.random.default_rng(0).choice(len(S), POOL_MAX, replace=False)
            S, Z = S[_k], Z[_k]
            all_sd = [np.concatenate(all_sd)[_k]]; all_zd = [np.concatenate(all_zd)[_k]]
            print(f"[pool] {POOL_MAX:,} 점으로 무작위 축소 (재현 가능, seed 0)")
        # (참고) 기존 방식: a 하나 + b 하나. 프레임 사이 거리-s 관계가 있으면 기울기가 치우침
        #   (7/29 3km/h: 0.388 vs 프레임별 중앙값 0.506, 4/20: 0.486 vs 0.499)
        pa, pb, pinl, prel = ransac_mad_fit(S, Z, iters=500)
        # 채택: fixed-effects = a 하나 + b 프레임별 (phase4 재-anchor 적용 모델과 동일)
        Sd = np.concatenate(all_sd); Zd = np.concatenate(all_zd)
        ga, gb_fe, ginl, grel = ransac_mad_fit(Sd, Zd, iters=500)
        fs, fz = np.array(fr_s), np.array(fr_z)
        between = float(np.polyfit(fs, fz, 1)[0]) if len(fs) > 2 else float("nan")
        r_between = float(np.corrcoef(fs, fz)[0, 1]) if len(fs) > 2 else float("nan")
        result.update(method="FIXED_EFFECTS_RANSAC_MAD",
                      global_a=ga, global_b=None, pooled_inlier=ginl, pooled_rel=grel,
                      n_pooled_points=int(len(Sd)),
                      pooled_single_b=dict(a=pa, b=pb, inlier=pinl, rel=prel),
                      between_frame=dict(slope=between, r=r_between),
                      note="a 공통 + b 프레임별 (프레임 중앙값 제거 후 RANSAC+MAD). "
                           "phase4 적용 모델과 동일. 기존 단일 b 적합값은 pooled_single_b 에 참고로 저장.")
        chosen = ga
    else:
        chosen = result["a_raw_stats"]["trimmed"]
        result.update(method="FALLBACK_ROBUST_ARAW",
                      global_a=chosen, global_b=None,
                      note=f"좌표정합 재현율 {gate_rate:.2f}<{GATE_RATE} -> pool 신뢰불가. "
                           f"robust a_raw(trimmed)=RadarCam Const 근사로 폴백. 좌표 재점검 권장.")
    result["chosen_global_a"] = chosen
    result["perframe"] = perframe
    json.dump(result, open(OUT_JSON,"w",encoding="utf-8"), indent=2, ensure_ascii=False)

    print("="*66)
    print(f"[frames] total={len(rows)} anchor_ok={n_used} align_fail={n_align_fail}")
    print(f"[자기검증] a_raw 재현율 = {gate_rate:.2f}  (>= {GATE_RATE} 이면 좌표정합 OK)")
    print(f"[a_raw 분포] median={result['a_raw_stats']['median']:.3f} "
          f"trimmed={result['a_raw_stats']['trimmed']:.3f} "
          f"[{result['a_raw_stats']['min']:.3f},{result['a_raw_stats']['max']:.3f}]")
    print("-"*66)
    print(f"[방법] {result['method']}")
    if "pooled_single_b" in result:
        print(f"[채택: a 공통 + b 프레임별] a={result['global_a']:.3f} "
              f"inlier={result['pooled_inlier']:.2f} (N={result['n_pooled_points']})")
        print(f"[참고: 기존 a·b 하나] a={result['pooled_single_b']['a']:.3f} "
              f"b={result['pooled_single_b']['b']:.3f}")
        print(f"[진단] 프레임 사이 (중앙 s → 중앙 z) 기울기 {result['between_frame']['slope']:+.3f}, "
              f"r={result['between_frame']['r']:+.2f}  (|r| 클수록 기존 방식이 치우침)")
    print(f"[채택 global_a] {chosen:.3f}")
    print(f"  vs median a_raw {result['a_raw_stats']['median']:.3f}  "
          f"({'큼' if chosen>result['a_raw_stats']['median'] else '작거나같음'})")
    print("-"*66)
    print("적용: phase4_reanchor.py GLOBAL_A_METHOD='manual', GLOBAL_A_MANUAL=<채택값> --apply")
    print("주의: depth 는 부분표면 그대로. full 두께로 부풀리지 않음(SMPL 은 A0 에서 이 부분표면에 등록).")
    print(result["note"])
    print("="*66)


if __name__ == "__main__":
    main()
