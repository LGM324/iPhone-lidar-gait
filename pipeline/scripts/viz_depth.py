"""
viz_depth.py — NB2/NB3 사람 깊이 결과 시각화 (프레임 몇 장)

프레임마다 PNG 1장 (4 패널) + PLY 1개:
  1) RGB + 사람 마스크 윤곽 + LiDAR 고신뢰(conf==2) 점 위치
  2) 최종 metric depth (재-anchor 후) 컬러맵
  3) 옆에서 본 단면 (깊이 Z ↔ 높이 Y): 재-anchor 전(회색) / 후(파랑) / LiDAR(빨강)
     → LiDAR 와 겹치면 anchor(RANSAC) 정합 OK, 몸 두께·형태 확인
  4) 위에서 본 다리 (X ↔ Z, 아래쪽 45%): 두 다리가 깊이로 분리되는지
PLY: 사람 점(RGB 색) + LiDAR 점(빨강) — MeshLab / Open3D 에서 회전해 보기

출력: {DEPTH_BASE}/work/viz/depth_{fid}.png, depth_{fid}.ply, viz_summary.json

사용:
  python viz_depth.py                 # 균등 간격 6 프레임
  python viz_depth.py --fids 5660 5750 --n 0
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walkcfg  # noqa: E402


def imread_rgb(p):
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)   # 한글 경로 대응
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def backproject(z, K, mask=None):
    H, W = z.shape
    v, u = np.nonzero((z > 0) if mask is None else (mask & (z > 0)))
    zz = z[v, u]
    x = (u - K["cx"]) / K["fx"] * zz
    y = (v - K["cy"]) / K["fy"] * zz
    return np.stack([x, y, zz], 1), v, u


def write_ply(path, pts, cols):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.zeros(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = pts[:, 0], -pts[:, 1], -pts[:, 2]   # 뷰어용: Y-up, 카메라 앞 = -Z
        rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fids", type=int, nargs="*", default=[])
    ap.add_argument("--n", type=int, default=6, help="균등 간격으로 고를 프레임 수")
    ap.add_argument("--stride", type=int, default=3, help="점군 다운샘플 간격(px)")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{walkcfg.DEPTH_SEQ}/WALK_side_*_person_metric.npy"))
    allf = [int(re.search(r"_(\d+)_person_metric", f).group(1)) for f in files]
    if not allf:
        raise SystemExit(f"깊이 결과 없음: {walkcfg.DEPTH_SEQ}")
    fids = list(args.fids)
    if args.n > 0:
        fids += [allf[i] for i in np.linspace(0, len(allf) - 1, args.n).astype(int)]
    fids = sorted(set(fids))
    K = json.load(open(walkcfg.CAM_INTR))
    out = f"{walkcfg.DEPTH_BASE}/work/viz"
    os.makedirs(out, exist_ok=True)
    summary = []

    for fid in fids:
        s = f"{fid:05d}"
        rgb = imread_rgb(f"{walkcfg.EXTRACT_DIR}/rgb/{s}.jpg")
        z = np.load(f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy")
        bak_p = f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy.bak"
        zb = np.load(bak_p) if os.path.exists(bak_p) else None
        lid = np.load(f"{walkcfg.EXTRACT_DIR}/depth_npy/{s}.npy").astype(np.float32)
        conf = np.load(f"{walkcfg.EXTRACT_DIR}/conf_npy/{s}.npy")
        H, W = z.shape
        h, w = lid.shape
        sx, sy = w / W, h / H
        Kl = dict(fx=K["fx"] * sx, fy=K["fy"] * sy, cx=K["cx"] * sx, cy=K["cy"] * sy)
        person = z > 0
        pm_l = cv2.resize(person.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        pm_l = cv2.erode(pm_l.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        lmask = pm_l & (conf >= 2) & (lid > 0)
        P_l, vl, ul = backproject(lid, Kl, lmask)

        # LiDAR 픽셀에서 최종 depth 와의 차이 (anchor 정합 확인)
        zs = cv2.resize(z, (w, h), interpolation=cv2.INTER_NEAREST)
        both = lmask & (zs > 0)
        dz = (zs[both] - lid[both]) * 100
        rec = dict(fid=fid, n_lidar=int(lmask.sum()),
                   dz_median_cm=float(np.median(dz)) if dz.size else None,
                   dz_absmed_cm=float(np.median(np.abs(dz))) if dz.size else None)

        st = args.stride
        sub = np.zeros_like(person)
        sub[::st, ::st] = True
        P, v, u = backproject(z, K, sub)
        Pb = backproject(zb, K, sub)[0] if zb is not None else None
        ys = np.where(person.any(1))[0]
        y_cut = ys.min() + 0.55 * (ys.max() - ys.min())      # 아래쪽 45% = 다리
        legs = v >= y_cut

        fig, ax = plt.subplots(1, 4, figsize=(20, 6.4), gridspec_kw=dict(width_ratios=[1, 1, 1.15, 1.15]))
        x0, x1 = np.where(person.any(0))[0][[0, -1]]
        y0, y1 = ys[[0, -1]]
        pad = 60
        cx0, cx1, cy0, cy1 = max(0, x0 - pad), min(W, x1 + pad), max(0, y0 - pad), min(H, y1 + pad)
        ax[0].imshow(rgb[cy0:cy1, cx0:cx1])
        ax[0].contour(person[cy0:cy1, cx0:cx1], levels=[0.5], colors="lime", linewidths=1)
        ax[0].scatter(ul / sx - cx0, vl / sy - cy0, s=2, c="red")
        ax[0].set_title(f"fid {fid}  mask + LiDAR conf2 ({lmask.sum()} pts)"); ax[0].axis("off")

        zc = np.where(person, z, np.nan)[cy0:cy1, cx0:cx1]
        im = ax[1].imshow(zc, cmap="turbo")
        plt.colorbar(im, ax=ax[1], fraction=0.046, label="depth (m)")
        ax[1].set_title("final metric depth (re-anchored)"); ax[1].axis("off")

        if Pb is not None:
            ax[2].scatter(Pb[:, 2], -Pb[:, 1], s=0.3, c="0.7", label="before re-anchor")
        ax[2].scatter(P[:, 2], -P[:, 1], s=0.3, c="tab:blue", label="after re-anchor")
        ax[2].scatter(P_l[:, 2], -P_l[:, 1], s=6, c="red", label="LiDAR conf2")
        ax[2].set_aspect("equal"); ax[2].set_xlabel("depth Z (m)"); ax[2].set_ylabel("height -Y (m)")
        ax[2].set_title(f"side view  |after-LiDAR| median {rec['dz_absmed_cm']:.1f} cm")
        ax[2].legend(markerscale=8, fontsize=8, loc="upper right")

        sc = ax[3].scatter(P[legs, 0], P[legs, 2], s=0.5, c=-P[legs, 1], cmap="viridis")
        lp = (vl / sy) >= y_cut                               # LiDAR 점도 같은 아래쪽 45%
        if lp.any():
            ax[3].scatter(P_l[lp, 0], P_l[lp, 2], s=6, c="red")
        ax[3].invert_yaxis(); ax[3].set_aspect("equal")
        ax[3].set_xlabel("X (m)"); ax[3].set_ylabel("depth Z (m, camera at top)")
        ax[3].set_title("top view of legs (lower 45%)  color = height")
        fig.tight_layout()
        fig.savefig(f"{out}/depth_{s}.png", dpi=90)
        plt.close(fig)

        cols = rgb[v, u]
        allp = np.vstack([P, P_l])
        allc = np.vstack([cols, np.tile([[255, 0, 0]], (len(P_l), 1))]).astype(np.uint8)
        write_ply(f"{out}/depth_{s}.ply", allp, allc)
        summary.append(rec)
        print(f"[viz] fid {fid}: LiDAR conf2 {rec['n_lidar']}점, final−LiDAR median {rec['dz_median_cm']:+.1f} cm "
              f"(|·| {rec['dz_absmed_cm']:.1f} cm) → depth_{s}.png / .ply")

    json.dump(summary, open(f"{out}/viz_summary.json", "w"), indent=2)
    print("[saved]", out)


if __name__ == "__main__":
    main()
