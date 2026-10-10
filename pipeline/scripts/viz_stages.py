"""
viz_stages.py — 노트북(NB1~NB7) 단계별 시각화. 결과는 {SESS}/viz/NB{k}/ 에 단계마다 따로 저장.

  NB1  frame_{fid}.png   사진 / raw LiDAR 깊이 / LiDAR 신뢰도
       overview.png      영상 전체 움직임 + 추출 구간
  NB2  frame_{fid}.png   사람 마스크 / Sapiens 상대 깊이 / RANSAC 미터 깊이 / LiDAR 대조 단면
       overview.png      프레임별 a, b, 기준점 수, inlier, 깊이 중심
  NB3  frame_{fid}.png   재-anchor 전 / 후 / 차이 / 단면 비교 (+ LiDAR)
       overview.png      프레임별 a vs 전역 a, 몸통 두께 전/후, 깊이 중심 전/후
  NB4  frame_{fid}.png   4D-Humans 자세 메쉬 (정면 / 측면) + 검출 상자
       overview.png      4D-Humans 무릎각 좌우, 검출 신뢰도, 프레임간 몸 방향 변화
  NB5  frame_{fid}.png   좌우 다리 구분 / SMPL 투영 / 단면 / 위에서 본 다리  (+ points.ply, smpl.ply)
       overview.png      A0 잔차·flip margin, 다리 점 수, near·far 무릎각 fit
  NB6  overview.png      far 허벅지 점 수 + 가림 판정 + 최종 무릎각(라벨)
  NB7  overview.png      IMU vs SMPL 곡선 / 산점도 / Bland-Altman / 오차 분포

사용:
  python viz_stages.py                          # 가능한 단계 전부, 프레임 3장(균등)
  python viz_stages.py --nb 2 3 --fids 5660 5750 --n 0
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

ROOT_OUT = f"{walkcfg.SESS}/viz"
LEG_PARTS = {"R": ("rightUpLeg", "rightLeg", "rightFoot", "rightToeBase"),
             "L": ("leftUpLeg", "leftLeg", "leftFoot", "leftToeBase")}
SIDE_COL = np.array([[150, 150, 150], [40, 90, 230], [220, 40, 40]], np.uint8)   # upper / far / near


# ------------------------------------------------------------------ 공용
def out_dir(nb):
    d = f"{ROOT_OUT}/NB{nb}"
    os.makedirs(d, exist_ok=True)
    return d


def imread_rgb(p):
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_npz(p):
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def load_json(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fid_of(key):
    return int(re.search(r"(\d+)$", key).group(1))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print("  saved", os.path.relpath(path, walkcfg.SESS))


def person_crop(mask, pad=50):
    ys, xs = np.where(mask)
    H, W = mask.shape
    return max(0, ys.min() - pad), min(H, ys.max() + pad), max(0, xs.min() - pad), min(W, xs.max() + pad)


def backproject(z, K, mask=None, stride=1):
    sel = (z > 0) if mask is None else (mask & (z > 0))
    if stride > 1:
        sub = np.zeros_like(sel); sub[::stride, ::stride] = True; sel &= sub
    v, u = np.nonzero(sel)
    zz = z[v, u]
    return np.stack([(u - K["cx"]) / K["fx"] * zz, (v - K["cy"]) / K["fy"] * zz, zz], 1), v, u


def lidar_person_points(fid, person, K):
    s = f"{fid:05d}"
    lid = np.load(f"{walkcfg.EXTRACT_DIR}/depth_npy/{s}.npy").astype(np.float32)
    conf = np.load(f"{walkcfg.EXTRACT_DIR}/conf_npy/{s}.npy")
    H, W = person.shape; h, w = lid.shape; sx, sy = w / W, h / H
    pm = cv2.resize(person.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    pm = cv2.erode(pm, np.ones((3, 3), np.uint8)).astype(bool)
    Kl = dict(fx=K["fx"] * sx, fy=K["fy"] * sy, cx=K["cx"] * sx, cy=K["cy"] * sy)
    P, v, u = backproject(lid, Kl, pm & (conf >= 2))
    return P, lid, conf, pm


def write_ply_points(path, pts, cols):
    rec = np.zeros(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], -pts[:, 1], -pts[:, 2]
    rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\nproperty float x\n"
                 "property float y\nproperty float z\nproperty uchar red\nproperty uchar green\n"
                 "property uchar blue\nend_header\n").encode())
        f.write(rec.tobytes())


def write_ply_mesh(path, verts, faces, cols):
    v = np.zeros(len(verts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v["x"], v["y"], v["z"] = verts[:, 0], -verts[:, 1], -verts[:, 2]
    v["r"], v["g"], v["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    fc = np.zeros(len(faces), dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    fc["n"] = 3
    fc["a"], fc["b"], fc["c"] = faces[:, 0], faces[:, 2], faces[:, 1]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\nproperty float x\n"
                 "property float y\nproperty float z\nproperty uchar red\nproperty uchar green\n"
                 f"property uchar blue\nelement face {len(faces)}\nproperty list uchar int vertex_indices\n"
                 "end_header\n").encode())
        f.write(v.tobytes()); f.write(fc.tobytes())


class Smpl:
    """β 고정 SMPL. 4D-Humans 자세(NB4) 또는 NB5 fit 자세로 정점 생성."""
    def __init__(self):
        import torch
        import smplx
        self.torch = torch
        self.m = smplx.create(walkcfg.SMPL_MODEL, model_type="smpl", gender="neutral", num_betas=10)
        self.faces = self.m.faces.astype(np.int64)
        self.betas = torch.tensor(np.load(walkcfg.FROZEN_NPZ)["betas"][:10].astype(np.float32)).reshape(1, 10)
        seg = load_json(walkcfg.SEG_JSON) or {}
        self.leg = {k: np.zeros(6890, bool) for k in "RL"}
        for k, parts in LEG_PARTS.items():
            for p in parts:
                self.leg[k][seg.get(p, [])] = True
        self.cols = np.full((6890, 3), 230, np.uint8)
        self.cols[self.leg["R"]] = (220, 40, 40)
        self.cols[self.leg["L"]] = (40, 90, 230)

    def verts(self, go, bp, tr=None):
        t = self.torch
        with t.no_grad():
            out = self.m(betas=self.betas, body_pose=t.tensor(np.asarray(bp, np.float32)).reshape(1, 69),
                         global_orient=t.tensor(np.asarray(go, np.float32)).reshape(1, 3),
                         transl=t.tensor(np.zeros(3) if tr is None else np.asarray(tr), dtype=t.float32).reshape(1, 3))
        return out.vertices[0].numpy()


def frame_list(args):
    allf = sorted(int(re.search(r"(\d+)\.jpg$", p).group(1)) for p in glob.glob(f"{walkcfg.EXTRACT_DIR}/rgb/*.jpg"))
    fids = list(args.fids)
    if args.n > 0 and allf:
        fids += [allf[i] for i in np.linspace(0, len(allf) - 1, args.n).astype(int)]
    return sorted(set(fids))


# ------------------------------------------------------------------ NB1
def nb1(fids, K, args):
    d = out_dir(1)
    meta = load_json(f"{walkcfg.EXTRACT_DIR}/meta.json") or {}
    env = load_npz(f"{walkcfg.EXTRACT_DIR}/motion_envelope.npz")
    win = meta.get("window") or {}
    if env is not None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        k = max(1, int(meta.get("fps", 30)))
        ax.plot(env["fid"], env["motion"], lw=.4, c="0.6")
        ax.plot(env["fid"], np.convolve(env["motion"], np.ones(k) / k, mode="same"), c="k", lw=1.2, label="1 s mean")
        if win:
            ax.axvspan(win["start"], win["end"], color="tab:red", alpha=.25, label=f"extracted {win['start']}–{win['end']}")
        ax.set_xlabel("fid"); ax.set_ylabel("depth motion (m)"); ax.legend()
        ax.set_title(f"NB1 · whole video motion  ({meta.get('r3d')}, {meta.get('n_frames')} f @ {meta.get('fps')} fps)")
        save(fig, f"{d}/overview.png")
    for fid in fids:
        s = f"{fid:05d}"
        rgb = imread_rgb(f"{walkcfg.EXTRACT_DIR}/rgb/{s}.jpg")
        lid = np.load(f"{walkcfg.EXTRACT_DIR}/depth_npy/{s}.npy")
        conf = np.load(f"{walkcfg.EXTRACT_DIR}/conf_npy/{s}.npy")
        fig, ax = plt.subplots(1, 3, figsize=(15, 6.5))
        ax[0].imshow(rgb); ax[0].set_title(f"RGB {rgb.shape[1]}×{rgb.shape[0]}")
        im = ax[1].imshow(np.where(lid > 0, lid, np.nan), cmap="turbo"); plt.colorbar(im, ax=ax[1], fraction=.046, label="m")
        ax[1].set_title(f"raw LiDAR depth {lid.shape[1]}×{lid.shape[0]}")
        im = ax[2].imshow(conf, cmap="viridis", vmin=0, vmax=2); plt.colorbar(im, ax=ax[2], fraction=.046, ticks=[0, 1, 2])
        ax[2].set_title(f"LiDAR confidence  (conf2 {100 * (conf == 2).mean():.0f}% of pixels)")
        for a in ax: a.axis("off")
        fig.suptitle(f"NB1 · fid {fid}")
        save(fig, f"{d}/frame_{s}.png")


# ------------------------------------------------------------------ NB2
def nb2(fids, K, args):
    d = out_dir(2)
    rep = load_json(f"{walkcfg.DEPTH_SEQ}/anchor_report.json")
    if rep:
        f = [fid_of(r["key"]) for r in rep]
        fig, ax = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
        ax[0].plot(f, [r["a"] for r in rep], ".-", ms=3); ax[0].set_ylabel("a"); ax[0].set_title("per-frame anchor slope a (RANSAC, z = a·s + b)")
        ax[1].plot(f, [r["b"] for r in rep], ".-", ms=3, c="tab:orange"); ax[1].set_ylabel("b (m)"); ax[1].set_title("per-frame offset b")
        ax[2].plot(f, [r["n_anchor"] for r in rep], c="tab:green"); ax[2].set_ylabel("points"); ax[2].set_title("LiDAR anchor points used (conf==2 inside person)")
        ax[3].plot(f, [r["inlier"] * 100 for r in rep], c="tab:purple", label="inlier %")
        t = ax[3].twinx(); t.plot(f, [r["rel"] * 100 for r in rep], c="k", lw=.7, label="rel resid %"); t.set_ylabel("rel resid %")
        ax[3].set_ylabel("inlier %"); ax[3].set_title("RANSAC inlier ratio / relative residual")
        ax[4].plot(f, [r["z_med"] for r in rep], c="tab:blue"); ax[4].set_ylabel("m"); ax[4].set_title("person depth center z_med"); ax[4].set_xlabel("fid")
        fig.suptitle("NB2 · Sapiens + LiDAR anchor (per frame)")
        save(fig, f"{d}/overview.png")
    for fid in fids:
        s = f"{fid:05d}"
        bak = f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy.bak"
        zp = bak if os.path.exists(bak) else f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy"
        if not os.path.exists(zp):
            continue
        z = np.load(zp); person = z > 0
        rgb = imread_rgb(f"{walkcfg.EXTRACT_DIR}/rgb/{s}.jpg")
        vid = load_npz(f"{walkcfg.VIDEO_SEQ}/WALK_side_{s}.npz")
        y0, y1, x0, x1 = person_crop(person)
        P_l, lid, conf, pm = lidar_person_points(fid, person, K)
        P, _, _ = backproject(z, K, stride=3)
        zs = cv2.resize(z, lid.shape[::-1], interpolation=cv2.INTER_NEAREST)
        both = pm & (conf >= 2) & (zs > 0) & (lid > 0)
        dz = (zs[both] - lid[both]) * 100
        fig, ax = plt.subplots(1, 4, figsize=(20, 7), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.1]))
        ax[0].imshow(rgb[y0:y1, x0:x1]); ax[0].contour(person[y0:y1, x0:x1], levels=[.5], colors="lime", linewidths=1)
        ax[0].set_title("Sapiens person mask"); ax[0].axis("off")
        if vid is not None:
            im = ax[1].imshow(np.where(vid["mask_crop"], vid["s_relative"], np.nan), cmap="magma")
            plt.colorbar(im, ax=ax[1], fraction=.046, label="relative")
            ax[1].set_title(f"Sapiens relative depth s\na={float(vid['a_raw']):.3f}  b={float(vid['b_raw']):.2f}  "
                            f"inlier {float(vid['inlier']) * 100:.0f}%")
        ax[1].axis("off")
        im = ax[2].imshow(np.where(person, z, np.nan)[y0:y1, x0:x1], cmap="turbo"); plt.colorbar(im, ax=ax[2], fraction=.046, label="m")
        ax[2].set_title("metric depth after RANSAC anchor\n(NB2 output, before NB3)"); ax[2].axis("off")
        ax[3].scatter(P[:, 2], -P[:, 1], s=.3, c="tab:blue", label="NB2 metric depth")
        ax[3].scatter(P_l[:, 2], -P_l[:, 1], s=5, c="red", label="LiDAR conf2")
        ax[3].set_aspect("equal"); ax[3].set_xlabel("depth Z (m)"); ax[3].set_ylabel("height -Y (m)")
        ax[3].set_title(f"side view vs LiDAR\n(depth − LiDAR) median {np.median(dz):+.1f} cm, |·| {np.median(np.abs(dz)):.1f} cm")
        ax[3].legend(markerscale=6, fontsize=8)
        fig.suptitle(f"NB2 · fid {fid}")
        save(fig, f"{d}/frame_{s}.png")


# ------------------------------------------------------------------ NB3
def nb3(fids, K, args):
    d = out_dir(3)
    A = f"{walkcfg.DEPTH_BASE}/work/analysis"
    g, p4 = load_json(f"{A}/global_const_fit.json"), load_json(f"{A}/phase4_report.json")
    if g and p4:
        pf = [r for r in g["perframe"] if r.get("status") == "ok"]
        fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        ax[0].plot([fid_of(r["key"]) for r in pf], [r["a_raw"] for r in pf], ".-", ms=3, label="per-frame a (NB2)")
        ax[0].axhline(g["chosen_global_a"], c="r", ls="--", label=f"global a = {g['chosen_global_a']:.3f}")
        st = g["a_raw_stats"]
        ax[0].axhline(st["median"], c="0.4", ls=":", label=f"per-frame median {st['median']:.3f}")
        ax[0].legend(fontsize=8); ax[0].set_ylabel("a")
        ax[0].set_title(f"anchor slope: per-frame vs single global (pooled {g.get('n_pooled_points', 0):,} pts, inlier {g.get('pooled_inlier', 0) * 100:.0f}%)")
        q = p4["per_frame"]; f = [fid_of(r["key"]) for r in q]
        sm = p4["summary"]
        ax[1].plot(f, [r["torso_before_cm"] for r in q], c="0.6", label=f"before  std {sm['torso_before']['std']:.2f} cm")
        ax[1].plot(f, [r["torso_after_cm"] for r in q], c="tab:blue", label=f"after   std {sm['torso_after']['std']:.2f} cm")
        ax[1].set_ylabel("cm"); ax[1].legend(fontsize=8); ax[1].set_title("torso thickness (near-side relief)")
        ax[2].plot(f, [r["z_med_before"] for r in q], c="0.6", label="before")
        ax[2].plot(f, [r["z_med_after"] for r in q], c="tab:blue", ls="--", label=f"after (max drift {sm['z_med_preserved_max_drift_cm']:.2f} cm)")
        ax[2].set_ylabel("m"); ax[2].legend(fontsize=8); ax[2].set_title("depth center z_med"); ax[2].set_xlabel("fid")
        fig.suptitle("NB3 · global re-anchor")
        save(fig, f"{d}/overview.png")
    for fid in fids:
        s = f"{fid:05d}"
        zp, bp = f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy", f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy.bak"
        if not (os.path.exists(zp) and os.path.exists(bp)):
            continue
        z, zb = np.load(zp), np.load(bp); person = z > 0
        y0, y1, x0, x1 = person_crop(person)
        vmin, vmax = np.percentile(z[person], [1, 99])
        diff = np.where(person & (zb > 0), (z - zb) * 100, np.nan)
        P, _, _ = backproject(z, K, stride=3); Pb, _, _ = backproject(zb, K, stride=3)
        P_l = lidar_person_points(fid, person, K)[0]
        fig, ax = plt.subplots(1, 4, figsize=(20, 7))
        for a, arr, ttl in ((ax[0], zb, "before (NB2)"), (ax[1], z, "after (NB3)")):
            im = a.imshow(np.where(arr > 0, arr, np.nan)[y0:y1, x0:x1], cmap="turbo", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=a, fraction=.046, label="m"); a.set_title(ttl); a.axis("off")
        lim = np.nanpercentile(np.abs(diff), 98)
        im = ax[2].imshow(diff[y0:y1, x0:x1], cmap="coolwarm", vmin=-lim, vmax=lim)
        plt.colorbar(im, ax=ax[2], fraction=.046, label="cm"); ax[2].axis("off")
        ax[2].set_title(f"after − before\nmedian {np.nanmedian(diff):+.2f} cm, p5/p95 {np.nanpercentile(diff, 5):+.1f}/{np.nanpercentile(diff, 95):+.1f}")
        ax[3].scatter(Pb[:, 2], -Pb[:, 1], s=.3, c="0.65", label="before")
        ax[3].scatter(P[:, 2], -P[:, 1], s=.3, c="tab:blue", label="after")
        ax[3].scatter(P_l[:, 2], -P_l[:, 1], s=5, c="red", label="LiDAR conf2")
        ax[3].set_aspect("equal"); ax[3].set_xlabel("depth Z (m)"); ax[3].set_ylabel("height -Y (m)")
        ax[3].set_title("side view: thickness before vs after"); ax[3].legend(markerscale=6, fontsize=8)
        fig.suptitle(f"NB3 · fid {fid}")
        save(fig, f"{d}/frame_{s}.png")


# ------------------------------------------------------------------ NB4
def nb4(fids, K, args, smpl):
    d = out_dir(4)
    fdh = load_npz(walkcfg.FOURDH_SEQ)
    if fdh is None:
        return
    near, far = load_npz(walkcfg.NEAR_F3), load_npz(walkcfg.FAR_F2)
    f = fdh["fids"]
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    if near is not None:
        ax[0].plot(near["fid"], near["knee_4d"], c="tab:red", label="R knee (4D-Humans)")
    if far is not None:
        ax[0].plot(far["fid"], far["far_knee_4d"], c="tab:blue", label="L knee (4D-Humans)")
    ax[0].set_ylabel("deg"); ax[0].legend(fontsize=8); ax[0].set_title("4D-Humans initial knee angles (segment angle, same definition as fit)")
    ax[1].plot(f, fdh["detector_score"], c="k"); ax[1].set_ylabel("score")
    ax[1].set_title(f"person detector score (persons per frame: {sorted(set(np.asarray(fdh['detected_person_count']).tolist()))})")
    from scipy.spatial.transform import Rotation as R
    Rm = R.from_rotvec(fdh["global_orient_aa"]); rel = np.degrees((Rm[:-1].inv() * Rm[1:]).magnitude())
    ax[2].plot(f[1:], rel, c="tab:purple"); ax[2].set_ylabel("deg"); ax[2].set_xlabel("fid")
    ax[2].set_title(f"frame-to-frame body orientation change (true rotation)  median {np.median(rel):.1f}°, max {rel.max():.1f}°")
    fig.suptitle("NB4 · 4D-Humans (HMR 2.0) per-frame init")
    save(fig, f"{d}/overview.png")
    if smpl is None:
        return
    row = {int(x): i for i, x in enumerate(f)}
    bp_all = np.asarray(fdh["body_pose_aa"]).reshape(len(f), -1)
    for fid in fids:
        if fid not in row:
            continue
        i = row[fid]
        V = smpl.verts(fdh["global_orient_aa"][i], bp_all[i])
        rgb = imread_rgb(f"{walkcfg.EXTRACT_DIR}/rgb/{fid:05d}.jpg")
        fig, ax = plt.subplots(1, 3, figsize=(16, 7))
        ax[0].imshow(rgb); x0, y0, x1, y1 = fdh["bbox_xyxy"][i]
        ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=2))
        ax[0].set_title(f"detection box (score {fdh['detector_score'][i]:.2f})"); ax[0].axis("off")
        c = smpl.cols / 255
        ax[1].scatter(V[:, 0], -V[:, 1], s=.5, c=c); ax[1].set_aspect("equal"); ax[1].set_title("4D-Humans pose, camera view (x, -y)")
        ax[2].scatter(V[:, 2], -V[:, 1], s=.5, c=c); ax[2].set_aspect("equal"); ax[2].set_title("4D-Humans pose, side view (z, -y)")
        fig.suptitle(f"NB4 · fid {fid}  (β frozen; red = right leg, blue = left leg)")
        save(fig, f"{d}/frame_{fid:05d}.png")


# ------------------------------------------------------------------ NB5
def nb5(fids, K, args, smpl):
    d = out_dir(5)
    a0, near, far = load_npz(walkcfg.A0_NPZ), load_npz(walkcfg.NEAR_F3), load_npz(walkcfg.FAR_F2)
    fig, ax = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    if a0 is not None:
        ax[0].plot(a0["frame"], a0["residual_cm"], c="k", label="A0 residual median")
        t = ax[0].twinx(); t.plot(a0["frame"], a0["flip_margin"], c="tab:orange", alpha=.6); t.set_ylabel("flip margin", color="tab:orange")
        ax[0].set_ylabel("cm"); ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title("A0 root alignment: residual / facing ambiguity (flip margin ≈ 0 → ambiguous)")
    if near is not None:
        ax[1].plot(near["fid"], near["n_near_leg"], c="tab:red", label="near(R) leg pts")
    if far is not None:
        ax[1].plot(far["fid"], far["n_far_leg"], c="tab:blue", label="far(L) leg pts")
        ax[1].plot(far["fid"], far["thigh_n"], c="tab:blue", ls=":", label="far thigh pts")
        for x in far["fid"][far["label"].astype(str) != "MEASURED"]:
            ax[1].axvspan(x - .5, x + .5, color="0.85", lw=0)
        ax[1].set_yscale("symlog", linthresh=100); ax[1].legend(fontsize=8)
    ax[1].set_title("left/right split: leg point counts (gray = far INFERRED)")
    if near is not None:
        ax[2].plot(near["fid"], near["knee_4d"], c="0.5", label="4D init")
        ax[2].plot(near["fid"], near["knee_fit"], c="tab:red", label="depth fit")
        t = ax[2].twinx(); t.plot(near["fid"], near["resid_after_cm"], c="k", lw=.6, alpha=.5); t.set_ylabel("resid cm")
        ax[2].set_ylabel("deg"); ax[2].legend(fontsize=8, loc="upper left")
    ax[2].set_title("near(R) knee: 4D init vs depth fit (λ = 0)")
    if far is not None:
        lab = far["label"].astype(str); m = lab == "MEASURED"
        ax[3].plot(far["fid"], far["far_knee_4d"], c="0.5", label="4D init")
        ax[3].scatter(far["fid"][m], far["far_knee_fit"][m], s=8, c="tab:blue", label=f"MEASURED fit ({m.sum()})")
        ax[3].scatter(far["fid"][~m], far["far_knee_fit"][~m], s=8, c="0.6", marker="x", label=f"INFERRED = 4D ({(~m).sum()})")
        ax[3].set_ylabel("deg"); ax[3].legend(fontsize=8)
    ax[3].set_title("far(L) knee: depth fit (MEASURED) vs INFERRED"); ax[3].set_xlabel("fid")
    fig.suptitle("NB5 · SMPL lower-body fit")
    save(fig, f"{d}/overview.png")

    if a0 is None or smpl is None:
        return
    from a0_align_v2 import backproject as bp_a0, side_from_zorder, SIDE_UPPER, SIDE_FAR, SIDE_NEAR
    fdh = np.load(walkcfg.FOURDH_SEQ, allow_pickle=True)
    bp_all = np.asarray(fdh["body_pose_aa"]).reshape(len(fdh["fids"]), -1)
    rows = lambda arr: {int(x): i for i, x in enumerate(np.asarray(arr).astype(int))}
    r_a0, r_fdh = rows(a0["frame"]), rows(fdh["fids"])
    r_nr = rows(near["fid"]) if near is not None else {}
    r_fr = rows(far["fid"]) if far is not None else {}
    from knee_measurement_basis import assemble_body_pose
    for fid in fids:
        s = f"{fid:05d}"
        zp = f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy"
        if fid not in r_a0 or fid not in r_fdh or not os.path.exists(zp):
            continue
        z = np.load(zp); person = z > 0
        rgb = imread_rgb(f"{walkcfg.EXTRACT_DIR}/rgb/{s}.jpg")
        y0, y1, x0, x1 = person_crop(person)
        pts, pix = bp_a0(z, K)
        side = side_from_zorder(pts)
        pv, pu = np.unravel_index(pix, z.shape)
        base = bp_all[r_fdh[fid]]
        if fid in r_nr:
            ft = far["far_theta_fit"][r_fr[fid]] if fid in r_fr else np.full(9, np.nan)
            fit = bool(far["is_fit"][r_fr[fid]]) if fid in r_fr else False
            bp = assemble_body_pose(base, near["theta_fit"][r_nr[fid]], ft, fit)
        else:
            bp = base
        i = r_a0[fid]
        V = smpl.verts(a0["global_orient"][i], bp, a0["transl"][i])
        n_near, n_far = int((side == SIDE_NEAR).sum()), int((side == SIDE_FAR).sum())
        st = args.stride
        fig, ax = plt.subplots(1, 4, figsize=(22, 7.5), gridspec_kw=dict(width_ratios=[.8, .8, 1, 1.3]))
        ax[0].imshow(rgb[y0:y1, x0:x1])
        ax[0].scatter(pu[::st] - x0, pv[::st] - y0, s=.3, c=SIDE_COL[side[::st]] / 255)
        ax[0].set_title(f"left/right split\nnear(R) {n_near} / far(L) {n_far}"); ax[0].axis("off")
        u = K["fx"] * V[:, 0] / V[:, 2] + K["cx"]; v = K["fy"] * V[:, 1] / V[:, 2] + K["cy"]
        ax[1].imshow(rgb[y0:y1, x0:x1]); ax[1].scatter(u - x0, v - y0, s=.4, c=smpl.cols / 255, alpha=.6)
        lab = far["label"][r_fr[fid]] if fid in r_fr else "?"
        ax[1].set_title(f"SMPL fit projected\nfar label: {lab}"); ax[1].axis("off")
        ps = pts[::st]
        ax[2].scatter(ps[:, 2], -ps[:, 1], s=.3, c="0.6", label="depth")
        ax[2].scatter(V[:, 2], -V[:, 1], s=.5, c=smpl.cols / 255, label="SMPL")
        ax[2].set_aspect("equal"); ax[2].set_xlabel("depth Z (m)"); ax[2].set_ylabel("height -Y (m)")
        ax[2].set_title("side view: depth + SMPL"); ax[2].legend(markerscale=10, fontsize=8)
        low = side != SIDE_UPPER
        ax[3].scatter(pts[low][::2, 0], pts[low][::2, 2], s=.4, c=SIDE_COL[side[low][::2]] / 255)
        for kk, cc in (("R", (.55, 0, 0)), ("L", (0, .15, .55))):
            lv = V[smpl.leg[kk]]
            ax[3].scatter(lv[:, 0], lv[:, 2], s=1.5, c=[cc], marker="x", label=f"SMPL {kk} leg")
        ax[3].invert_yaxis(); ax[3].set_aspect("equal"); ax[3].legend(fontsize=8, markerscale=4)
        ax[3].set_xlabel("X (m)"); ax[3].set_ylabel("depth Z (m, camera at top)")
        ax[3].set_title("top view of legs: depth (near red / far blue) + SMPL legs (x)")
        fig.suptitle(f"NB5 · fid {fid}")
        save(fig, f"{d}/frame_{s}.png")
        write_ply_points(f"{d}/frame_{s}_points.ply", pts[::2], SIDE_COL[side[::2]])
        write_ply_mesh(f"{d}/frame_{s}_smpl.ply", V, smpl.faces, smpl.cols)


# ------------------------------------------------------------------ NB6
def nb6(fids, K, args):
    d = out_dir(6)
    basis = load_npz(f"{walkcfg.KNEE_DIR}/knee_basis_full.npz")
    gate = load_npz(f"{walkcfg.KNEE_DIR}/far_thigh_gate_full.npz")
    kf = load_npz(f"{walkcfg.KNEE_DIR}/far_knee_final.npz")
    if kf is None:
        return
    fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    if basis is not None:
        ax[0].plot(basis["fid"], basis["thigh_far_npts"], c="tab:blue", label="far thigh pts")
        if gate is not None:
            ax[0].axhline(float(gate["N_low"]), c="r", ls="--", label=f"N_low = {float(gate['N_low']):.0f}")
            for st_, ln in zip(gate["start_fid"], gate["length"]):
                ax[0].axvspan(st_ - .5, st_ + ln - .5, color="tab:red", alpha=.15)
        ax[0].set_yscale("symlog", linthresh=100); ax[0].legend(fontsize=8)
        ax[0].set_title("collapse gate: far thigh observed points (red = collapsed run)")
        ax[1].plot(basis["fid"], basis["thigh_near_resid_cm"], c="tab:red", label="near thigh resid")
        ax[1].plot(basis["fid"], basis["shank_near_resid_cm"], c="tab:red", ls=":", label="near shank resid")
        ax[1].plot(basis["fid"], basis["thigh_far_resid_cm"], c="tab:blue", label="far thigh resid")
        ax[1].set_ylabel("cm"); ax[1].legend(fontsize=8, ncol=3); ax[1].set_title("measurement evidence: per-part surface residual")
    lab = kf["label"].astype(str)
    ax[2].plot(kf["fid"], kf["knee_near"], c="tab:red", lw=1, label="near(R) final")
    ax[2].plot(kf["fid"], kf["knee_far_final"], c="0.6", lw=.8)
    for nm, c in (("MEASURED", "tab:blue"), ("INTERPOLATED", "tab:green"), ("INTERPOLATED_HIGH_CURV", "orange")):
        m = lab == nm
        ax[2].scatter(kf["fid"][m], kf["knee_far_final"][m], s=10, c=c, label=f"far {nm} ({m.sum()})")
    ax[2].set_ylabel("deg"); ax[2].set_xlabel("fid"); ax[2].legend(fontsize=8)
    ax[2].set_title("final knee time series")
    fig.suptitle("NB6 · knee time series + far occlusion infill")
    save(fig, f"{d}/overview.png")


# ------------------------------------------------------------------ NB7
def nb7(fids, K, args):
    d = out_dir(7)
    r = load_npz(f"{walkcfg.IMU_OUT}/imu_validation.npz")
    js = load_json(f"{walkcfg.IMU_OUT}/imu_validation.json")
    if r is None:
        return
    lab = r["label"].astype(str)
    fig, ax = plt.subplots(3, 3, figsize=(18, 14), gridspec_kw=dict(height_ratios=[1, 1, 1]))
    for row, (k, nm, c) in enumerate((("R", "near(R)", "tab:red"), ("L", "far(L)", "tab:blue"))):
        s_, i_ = r[f"smpl_{k}"], r[f"imu_{k}"]
        ok = np.isfinite(i_)
        a = ax[row, 0]
        a.plot(r["fid"], i_, c="k", label="IMU"); a.plot(r["fid"], s_, c=c, label="SMPL")
        if k == "L":
            a.scatter(r["fid"][lab != "MEASURED"], s_[lab != "MEASURED"], s=8, facecolors="none", edgecolors="orange", label="interpolated")
        a.set_title(f"{nm} knee"); a.set_ylabel("deg"); a.legend(fontsize=8)
        a = ax[row, 1]
        cm = np.where(lab == "MEASURED", c, "orange") if k == "L" else c
        a.scatter(i_[ok], s_[ok], s=8, c=cm[ok] if k == "L" else c)
        lim = [0, max(np.nanmax(i_), np.nanmax(s_)) + 5]
        a.plot(lim, lim, "k--", lw=.8); a.set_xlim(lim); a.set_ylim(lim); a.set_aspect("equal")
        a.set_xlabel("IMU (deg)"); a.set_ylabel("SMPL (deg)")
        a.set_title(f"{nm}: r = {np.corrcoef(i_[ok], s_[ok])[0, 1]:.3f}")
        a = ax[row, 2]
        mean, diff = (i_ + s_) / 2, s_ - i_
        b, sd = np.nanmean(diff), np.nanstd(diff)
        a.scatter(mean, diff, s=8, c=c)
        for y, ls in ((b, "-"), (b + 1.96 * sd, "--"), (b - 1.96 * sd, "--")):
            a.axhline(y, c="k", ls=ls, lw=.8)
        a.set_xlabel("mean of IMU and SMPL (deg)"); a.set_ylabel("SMPL − IMU (deg)")
        a.set_title(f"{nm} Bland-Altman: bias {b:+.2f}°, LoA ±{1.96 * sd:.1f}°")
    a = ax[2, 0]
    for k, c, nm in (("R", "tab:red", "near(R)"), ("L", "tab:blue", "far(L)")):
        e = r[f"smpl_{k}"] - r[f"imu_{k}"]
        a.hist(e[np.isfinite(e)], bins=40, alpha=.5, color=c, label=nm)
    a.axvline(0, c="k", lw=.8); a.set_xlabel("SMPL − IMU (deg)"); a.legend(); a.set_title("error distribution")
    a = ax[2, 1]
    for k, c in (("R", "tab:red"), ("L", "tab:blue")):
        i_, e = r[f"imu_{k}"], r[f"smpl_{k}"] - r[f"imu_{k}"]
        a.scatter(i_, e, s=6, c=c, alpha=.6)
    a.axhline(0, c="k", lw=.8); a.axvline(30, c="0.5", ls=":")
    a.set_xlabel("IMU knee (deg)  ← stance | swing →"); a.set_ylabel("SMPL − IMU (deg)"); a.set_title("error vs knee angle (gait phase proxy)")
    a = ax[2, 2]; a.axis("off")
    if js:
        lines = [f"offset {js['offset_s']:.3f} s  (jump {js.get('jump_offset_s')}, coarse {js['coarse_offset_s']:.2f})",
                 f"sync ok: {js['sync_ok']}   r_fine {js['r_fine']:.3f}", ""]
        for m in js["metrics"]:
            lines.append(f"{m['leg']:8s} {m['subset']:8s} n={m['n']:3d}  RMSE {m['rmse']:.2f}  bias {m['bias']:+.2f}  r {m['r']:.3f}")
        a.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.suptitle("NB7 · IMU (Xsens) validation")
    save(fig, f"{d}/overview.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--fids", type=int, nargs="*", default=[])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    K = json.load(open(walkcfg.CAM_INTR))
    fids = frame_list(args)
    smpl = None
    if {4, 5} & set(args.nb):
        try:
            smpl = Smpl()
        except Exception as e:
            print("[warn] SMPL 준비 실패 → NB4/NB5 메쉬 그림 생략:", e)
    print(f"[viz] session {walkcfg.SESSION}  frames {fids}")
    for nb in args.nb:
        print(f"== NB{nb}")
        try:
            {1: nb1, 2: nb2, 3: nb3, 6: nb6, 7: nb7}.get(nb, lambda *a: None)(fids, K, args) if nb not in (4, 5) \
                else (nb4 if nb == 4 else nb5)(fids, K, args, smpl)
        except FileNotFoundError as e:
            print(f"  [skip] NB{nb}: 입력 없음 ({e})")
    print("[saved]", ROOT_OUT)


if __name__ == "__main__":
    main()
