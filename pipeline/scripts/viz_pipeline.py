"""
viz_pipeline.py — 파이프라인 단계별 시각화 (NB2 → NB3 → NB5) + 시계열 그래프

[프레임별] frame_{fid}.png (2×4 패널) + frame_{fid}_points.ply + frame_{fid}_smpl.ply
  1 raw LiDAR depth (256×192)            2 Sapiens 상대 깊이 s (crop)
  3 RANSAC anchor 직후 metric (NB2, .bak) 4 재-anchor 후 metric (NB3) + 차이
  5 좌우 다리 구분 (side_from_zorder: 상체 회색 / near 빨강 / far 파랑)
  6 SMPL 겹쳐보기 (NB5 결과로 다시 세운 메쉬를 사진에 투영: 오른다리 빨강, 왼다리 파랑)
  7 옆에서 본 단면: depth 점 + SMPL 정점      8 위에서 본 다리: depth 점(좌우색) + SMPL 다리

[시계열] sequence_overview.png
  a 프레임별 a (RANSAC) vs 전역 a       b 몸통 두께 재-anchor 전/후
  c 깊이 중심 z_med 전/후               d A0 잔차 · 정/역방향 차이
  e near/far 다리 점 수 + far 라벨      f near 무릎각 4D vs fit + 잔차
  g far 무릎각 최종 (라벨별)             h IMU 대조 (NB7 결과 있으면)

출력: {SESS}/viz/
사용: python viz_pipeline.py [--fids 5660 5750] [--n 4] [--no-frames] [--no-seq]
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

OUT = f"{walkcfg.SESS}/viz"
LEG_PARTS = {"R": ("rightUpLeg", "rightLeg", "rightFoot", "rightToeBase"),
             "L": ("leftUpLeg", "leftLeg", "leftFoot", "leftToeBase")}


def imread_rgb(p):
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_npz(p):
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


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
    fc["a"], fc["b"], fc["c"] = faces[:, 0], faces[:, 2], faces[:, 1]     # y,z 뒤집었으니 면 방향 유지
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\nproperty float x\n"
                 "property float y\nproperty float z\nproperty uchar red\nproperty uchar green\n"
                 f"property uchar blue\nelement face {len(faces)}\nproperty list uchar int vertex_indices\n"
                 "end_header\n").encode())
        f.write(v.tobytes()); f.write(fc.tobytes())


# ------------------------------------------------------------------ SMPL
class SmplPoser:
    def __init__(self):
        import torch
        import smplx
        from knee_measurement_basis import assemble_body_pose, smpl_forward
        self.torch, self.assemble, self.forward = torch, assemble_body_pose, smpl_forward
        self.smpl = smplx.create(walkcfg.SMPL_MODEL, model_type="smpl", gender="neutral", num_betas=10)
        self.faces = self.smpl.faces.astype(np.int64)
        self.betas = torch.tensor(np.load(walkcfg.FROZEN_NPZ)["betas"][:10].astype(np.float32)).reshape(1, 10)
        self.a0 = np.load(walkcfg.A0_NPZ)
        self.near = load_npz(walkcfg.NEAR_F3)
        self.far = load_npz(walkcfg.FAR_F2)
        self.fdh = np.load(walkcfg.FOURDH_SEQ, allow_pickle=True)
        bp = np.asarray(self.fdh["body_pose_aa"])
        self.fdh_bp = bp.reshape(bp.shape[0], -1)
        self.rows = lambda arr: {int(f): i for i, f in enumerate(np.asarray(arr).astype(int))}
        self.r_a0, self.r_fdh = self.rows(self.a0["frame"]), self.rows(self.fdh["fids"])
        self.r_nr = self.rows(self.near["fid"]) if self.near is not None else {}
        self.r_fr = self.rows(self.far["fid"]) if self.far is not None else {}
        seg = json.load(open(walkcfg.SEG_JSON))
        self.leg = {k: np.zeros(6890, bool) for k in "RL"}
        for k, parts in LEG_PARTS.items():
            for p in parts:
                self.leg[k][seg.get(p, [])] = True

    def verts(self, fid):
        t = self.torch
        if fid not in self.r_a0 or fid not in self.r_fdh:
            return None
        i = self.r_a0[fid]
        nt = self.near["theta_fit"][self.r_nr[fid]] if fid in self.r_nr else np.full(9, np.nan)
        ft, fit = np.full(9, np.nan), False
        if fid in self.r_fr:
            ft = self.far["far_theta_fit"][self.r_fr[fid]]
            fit = bool(self.far["is_fit"][self.r_fr[fid]])
        base = self.fdh_bp[self.r_fdh[fid]]
        if np.any(np.isnan(nt)):
            bp = base.astype(np.float32)
        else:
            bp = self.assemble(base, nt, ft, fit)
        return self.forward(self.smpl, self.betas,
                            t.tensor(self.a0["global_orient"][i], dtype=t.float32),
                            t.tensor(self.a0["transl"][i], dtype=t.float32),
                            t.tensor(np.asarray(bp, np.float32)))

    def colors(self):
        c = np.full((6890, 3), 230, np.uint8)
        c[self.leg["R"]] = (220, 40, 40)
        c[self.leg["L"]] = (40, 90, 230)
        return c


# ------------------------------------------------------------------ 프레임별
def frame_figure(fid, K, poser, stride):
    from a0_align_v2 import backproject, side_from_zorder, SIDE_UPPER, SIDE_FAR, SIDE_NEAR
    s = f"{fid:05d}"
    rgb = imread_rgb(f"{walkcfg.EXTRACT_DIR}/rgb/{s}.jpg")
    z = np.load(f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy")
    bak = f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy.bak"
    zb = np.load(bak) if os.path.exists(bak) else None
    lid = np.load(f"{walkcfg.EXTRACT_DIR}/depth_npy/{s}.npy").astype(np.float32)
    vid = load_npz(f"{walkcfg.VIDEO_SEQ}/WALK_side_{s}.npz")
    H, W = z.shape
    person = z > 0
    ys, xs = np.where(person)
    pad = 50
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(H, ys.max() + pad), max(0, xs.min() - pad), min(W, xs.max() + pad)
    crop = lambda a: a[y0:y1, x0:x1]

    pts, pix = backproject(z, K)
    side = side_from_zorder(pts)
    pv, pu = np.unravel_index(pix, z.shape)
    side_col = np.array([[150, 150, 150], [40, 90, 230], [220, 40, 40]], np.uint8)   # upper / far / near
    n_near, n_far = int((side == SIDE_NEAR).sum()), int((side == SIDE_FAR).sum())

    V = poser.verts(fid) if poser else None
    fig, ax = plt.subplots(2, 4, figsize=(22, 12))
    a = ax[0, 0]
    im = a.imshow(np.where(lid > 0, lid, np.nan), cmap="turbo")
    pm = cv2.resize(person.astype(np.uint8), lid.shape[::-1], interpolation=cv2.INTER_NEAREST)
    a.contour(pm, levels=[0.5], colors="white", linewidths=0.8)
    plt.colorbar(im, ax=a, fraction=0.046, label="m"); a.set_title("1 raw LiDAR depth (256×192)\nwhite = person mask"); a.axis("off")

    a = ax[0, 1]
    if vid is not None:
        sr = np.where(vid["mask_crop"], vid["s_relative"], np.nan)
        im = a.imshow(sr, cmap="magma"); plt.colorbar(im, ax=a, fraction=0.046, label="relative")
        a.set_title(f"2 Sapiens relative depth s\n(crop, a_raw={float(vid['a_raw']):.3f}, b_raw={float(vid['b_raw']):.2f})")
    a.axis("off")

    vmin, vmax = np.percentile(z[person], [1, 99])
    for a, arr, ttl in ((ax[0, 2], zb, "3 RANSAC anchor metric (NB2, before)"),
                        (ax[0, 3], z, "4 re-anchored metric (NB3, after)")):
        if arr is not None:
            im = a.imshow(crop(np.where(arr > 0, arr, np.nan)), cmap="turbo", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=a, fraction=0.046, label="m")
        a.set_title(ttl); a.axis("off")
    if zb is not None:
        both = person & (zb > 0)
        d = (z - zb)[both] * 100
        ax[0, 3].set_title(f"4 re-anchored metric (NB3, after)\nafter−before: median {np.median(d):+.1f} cm, "
                           f"p5/p95 {np.percentile(d, 5):+.1f}/{np.percentile(d, 95):+.1f}")

    a = ax[1, 0]
    a.imshow(crop(rgb))
    k = slice(None, None, stride)
    a.scatter(pu[k] - x0, pv[k] - y0, s=0.3, c=side_col[side[k]] / 255.0)
    a.set_title(f"5 left/right split (z 2-means)\nnear(R) {n_near} / far(L) {n_far} / upper gray"); a.axis("off")

    a = ax[1, 1]
    a.imshow(crop(rgb))
    if V is not None:
        u = K["fx"] * V[:, 0] / V[:, 2] + K["cx"]; v = K["fy"] * V[:, 1] / V[:, 2] + K["cy"]
        vc = poser.colors() / 255.0
        a.scatter(u - x0, v - y0, s=0.4, c=vc, alpha=0.6)
        a.set_title("6 SMPL (NB5) projected on RGB\nred = right(near) leg, blue = left(far) leg")
    else:
        a.set_title("6 SMPL: NB5 결과 없음")
    a.axis("off")

    a = ax[1, 2]
    ps = pts[::stride]
    a.scatter(ps[:, 2], -ps[:, 1], s=0.3, c="0.6", label="depth (NB3)")
    if V is not None:
        vc = poser.colors() / 255.0
        a.scatter(V[:, 2], -V[:, 1], s=0.5, c=vc, label="SMPL vertices")
    a.set_aspect("equal"); a.set_xlabel("depth Z (m)"); a.set_ylabel("height -Y (m)")
    a.set_title("7 side view: depth + SMPL"); a.legend(markerscale=10, fontsize=8)

    a = ax[1, 3]
    low = side != SIDE_UPPER
    pl = pts[low][::max(1, stride // 2)]; sl = side[low][::max(1, stride // 2)]
    a.scatter(pl[:, 0], pl[:, 2], s=0.4, c=side_col[sl] / 255.0)
    if V is not None:
        for kk, cc in (("R", (0.55, 0, 0)), ("L", (0, 0.15, 0.55))):
            lv = V[poser.leg[kk]]
            a.scatter(lv[:, 0], lv[:, 2], s=1.5, c=[cc], marker="x", label=f"SMPL {kk} leg")
        a.legend(fontsize=8, markerscale=4)
    a.invert_yaxis(); a.set_aspect("equal")
    a.set_xlabel("X (m)"); a.set_ylabel("depth Z (m, camera at top)")
    a.set_title("8 top view of legs: depth (near red / far blue) + SMPL legs (x)")

    fig.suptitle(f"{walkcfg.SESSION}  fid {fid}", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/frame_{s}.png", dpi=80)
    plt.close(fig)

    write_ply_points(f"{OUT}/frame_{s}_points.ply", pts[::2], side_col[side[::2]])
    if V is not None:
        write_ply_mesh(f"{OUT}/frame_{s}_smpl.ply", V, poser.faces, poser.colors())
    print(f"[frame] {fid}: near {n_near} / far {n_far} pts → frame_{s}.png + .ply")


# ------------------------------------------------------------------ 시계열
def sequence_figure():
    A = f"{walkcfg.DEPTH_BASE}/work/analysis"
    gfit = json.load(open(f"{A}/global_const_fit.json", encoding="utf-8")) if os.path.exists(f"{A}/global_const_fit.json") else None
    p4 = json.load(open(f"{A}/phase4_report.json", encoding="utf-8")) if os.path.exists(f"{A}/phase4_report.json") else None
    a0, near, far = load_npz(walkcfg.A0_NPZ), load_npz(walkcfg.NEAR_F3), load_npz(walkcfg.FAR_F2)
    kf = load_npz(f"{walkcfg.KNEE_DIR}/far_knee_final.npz")
    imu = load_npz(f"{walkcfg.IMU_OUT}/imu_validation.npz")
    fidof = lambda key: int(re.search(r"(\d+)$", key).group(1))

    fig, ax = plt.subplots(8, 1, figsize=(15, 26), sharex=True)
    if gfit:
        pf = [r for r in gfit["perframe"] if r.get("status") == "ok"]
        f = [fidof(r["key"]) for r in pf]
        ax[0].plot(f, [r["a_raw"] for r in pf], ".-", ms=3, label="a per frame (NB2 RANSAC)")
        ax[0].plot(f, [r["a_refit"] for r in pf], ".", ms=2, alpha=.5, label="a refit (check)")
        ax[0].axhline(gfit["chosen_global_a"], c="r", ls="--", label=f"global a = {gfit['chosen_global_a']:.3f} (NB3)")
        ax[0].legend(fontsize=8); ax[0].set_ylabel("slope a")
    ax[0].set_title("a  per-frame anchor slope vs single global a")
    if p4:
        pf = p4["per_frame"]; f = [fidof(r["key"]) for r in pf]
        ax[1].plot(f, [r["torso_before_cm"] for r in pf], c="0.6", label="before (NB2)")
        ax[1].plot(f, [r["torso_after_cm"] for r in pf], c="tab:blue", label="after (NB3)")
        ax[1].set_ylabel("cm"); ax[1].legend(fontsize=8)
        ax[2].plot(f, [r["z_med_before"] for r in pf], c="0.6", label="before")
        ax[2].plot(f, [r["z_med_after"] for r in pf], c="tab:blue", ls="--", label="after")
        ax[2].set_ylabel("m"); ax[2].legend(fontsize=8)
    ax[1].set_title("b  torso thickness (near-side relief)")
    ax[2].set_title("c  depth center z_med (re-anchor keeps it)")
    if a0 is not None:
        ax[3].plot(a0["frame"], a0["residual_cm"], c="k", label="A0 residual median (cm)")
        t = ax[3].twinx(); t.plot(a0["frame"], a0["flip_margin"], c="tab:orange", alpha=.6); t.set_ylabel("flip margin", color="tab:orange")
        ax[3].legend(fontsize=8, loc="upper left")
    ax[3].set_title("d  A0 root alignment residual / facing ambiguity")
    if near is not None:
        ax[4].plot(near["fid"], near["n_near_leg"], c="tab:red", label="near(R) leg pts")
    if far is not None:
        ax[4].plot(far["fid"], far["n_far_leg"], c="tab:blue", label="far(L) leg pts")
        ax[4].plot(far["fid"], far["thigh_n"], c="tab:blue", ls=":", label="far thigh pts")
        inf = far["label"].astype(str) != "MEASURED"
        for x in far["fid"][inf]:
            ax[4].axvspan(x - .5, x + .5, color="0.85", lw=0)
        ax[4].set_yscale("symlog", linthresh=100); ax[4].legend(fontsize=8)
    ax[4].set_title("e  left/right leg point counts (gray band = far INFERRED in NB5)")
    if near is not None:
        ax[5].plot(near["fid"], near["knee_4d"], c="0.5", label="4D-Humans init")
        ax[5].plot(near["fid"], near["knee_fit"], c="tab:red", label="depth fit (λ=0)")
        t = ax[5].twinx(); t.plot(near["fid"], near["resid_after_cm"], c="k", lw=.6, alpha=.5); t.set_ylabel("resid cm")
        ax[5].set_ylabel("deg"); ax[5].legend(fontsize=8, loc="upper left")
    ax[5].set_title("f  near(R) knee: 4D init vs depth fit")
    if kf is not None:
        lab = kf["label"].astype(str)
        ax[6].plot(kf["fid"], kf["knee_far_final"], c="0.6", lw=.8)
        for nm, c in (("MEASURED", "tab:blue"), ("INTERPOLATED", "tab:green"), ("INTERPOLATED_HIGH_CURV", "tab:red")):
            m = lab == nm
            ax[6].scatter(kf["fid"][m], kf["knee_far_final"][m], s=10, c=c, label=f"{nm} ({m.sum()})")
        ax[6].set_ylabel("deg"); ax[6].legend(fontsize=8)
    ax[6].set_title("g  far(L) knee final (NB6)")
    if imu is not None:
        ax[7].plot(imu["fid"], imu["imu_R"], c="k", label="IMU R"); ax[7].plot(imu["fid"], imu["smpl_R"], c="tab:red", label="SMPL R")
        ax[7].plot(imu["fid"], imu["imu_L"], c="0.5", label="IMU L"); ax[7].plot(imu["fid"], imu["smpl_L"], c="tab:blue", label="SMPL L")
        ax[7].set_ylabel("deg"); ax[7].legend(fontsize=8, ncol=4)
    ax[7].set_title("h  IMU vs SMPL knee (NB7)"); ax[7].set_xlabel("fid")
    fig.suptitle(f"{walkcfg.SESSION} — pipeline sequence overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sequence_overview.png", dpi=80)
    plt.close(fig)
    print(f"[seq] sequence_overview.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fids", type=int, nargs="*", default=[])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--no-seq", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.no_seq:
        sequence_figure()
    if args.no_frames:
        return
    K = json.load(open(walkcfg.CAM_INTR))
    allf = sorted(int(re.search(r"_(\d+)_person_metric", p).group(1))
                  for p in glob.glob(f"{walkcfg.DEPTH_SEQ}/WALK_side_*_person_metric.npy"))
    fids = list(args.fids)
    if args.n > 0:
        fids += [allf[i] for i in np.linspace(0, len(allf) - 1, args.n).astype(int)]
    try:
        poser = SmplPoser() if os.path.exists(walkcfg.A0_NPZ) else None
    except Exception as e:
        print("[warn] SMPL 준비 실패 → SMPL 패널 생략:", e)
        poser = None
    for fid in sorted(set(fids)):
        frame_figure(fid, K, poser, args.stride)
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
