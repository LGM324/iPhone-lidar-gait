"""
viz_video.py — NB 별 움직이는 영상 (세션 전체 프레임, mp4 H.264)

  {SESS}/viz/videos/NB1_raw.mp4    사진 / raw LiDAR 깊이 / LiDAR 신뢰도
  {SESS}/viz/videos/NB2_depth.mp4  사진+마스크 / Sapiens 상대 깊이 / RANSAC 미터 깊이 / 옆 단면(깊이 vs LiDAR)
  {SESS}/viz/videos/NB3_reanchor.mp4  재-anchor 전 / 후 / 차이 / 옆 단면(전·후·LiDAR)
  {SESS}/viz/videos/NB4_4dh.mp4    사진+검출 상자 / 4D-Humans 메쉬 정면 / 측면
  {SESS}/viz/videos/NB5_fit.mp4    좌우 다리 구분 / SMPL 투영 / 옆 단면 / 위에서 본 다리
  {SESS}/viz/videos/NB6_knee.mp4   사진 / far 허벅지 점 수 + 가림 / 최종 무릎각 + 라벨 (커서)
  {SESS}/viz/videos/NB7_imu.mp4    사진 / 오른 무릎 IMU vs SMPL / 왼 무릎 (커서)

사용: python viz_video.py --nb 1 2 3 [--fps 15] [--step 1]
  --fps 15 = 실제 속도의 절반 (30 fps 촬영)
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walkcfg  # noqa: E402

OUT = f"{walkcfg.SESS}/viz/videos"
PH = 640
SIDE_BGR = np.array([[150, 150, 150], [230, 90, 40], [40, 40, 220]], np.uint8)   # upper / far(blue) / near(red)
LEG_R = ("rightUpLeg", "rightLeg", "rightFoot", "rightToeBase")
LEG_L = ("leftUpLeg", "leftLeg", "leftFoot", "leftToeBase")


# ------------------------------------------------------------------ 공용
def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class Writer:
    def __init__(self, path, w, h, fps):
        self.path, self.w, self.h = path, w, h
        ff = ffmpeg_exe()
        if ff:
            self.p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                       "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                       "-pix_fmt", "yuv420p", "-crf", "24", path], stdin=subprocess.PIPE)
            self.vw = None
        else:
            self.p = None
            self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, img):
        if img.shape[:2] != (self.h, self.w):
            img = cv2.resize(img, (self.w, self.h))
        (self.p.stdin.write(img.tobytes()) if self.p else self.vw.write(img))

    def close(self):
        if self.p:
            self.p.stdin.close(); self.p.wait()
        else:
            self.vw.release()
        print("  saved", os.path.relpath(self.path, walkcfg.SESS))


class Video:
    """첫 프레임 크기로 writer 생성."""
    def __init__(self, name, fps):
        self.path, self.fps, self.w = f"{OUT}/{name}", fps, None

    def add(self, row):
        if self.w is None:
            self.w = Writer(self.path, even(row.shape[1]), even(row.shape[0]), self.fps)
        self.w.write(row)

    def close(self):
        if self.w:
            self.w.close()


def imread_bgr(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def load_npz(p):
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def even(x):
    return int(x) // 2 * 2


def fit_h(img, h=PH):
    s = h / img.shape[0]
    return cv2.resize(img, (max(2, even(img.shape[1] * s)), h), interpolation=cv2.INTER_AREA)


def label(img, text, sub=None):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30 if sub is None else 52), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (6, 44), cv2.FONT_HERSHEY_SIMPLEX, .5, (210, 210, 210), 1, cv2.LINE_AA)
    return img


def colorize(z, mask, lo, hi, cmap=cv2.COLORMAP_TURBO):
    n = np.clip((z - lo) / (hi - lo + 1e-9), 0, 1)
    c = cv2.applyColorMap((n * 255).astype(np.uint8), cmap)
    c[~mask] = 255
    return c


def colorbar(lo, hi, unit, h=PH, cmap=cv2.COLORMAP_TURBO, fmt="{:.2f}"):
    bar = cv2.applyColorMap(np.linspace(255, 0, h - 80).astype(np.uint8)[:, None].repeat(18, 1), cmap)
    img = np.full((h, 74, 3), 255, np.uint8)
    img[40:h - 40, 4:22] = bar
    for frac, val in ((0, hi), (.5, (lo + hi) / 2), (1, lo)):
        y = int(40 + frac * (h - 81))
        cv2.putText(img, fmt.format(val), (24, y + 4), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, unit, (24, 30), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def canvas(layers, bounds, size, flip_y=False, grid=0.1):
    """layers = [(xy(N,2), bgr, radius)], bounds = (x0, x1, y0, y1) m."""
    W, H = size
    img = np.full((H, W, 3), 255, np.uint8)
    x0, x1, y0, y1 = bounds
    if grid:
        for g in np.arange(np.ceil(x0 / grid) * grid, x1, grid):
            X = int((g - x0) / (x1 - x0) * (W - 1)); cv2.line(img, (X, 0), (X, H), (235, 235, 235), 1)
        for g in np.arange(np.ceil(y0 / grid) * grid, y1, grid):
            Y = int((g - y0) / (y1 - y0) * (H - 1)); Y = H - 1 - Y if flip_y else Y
            cv2.line(img, (0, Y), (W, Y), (235, 235, 235), 1)
    for pts, col, r in layers:
        if len(pts) == 0:
            continue
        X = ((pts[:, 0] - x0) / (x1 - x0) * (W - 1)).astype(int)
        Y = ((pts[:, 1] - y0) / (y1 - y0) * (H - 1)).astype(int)
        if flip_y:
            Y = H - 1 - Y
        ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        cols = np.broadcast_to(np.asarray(col, np.uint8), (len(pts), 3))[ok]
        X, Y = X[ok], Y[ok]
        if r <= 1:
            img[Y, X] = cols
        else:
            for x, y, c in zip(X, Y, cols):
                cv2.circle(img, (int(x), int(y)), r, tuple(int(v) for v in c), -1)
    return img


def backproject(z, K, stride=1, mask=None):
    sel = (z > 0) if mask is None else (mask & (z > 0))
    if stride > 1:
        sub = np.zeros_like(sel); sub[::stride, ::stride] = True; sel &= sub
    v, u = np.nonzero(sel)
    zz = z[v, u]
    return np.stack([(u - K["cx"]) / K["fx"] * zz, (v - K["cy"]) / K["fy"] * zz, zz], 1)


def lidar_person(fid, person, K):
    s = f"{fid:05d}"
    lid = np.load(f"{walkcfg.EXTRACT_DIR}/depth_npy/{s}.npy").astype(np.float32)
    conf = np.load(f"{walkcfg.EXTRACT_DIR}/conf_npy/{s}.npy")
    H, W = person.shape; h, w = lid.shape; sx, sy = w / W, h / H
    pm = cv2.resize(person.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    pm = cv2.erode(pm, np.ones((3, 3), np.uint8)).astype(bool)
    Kl = dict(fx=K["fx"] * sx, fy=K["fy"] * sy, cx=K["cx"] * sx, cy=K["cy"] * sy)
    return backproject(lid, Kl, mask=pm & (conf >= 2))


def plot_strip(builder, fids, size=(7.2, 3.1)):
    """matplotlib 그림 1장을 이미지로 + fid→x픽셀 매핑 (커서용)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size, dpi=100)
    builder(ax)
    ax.set_xlim(fids[0], fids[-1])
    fig.tight_layout(); fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR).copy()
    xs = {int(f): int(ax.transData.transform((f, 0))[0]) for f in fids}
    plt.close(fig)
    return img, xs


def with_cursor(strip, fid):
    img, xs = strip
    im = img.copy()
    if fid in xs:
        cv2.line(im, (xs[fid], 0), (xs[fid], im.shape[0]), (0, 160, 0), 2)
    return im


class Ctx:
    def __init__(self):
        self.K = json.load(open(walkcfg.CAM_INTR))
        self.rgb_fids = sorted(int(re.search(r"(\d+)\.jpg$", p).group(1)) for p in glob.glob(f"{walkcfg.EXTRACT_DIR}/rgb/*.jpg"))
        self.depth_fids = sorted(int(re.search(r"_(\d+)_person_metric\.npy$", p).group(1))
                                 for p in glob.glob(f"{walkcfg.DEPTH_SEQ}/WALK_side_*_person_metric.npy"))
        self.box = None
        if self.depth_fids:
            v = load_npz(f"{walkcfg.VIDEO_SEQ}/WALK_side_{self.depth_fids[0]:05d}.npz")
            z0 = np.load(f"{walkcfg.DEPTH_SEQ}/WALK_side_{self.depth_fids[0]:05d}_person_metric.npy")
            H, W = z0.shape
            if v is not None:
                x0, y0, x1, y1 = [int(t) for t in v["bbox"]]          # NB2 공통 crop 상자
            else:
                ys, xs = np.where(z0 > 0); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            pad = 40
            self.box = (max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad))

    def crop(self, img):
        if self.box is None:
            return img
        y0, y1, x0, x1 = self.box
        return img[y0:y1, x0:x1]


def smpl_model():
    import torch
    import smplx
    m = smplx.create(walkcfg.SMPL_MODEL, model_type="smpl", gender="neutral", num_betas=10)
    betas = torch.tensor(np.load(walkcfg.FROZEN_NPZ)["betas"][:10].astype(np.float32)).reshape(1, 10)
    seg = json.load(open(walkcfg.SEG_JSON))
    col = np.full((6890, 3), 200, np.uint8)
    legR = np.zeros(6890, bool); legL = np.zeros(6890, bool)
    for p in LEG_R:
        legR[seg.get(p, [])] = True
    for p in LEG_L:
        legL[seg.get(p, [])] = True
    col[legR] = (40, 40, 220); col[legL] = (230, 90, 40)

    def verts(go, bp, tr=None):
        with torch.no_grad():
            return m(betas=betas, body_pose=torch.tensor(np.asarray(bp, np.float32)).reshape(1, 69),
                     global_orient=torch.tensor(np.asarray(go, np.float32)).reshape(1, 3),
                     transl=torch.tensor(np.zeros(3) if tr is None else np.asarray(tr), dtype=torch.float32).reshape(1, 3)
                     ).vertices[0].numpy()
    return verts, col, legR, legL


# ------------------------------------------------------------------ NB1
def nb1(c, a):
    v = Video("NB1_raw.mp4", a.fps)
    for i, fid in enumerate(c.rgb_fids[::a.step]):
        s = f"{fid:05d}"
        rgb = imread_bgr(f"{walkcfg.EXTRACT_DIR}/rgb/{s}.jpg")
        lid = np.load(f"{walkcfg.EXTRACT_DIR}/depth_npy/{s}.npy")
        conf = np.load(f"{walkcfg.EXTRACT_DIR}/conf_npy/{s}.npy")
        p2 = colorize(lid, lid > 0, 0.5, 5.0)
        p3 = cv2.applyColorMap((conf.astype(np.float32) / 2 * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        row = np.hstack([label(fit_h(rgb), f"RGB  fid {fid}"),
                         label(fit_h(p2), "raw LiDAR depth", "0.5-5 m"), colorbar(0.5, 5.0, "m"),
                         label(fit_h(p3), "LiDAR confidence", f"conf2 {100 * (conf == 2).mean():.0f}%")])
        v.add(row)
    v.close()


# ------------------------------------------------------------------ NB2
def nb2(c, a):
    K = c.K
    samp = [np.load(f"{walkcfg.DEPTH_SEQ}/WALK_side_{f:05d}_person_metric.npy") for f in c.depth_fids[::50]]
    lo, hi = np.percentile(np.concatenate([z[z > 0] for z in samp]), [1, 99])
    P0 = backproject(samp[0], K, 4)
    zc, yc = np.median(P0[:, 2]), np.median(P0[:, 1])
    sb = (zc - 0.45, zc + 0.55, -yc - 1.0, -yc + 1.0)
    v = Video("NB2_depth.mp4", a.fps)
    for i, fid in enumerate(c.depth_fids[::a.step]):
        s = f"{fid:05d}"
        bak = f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy.bak"
        z = np.load(bak if os.path.exists(bak) else f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy")
        m = z > 0
        rgb = imread_bgr(f"{walkcfg.EXTRACT_DIR}/rgb/{s}.jpg")
        vid = load_npz(f"{walkcfg.VIDEO_SEQ}/WALK_side_{s}.npz")
        p1 = c.crop(rgb).copy()
        cnt, _ = cv2.findContours(c.crop(m.astype(np.uint8)).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p1, cnt, -1, (0, 220, 0), 2)
        sub2 = None
        if vid is not None:
            sr = vid["s_relative"].astype(np.float32); mc = vid["mask_crop"].astype(bool)
            slo, shi = np.percentile(sr[mc], [1, 99])
            p2 = colorize(sr, mc, slo, shi, cv2.COLORMAP_MAGMA)
            sub2 = f"a={float(vid['a_raw']):.3f} b={float(vid['b_raw']):.2f} inlier {float(vid['inlier']) * 100:.0f}%"
        else:
            p2 = np.full_like(p1, 255)
        p3 = c.crop(colorize(z, m, lo, hi))
        P = backproject(z, K, 3); L = lidar_person(fid, m, K)
        p4 = canvas([(np.c_[P[:, 2], -P[:, 1]], (200, 120, 40), 1), (np.c_[L[:, 2], -L[:, 1]], (0, 0, 230), 2)],
                    sb, (420, PH), flip_y=True)
        row = np.hstack([label(fit_h(p1), f"RGB + Sapiens mask  fid {fid}"),
                         label(fit_h(p2), "Sapiens relative depth s", sub2),
                         label(fit_h(p3), "NB2 RANSAC metric depth"), colorbar(lo, hi, "m"),
                         label(p4, "side view (Z, -Y)", "blue = NB2 depth, red = LiDAR conf2")])
        v.add(row)
        if i % 50 == 0:
            print(f"  NB2 {i}/{len(c.depth_fids[::a.step])}")
    v.close()


# ------------------------------------------------------------------ NB3
def nb3(c, a):
    K = c.K
    samp = [np.load(f"{walkcfg.DEPTH_SEQ}/WALK_side_{f:05d}_person_metric.npy") for f in c.depth_fids[::50]]
    lo, hi = np.percentile(np.concatenate([z[z > 0] for z in samp]), [1, 99])
    P0 = backproject(samp[0], K, 4)
    zc, yc = np.median(P0[:, 2]), np.median(P0[:, 1])
    sb = (zc - 0.45, zc + 0.55, -yc - 1.0, -yc + 1.0)
    D = 3.0
    v = Video("NB3_reanchor.mp4", a.fps)
    for i, fid in enumerate(c.depth_fids[::a.step]):
        s = f"{fid:05d}"
        z = np.load(f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy")
        bak = f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy.bak"
        zb = np.load(bak) if os.path.exists(bak) else z
        m = z > 0
        diff = np.where(m & (zb > 0), (z - zb) * 100, 0)
        dimg = colorize(diff, m & (zb > 0), -D, D, cv2.COLORMAP_JET)
        P = backproject(z, K, 3); Pb = backproject(zb, K, 3); L = lidar_person(fid, m, K)
        p4 = canvas([(np.c_[Pb[:, 2], -Pb[:, 1]], (170, 170, 170), 1), (np.c_[P[:, 2], -P[:, 1]], (200, 120, 40), 1),
                     (np.c_[L[:, 2], -L[:, 1]], (0, 0, 230), 2)], sb, (420, PH), flip_y=True)
        md = float(np.median(diff[m & (zb > 0)])) if (m & (zb > 0)).any() else 0
        row = np.hstack([label(fit_h(c.crop(colorize(zb, zb > 0, lo, hi))), f"before (NB2)  fid {fid}"),
                         label(fit_h(c.crop(colorize(z, m, lo, hi))), "after (NB3)"), colorbar(lo, hi, "m"),
                         label(fit_h(c.crop(dimg)), "after - before", f"median {md:+.2f} cm"),
                         colorbar(-D, D, "cm", cmap=cv2.COLORMAP_JET, fmt="{:+.1f}"),
                         label(p4, "side view (Z, -Y)", "gray before / blue after / red LiDAR")])
        v.add(row)
        if i % 50 == 0:
            print(f"  NB3 {i}/{len(c.depth_fids[::a.step])}")
    v.close()


# ------------------------------------------------------------------ NB4
def nb4(c, a):
    fdh = load_npz(walkcfg.FOURDH_SEQ)
    if fdh is None:
        print("  [skip] NB4 결과 없음"); return
    verts, col, _, _ = smpl_model()
    f = np.asarray(fdh["fids"]).astype(int)
    bp = np.asarray(fdh["body_pose_aa"]).reshape(len(f), -1)
    V0 = verts(fdh["global_orient_aa"][0], bp[0])
    cx, cy, cz = np.median(V0, 0)
    fb = (cx - 0.7, cx + 0.7, -cy - 1.0, -cy + 1.0)
    sbx = (cz - 0.7, cz + 0.7, -cy - 1.0, -cy + 1.0)
    v = Video("NB4_4dh.mp4", a.fps)
    for i in range(0, len(f), a.step):
        fid = int(f[i])
        rgb = imread_bgr(f"{walkcfg.EXTRACT_DIR}/rgb/{fid:05d}.jpg")
        x0, y0, x1, y1 = [int(t) for t in fdh["bbox_xyxy"][i]]
        cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 220, 255), 4)
        V = verts(fdh["global_orient_aa"][i], bp[i])
        p2 = canvas([(np.c_[V[:, 0], -V[:, 1]], col, 1)], fb, (440, PH), flip_y=True)
        p3 = canvas([(np.c_[V[:, 2], -V[:, 1]], col, 1)], sbx, (440, PH), flip_y=True)
        row = np.hstack([label(fit_h(c.crop(rgb)), f"detection  fid {fid}", f"score {float(fdh['detector_score'][i]):.2f}"),
                         label(p2, "4D-Humans pose: camera view", "red = right leg, blue = left leg"),
                         label(p3, "4D-Humans pose: side view (Z, -Y)")])
        v.add(row)
        if i % 50 == 0:
            print(f"  NB4 {i}/{len(f)}")
    v.close()


# ------------------------------------------------------------------ NB5
def nb5(c, a):
    from a0_align_v2 import backproject as bp_a0, side_from_zorder, SIDE_UPPER
    from knee_measurement_basis import assemble_body_pose
    K = c.K
    a0, near, far, fdh = load_npz(walkcfg.A0_NPZ), load_npz(walkcfg.NEAR_F3), load_npz(walkcfg.FAR_F2), load_npz(walkcfg.FOURDH_SEQ)
    if a0 is None or fdh is None:
        print("  [skip] NB5 결과 없음"); return
    verts, vcol, legR, legL = smpl_model()
    bp_all = np.asarray(fdh["body_pose_aa"]).reshape(len(fdh["fids"]), -1)
    rows = lambda arr: {int(x): i for i, x in enumerate(np.asarray(arr).astype(int))}
    r_a0, r_fdh = rows(a0["frame"]), rows(fdh["fids"])
    r_nr = rows(near["fid"]) if near is not None else {}
    r_fr = rows(far["fid"]) if far is not None else {}
    z0 = np.load(f"{walkcfg.DEPTH_SEQ}/WALK_side_{c.depth_fids[0]:05d}_person_metric.npy")
    P0 = backproject(z0, K, 4)
    zc, yc, xc = np.median(P0[:, 2]), np.median(P0[:, 1]), np.median(P0[:, 0])
    sb = (zc - 0.45, zc + 0.55, -yc - 1.0, -yc + 1.0)
    tb = (xc - 0.55, xc + 0.55, zc - 0.30, zc + 0.50)
    v = Video("NB5_fit.mp4", a.fps)
    y0c, _, x0c, _ = c.box
    for i, fid in enumerate(c.depth_fids[::a.step]):
        if fid not in r_a0 or fid not in r_fdh:
            continue
        s = f"{fid:05d}"
        rgb = imread_bgr(f"{walkcfg.EXTRACT_DIR}/rgb/{s}.jpg")
        z = np.load(f"{walkcfg.DEPTH_SEQ}/WALK_side_{s}_person_metric.npy")
        pts, pix = bp_a0(z, K)
        side = side_from_zorder(pts)
        pv, pu = np.unravel_index(pix, z.shape)
        base = bp_all[r_fdh[fid]]
        lab = "?"
        if fid in r_nr:
            ft = far["far_theta_fit"][r_fr[fid]] if fid in r_fr else np.full(9, np.nan)
            ok = bool(far["is_fit"][r_fr[fid]]) if fid in r_fr else False
            lab = str(far["label"][r_fr[fid]]) if fid in r_fr else "?"
            bp = assemble_body_pose(base, near["theta_fit"][r_nr[fid]], ft, ok)
        else:
            bp = base
        k = r_a0[fid]
        V = verts(a0["global_orient"][k], bp, a0["transl"][k])
        p1 = c.crop(rgb).copy()
        okp = (pv - y0c >= 0) & (pv - y0c < p1.shape[0]) & (pu - x0c >= 0) & (pu - x0c < p1.shape[1])
        ov = p1.copy(); ov[pv[okp] - y0c, pu[okp] - x0c] = SIDE_BGR[side[okp]]
        p1 = cv2.addWeighted(p1, .35, ov, .65, 0)
        p2 = c.crop(rgb).copy()
        u = (K["fx"] * V[:, 0] / V[:, 2] + K["cx"] - x0c).astype(int); w = (K["fy"] * V[:, 1] / V[:, 2] + K["cy"] - y0c).astype(int)
        okv = (u >= 0) & (u < p2.shape[1]) & (w >= 0) & (w < p2.shape[0])
        for uu, ww, cc in zip(u[okv][::2], w[okv][::2], vcol[okv][::2]):
            cv2.circle(p2, (int(uu), int(ww)), 2, tuple(int(q) for q in cc), -1)
        ps = pts[::3]
        p3 = canvas([(np.c_[ps[:, 2], -ps[:, 1]], (170, 170, 170), 1), (np.c_[V[:, 2], -V[:, 1]], vcol, 1)], sb, (420, PH), flip_y=True)
        low = side != SIDE_UPPER
        pl, sl = pts[low][::2], side[low][::2]
        p4 = canvas([(np.c_[pl[:, 0], pl[:, 2]], SIDE_BGR[sl], 1), (np.c_[V[legR, 0], V[legR, 2]], (0, 0, 120), 2),
                     (np.c_[V[legL, 0], V[legL, 2]], (120, 40, 0), 2)], tb, (560, 400))
        p4 = np.vstack([p4, np.full((PH - 400, 560, 3), 255, np.uint8)])
        cv2.putText(p4, "camera at top; dots = depth, circles = SMPL legs", (8, 420), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)
        row = np.hstack([label(fit_h(p1), f"L/R split  fid {fid}", f"near(R red) {(side == 2).sum()}  far(L blue) {(side == 1).sum()}"),
                         label(fit_h(p2), "SMPL fit projected", f"far: {lab}"),
                         label(p3, "side view (Z, -Y)", "gray depth / SMPL"), label(p4, "top view of legs (X, Z)")])
        v.add(row)
        if i % 50 == 0:
            print(f"  NB5 {i}/{len(c.depth_fids[::a.step])}")
    v.close()


# ------------------------------------------------------------------ NB6
def nb6(c, a):
    kf = load_npz(f"{walkcfg.KNEE_DIR}/far_knee_final.npz")
    basis = load_npz(f"{walkcfg.KNEE_DIR}/knee_basis_full.npz")
    gate = load_npz(f"{walkcfg.KNEE_DIR}/far_thigh_gate_full.npz")
    if kf is None:
        print("  [skip] NB6 결과 없음"); return
    f = np.asarray(kf["fid"]).astype(int); lab = kf["label"].astype(str)

    def b1(ax):
        if basis is not None:
            ax.plot(basis["fid"], basis["thigh_far_npts"], c="tab:blue", lw=1.2, label="far thigh pts")
            ax.set_yscale("symlog", linthresh=100)
        if gate is not None:
            ax.axhline(float(gate["N_low"]), c="r", ls="--", lw=1, label=f"N_low {float(gate['N_low']):.0f}")
            for st, ln in zip(gate["start_fid"], gate["length"]):
                ax.axvspan(st - .5, st + ln - .5, color="tab:red", alpha=.15)
        ax.set_title("far thigh observed points (red = collapsed run)"); ax.legend(fontsize=7, loc="upper right")

    def b2(ax):
        ax.plot(f, kf["knee_near"], c="tab:red", lw=1, label="near(R)")
        ax.plot(f, kf["knee_far_final"], c="0.6", lw=.8)
        for nm, cc in (("MEASURED", "tab:blue"), ("INTERPOLATED", "tab:green"), ("INTERPOLATED_HIGH_CURV", "orange")):
            m = lab == nm
            ax.scatter(f[m], kf["knee_far_final"][m], s=9, c=cc, label=f"far {nm.replace('INTERPOLATED', 'INTERP')} ({m.sum()})")
        ax.set_ylabel("deg"); ax.set_title("final knee time series"); ax.legend(fontsize=6, loc="upper right", ncol=2)
    s1, s2 = plot_strip(b1, f), plot_strip(b2, f)
    v = Video("NB6_knee.mp4", a.fps)
    for i, fid in enumerate(f[::a.step]):
        rgb = c.crop(imread_bgr(f"{walkcfg.EXTRACT_DIR}/rgb/{fid:05d}.jpg"))
        right = np.vstack([with_cursor(s1, fid), with_cursor(s2, fid)])
        right = cv2.resize(right, (even(right.shape[1] * PH / right.shape[0]), PH))
        j = np.where(f == fid)[0][0]
        row = np.hstack([label(fit_h(rgb.copy()), f"fid {fid}", f"far: {lab[j]}  {kf['knee_far_final'][j]:.0f} deg"), right])
        v.add(row)
    v.close()


# ------------------------------------------------------------------ NB7
def nb7(c, a):
    r = load_npz(f"{walkcfg.IMU_OUT}/imu_validation.npz")
    if r is None:
        print("  [skip] NB7 결과 없음"); return
    f = np.asarray(r["fid"]).astype(int); lab = r["label"].astype(str)

    def mk(k, nm, color):
        def b(ax):
            ax.plot(f, r[f"imu_{k}"], c="k", lw=1.6, label="IMU")
            ax.plot(f, r[f"smpl_{k}"], c=color, lw=1.3, label="SMPL")
            if k == "L":
                ax.scatter(f[lab != "MEASURED"], r["smpl_L"][lab != "MEASURED"], s=10, facecolors="none", edgecolors="orange", label="interpolated")
            e = r[f"smpl_{k}"] - r[f"imu_{k}"]
            ax.set_title(f"{nm}  RMSE {np.sqrt(np.nanmean(e ** 2)):.2f} deg, bias {np.nanmean(e):+.2f}")
            ax.set_ylabel("deg"); ax.legend(fontsize=7, loc="upper right")
        return b
    sR, sL = plot_strip(mk("R", "right (near) knee", "tab:red"), f), plot_strip(mk("L", "left (far) knee", "tab:blue"), f)
    v = Video("NB7_imu.mp4", a.fps)
    for i, fid in enumerate(f[::a.step]):
        rgb = c.crop(imread_bgr(f"{walkcfg.EXTRACT_DIR}/rgb/{fid:05d}.jpg"))
        right = np.vstack([with_cursor(sR, fid), with_cursor(sL, fid)])
        right = cv2.resize(right, (even(right.shape[1] * PH / right.shape[0]), PH))
        j = np.where(f == fid)[0][0]
        sub = f"R: SMPL {r['smpl_R'][j]:.0f} / IMU {r['imu_R'][j]:.0f}   L: SMPL {r['smpl_L'][j]:.0f} / IMU {r['imu_L'][j]:.0f}"
        row = np.hstack([label(fit_h(rgb.copy()), f"fid {fid}  far: {lab[j]}", sub), right])
        v.add(row)
    v.close()


FUNCS = {1: nb1, 2: nb2, 3: nb3, 4: nb4, 5: nb5, 6: nb6, 7: nb7}
OLD = {"depth": [2, 3], "fit": [5], "knee": [7]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, nargs="*", default=None, help="1~7")
    ap.add_argument("--only", nargs="*", default=None, help="(이전 호환) depth fit knee")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    nbs = a.nb or sorted({n for k in (a.only or []) for n in OLD.get(k, [])}) or [1, 2, 3, 4, 5, 6, 7]
    os.makedirs(OUT, exist_ok=True)
    c = Ctx()
    print(f"[video] {walkcfg.SESSION}: rgb {len(c.rgb_fids)} f, depth {len(c.depth_fids)} f, fps {a.fps}, NB {nbs}")
    for n in nbs:
        print(f"== NB{n}")
        try:
            FUNCS[n](c, a)
        except FileNotFoundError as e:
            print(f"  [skip] NB{n}: 입력 없음 ({e})")


if __name__ == "__main__":
    main()
