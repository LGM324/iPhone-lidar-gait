"""
extract_session.py — Record3D .r3d → 세션 입력 (NB1)

1) --scan   : 전체 프레임을 훑어(깊이만 디코딩) meta.json + motion_envelope.npz + 미리보기 생성
              motion = 0.3~4m 범위 LiDAR depth 의 프레임간 평균 변화 (IMU 동기화 coarse 신호)
2) --start S --n N : 프레임 S..S+N-1 만 00_extract/ 에 추출 (rgb jpg 원본 그대로, depth/conf npy)

출력 ({SESS}/00_extract/):
  meta.json            fps, n_frames, 해상도, r3d 이름/시작시각, 추출 window
  motion_envelope.npz  fid, t(s), motion
  scan_preview.png     motion 곡선 + 대표 프레임 썸네일
  rgb/{fid:05d}.jpg  depth_npy/{fid:05d}.npy (float32 m)  conf_npy/{fid:05d}.npy (uint8)
  intrinsics.json      {fx,fy,cx,cy,...}  (window 첫 프레임 perFrameIntrinsicCoeffs, 없으면 K)

원본 Extract_r3d.py 와 같은 디코딩 (lzfse → float32 / uint8, depth 256x192).
"""
import argparse
import io
import json
import os
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walkcfg  # noqa: E402

try:
    import lzfse
except ImportError:
    raise SystemExit("pip install lzfse 먼저")


def dec(raw, dtype, hw):
    n = hw[0] * hw[1]
    for buf in (lambda: lzfse.decompress(raw), lambda: raw):
        try:
            a = np.frombuffer(buf(), dtype=dtype)
            if a.size >= n:
                return a[:n].reshape(hw).copy()
        except Exception:
            pass
    return None


def open_r3d(path):
    z = zipfile.ZipFile(path)
    meta = json.load(z.open("metadata"))
    names = set(z.namelist())
    fids = sorted(int(m.group(1)) for n in names if (m := re.match(r"rgbd/(\d+)\.jpg$", n)))
    hw = (int(meta.get("dh", 256)), int(meta.get("dw", 192)))
    return z, meta, names, fids, hw


def frame_times(meta, fids):
    fps = float(meta.get("fps", 60))
    ts = meta.get("frameTimestamps")
    if ts and len(ts) >= len(fids):
        t = np.array([ts[f] for f in fids], float)
        return t - t[0], fps, "frameTimestamps"
    return np.array(fids, float) / fps, fps, "fid/fps"


def intrinsics(meta, fid):
    pf = meta.get("perFrameIntrinsicCoeffs")
    if pf and len(pf) > fid and len(pf[fid]) >= 4:
        r = pf[fid]
        return dict(fx=float(r[0]), fy=float(r[1]), cx=float(r[2]), cy=float(r[3]),
                    source=f"perFrameIntrinsicCoeffs[{fid}]")
    K = meta["K"]
    return dict(fx=float(K[0]), fy=float(K[4]), cx=float(K[6]), cy=float(K[7]), source="K_3x3")


def scan(r3d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    z, meta, names, fids, hw = open_r3d(r3d)
    t, fps, tsrc = frame_times(meta, fids)
    motion = np.zeros(len(fids), np.float32)
    prev = None
    for i, f in enumerate(fids):
        d = dec(z.read(f"rgbd/{f}.depth"), np.float32, hw) if f"rgbd/{f}.depth" in names else None
        if d is None:
            motion[i] = np.nan
            continue
        if prev is not None:
            m = (d > 0.3) & (d < 4.0) & (prev > 0.3) & (prev < 4.0)
            motion[i] = float(np.abs(d - prev)[m].mean()) if m.any() else np.nan
        prev = d
        if i % 1000 == 0:
            print(f"  scan {i}/{len(fids)}")
    motion = np.nan_to_num(motion, nan=float(np.nanmedian(motion)))
    name = os.path.basename(r3d)
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})--(\d{2})-(\d{2})-(\d{2})", name)
    info = dict(r3d=name, r3d_start_local=(f"{m.group(1)}-{m.group(2)}-{m.group(3)} "
                                          f"{m.group(4)}:{m.group(5)}:{m.group(6)}") if m else None,
                fps=fps, time_source=tsrc, n_frames=len(fids), fid_first=fids[0], fid_last=fids[-1],
                duration_s=float(t[-1]), rgb_wh=[meta.get("w"), meta.get("h")], depth_hw=list(hw),
                metadata_keys=sorted(meta.keys()))
    os.makedirs(out, exist_ok=True)
    mp = f"{out}/meta.json"
    if os.path.exists(mp):                      # 재스캔해도 기존 추출 window 기록 유지
        old = json.load(open(mp, encoding="utf-8"))
        if "window" in old:
            info["window"] = old["window"]
    json.dump(info, open(mp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    np.savez(f"{out}/motion_envelope.npz", fid=np.array(fids), t=t, motion=motion)

    # 미리보기: motion 곡선 + 10장 썸네일
    picks = np.linspace(0, len(fids) - 1, 10).astype(int)
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t, motion, lw=.5)
    k = max(1, int(fps))
    ax.plot(t, np.convolve(motion, np.ones(k) / k, mode="same"), "r", lw=1.2, label="1s mean")
    for p in picks:
        ax.axvline(t[p], c="gray", ls=":", lw=.8)
    ax.set_xlabel("video t (s)"); ax.set_ylabel("depth motion (m, 0.3-4m band)")
    ax.set_title(f"{name}  {len(fids)}f @ {fps:.2f}fps  {t[-1]:.1f}s")
    ax.legend()
    for j, p in enumerate(picks):
        im = Image.open(io.BytesIO(z.read(f"rgbd/{fids[p]}.jpg")))
        im.thumbnail((240, 320))
        a = fig.add_subplot(2, 10, 11 + j); a.imshow(im); a.axis("off")
        a.set_title(f"fid {fids[p]}\n{t[p]:.1f}s", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/scan_preview.png", dpi=100); plt.close(fig)
    print(json.dumps({k: info[k] for k in ("r3d", "r3d_start_local", "fps", "n_frames", "duration_s"
                                            )}, ensure_ascii=False))
    print(f"[saved] {out}/meta.json, motion_envelope.npz, scan_preview.png")


def extract(r3d, out, start, n, step=1, fid_list=None):
    """fid_list 가 있으면 그 프레임들만 (정지 프레임 골라 뽑기용). step=2 → 2프레임마다 1장 (60fps 영상을 30fps 로 맞춤. 프레임 수 기준 파라미터가 30fps 에서 검증됨)."""
    z, meta, names, fids, hw = open_r3d(r3d)
    fset = set(fids)
    want = sorted(fid_list) if fid_list else list(range(start, start + n * step, step))
    sel = [f for f in want if f in fset]
    if len(sel) != len(want):
        print(f"[WARN] 요청 {len(want)}f 중 {len(sel)}f 만 존재")
    import shutil
    for d in ("rgb", "depth_npy", "conf_npy"):
        shutil.rmtree(f"{out}/{d}", ignore_errors=True)   # 이전 window 프레임이 섞이지 않게
        os.makedirs(f"{out}/{d}", exist_ok=True)
    K = intrinsics(meta, sel[0])
    json.dump(K, open(f"{out}/intrinsics.json", "w"), indent=2)
    nd = nc = 0
    for i, f in enumerate(sel):
        s = f"{f:05d}"
        with open(f"{out}/rgb/{s}.jpg", "wb") as fo:
            fo.write(z.read(f"rgbd/{f}.jpg"))                  # 재압축 없이 원본 바이트
        if f"rgbd/{f}.depth" in names:
            d = dec(z.read(f"rgbd/{f}.depth"), np.float32, hw)
            if d is not None:
                np.save(f"{out}/depth_npy/{s}.npy", d); nd += 1
        if f"rgbd/{f}.conf" in names:
            c = dec(z.read(f"rgbd/{f}.conf"), np.uint8, hw)
            if c is not None:
                np.save(f"{out}/conf_npy/{s}.npy", c); nc += 1
        if i % 50 == 0:
            print(f"  extract {i + 1}/{len(sel)}  fid {f}")
    mp = f"{out}/meta.json"
    info = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
    info["window"] = dict(start=sel[0], end=sel[-1], n=len(sel), n_depth=nd, n_conf=nc, step=step,
                          fid_list=sel if fid_list else None,
                          fps_eff=float(meta.get("fps", 30)) / step)
    json.dump(info, open(mp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[extract] fid {sel[0]}~{sel[-1]} step {step} ({len(sel)}f, {float(meta.get('fps', 30)) / step:.0f} fps)  depth={nd} conf={nc}  K={K}")


def preview_video(r3d, out, vfrom=None, vto=None, width=480, name=None):
    """fid / 시각이 찍힌 미리보기 mp4. 추출 window 가 있으면 그 구간에 초록 테두리 + EXTRACTED.
    ffmpeg(H.264, 브라우저·드라이브 재생 가능)이 있으면 사용, 없으면 cv2 mp4v."""
    import shutil
    import subprocess
    import cv2

    z, meta, names, fids, hw = open_r3d(r3d)
    t, fps, _ = frame_times(meta, fids)
    sel = [(f, tt) for f, tt in zip(fids, t)
           if (vfrom is None or f >= vfrom) and (vto is None or f <= vto)]
    mp = f"{out}/meta.json"
    win = (json.load(open(mp, encoding="utf-8")).get("window") if os.path.exists(mp) else None) or {}
    w0, w1 = win.get("start", -1), win.get("end", -2)
    first = cv2.imdecode(np.frombuffer(z.read(f"rgbd/{sel[0][0]}.jpg"), np.uint8), cv2.IMREAD_COLOR)
    H = int(round(first.shape[0] * width / first.shape[1] / 2) * 2)
    os.makedirs(out, exist_ok=True)
    fp = f"{out}/{name or 'preview_fid'}.mp4"
    ff = shutil.which("ffmpeg")
    if ff:
        proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                 "-s", f"{width}x{H}", "-r", f"{fps:.3f}", "-i", "-",
                                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", fp],
                                stdin=subprocess.PIPE)
        put = lambda im: proc.stdin.write(im.tobytes())
    else:
        vw = cv2.VideoWriter(fp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, H))
        put = vw.write
    for i, (f, tt) in enumerate(sel):
        im = cv2.imdecode(np.frombuffer(z.read(f"rgbd/{f}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (width, H), interpolation=cv2.INTER_AREA)
        inwin = w0 <= f <= w1
        txt = f"fid {f:05d}  t={tt:7.2f}s"
        cv2.rectangle(im, (0, 0), (width, 34), (0, 0, 0), -1)
        cv2.putText(im, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if inwin:
            cv2.rectangle(im, (2, 2), (width - 3, H - 3), (0, 220, 0), 6)
            cv2.putText(im, "EXTRACTED", (8, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2,
                        cv2.LINE_AA)
        put(im)
        if i % 1000 == 0:
            print(f"  video {i}/{len(sel)}")
    if ff:
        proc.stdin.close(); proc.wait()
    else:
        vw.release()
    print(f"[video] {fp}  ({len(sel)}f, fid {sel[0][0]}~{sel[-1][0]}, {width}x{H}, "
          f"{'H.264' if ff else 'mp4v'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r3d", required=True)
    ap.add_argument("--out", default=walkcfg.EXTRACT_DIR)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--start", type=int)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--step", type=int, default=1, help="2 = 2프레임마다 1장 (60fps → 30fps)")
    ap.add_argument("--fids", default=None, help="쉼표로 구분한 fid 목록 (있으면 --start/--n/--step 무시)")
    ap.add_argument("--video", action="store_true", help="fid 표시 미리보기 mp4")
    ap.add_argument("--vfrom", type=int, default=None, help="영상 시작 fid (기본 처음)")
    ap.add_argument("--vto", type=int, default=None, help="영상 끝 fid (기본 끝)")
    ap.add_argument("--vwidth", type=int, default=480)
    ap.add_argument("--vname", default=None, help="출력 파일명 (기본 preview_fid)")
    a = ap.parse_args()
    if a.scan:
        scan(a.r3d, a.out)
    if a.fids:
        extract(a.r3d, a.out, None, 0, 1, [int(x) for x in a.fids.split(",") if x.strip()])
    elif a.start is not None:
        extract(a.r3d, a.out, a.start, a.n, a.step)
    if a.video:
        preview_video(a.r3d, a.out, a.vfrom, a.vto, a.vwidth, a.vname)
