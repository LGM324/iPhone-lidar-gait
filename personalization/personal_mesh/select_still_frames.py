"""돌기 촬영에서 정지 프레임 고르기 (.r3d 원본, 30 fps).

프레임 간 RGB 차이(사람 주변 영역, 축소 회색조)를 움직임으로 보고, 5 프레임 이동평균이
하위 PCT 백분위 아래로 MIN_LEN 프레임 이상 이어지는 구간을 "정지"로 판정한다.
각 구간 양 끝 2 프레임을 버리고, 20 프레임 이상이면 5장, 아니면 3장을 고르게 뽑는다.
돌고 있는 순간은 흔들림과 RGB·LiDAR 시점 차이로 마스크와 depth 가 어긋나므로 쓰지 않는다.

usage: python select_still_frames.py R3D START END [--pct 35] [--crop x0 y0 x1 y1]
출력: 쉼표로 구분한 fid 목록 (pipeline/scripts/extract_session.py --fids 에 그대로 넣음)
"""
import argparse, zipfile
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("r3d"); ap.add_argument("start", type=int); ap.add_argument("end", type=int)
ap.add_argument("--pct", type=float, default=35); ap.add_argument("--min_len", type=int, default=10)
ap.add_argument("--crop", type=int, nargs=4, default=[250, 150, 1250, 1800], help="x0 y0 x1 y1 (사람 주변)")
a = ap.parse_args()

z = zipfile.ZipFile(a.r3d); x0, y0, x1, y1 = a.crop
F = list(range(a.start, a.end)); prev = None; mot = []
for f in F:
    g = cv2.imdecode(np.frombuffer(z.read(f"rgbd/{f}.jpg"), np.uint8), 0)
    g = cv2.resize(g[y0:y1, x0:x1], ((x1 - x0) // 4, (y1 - y0) // 4)).astype(np.float32)
    mot.append(np.nan if prev is None else float(np.mean(np.abs(g - prev)))); prev = g
sm = np.convolve(np.nan_to_num(np.array(mot), nan=99), np.ones(5) / 5, "same")
still = sm < np.percentile(sm[5:-5], a.pct)
runs, s = [], None
for i, v in enumerate(still):
    if v and s is None: s = i
    if (not v or i == len(still) - 1) and s is not None:
        e = i if not v else i + 1
        if e - s >= a.min_len: runs.append((F[s], F[e - 1]))
        s = None
out = []
for r0, r1 in runs:
    r0, r1 = r0 + 2, r1 - 2; k = 5 if r1 - r0 >= 20 else 3
    out += sorted(set(np.linspace(r0, r1, k).round().astype(int).tolist()))
print("정지 구간:", runs)
print(",".join(map(str, out)))
