import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
out = sys.argv[1]
V = []; F = []
for l in open(f"{out}/personal_canonical.obj"):
    if l.startswith("v "): V.append(l.split()[1:4])
    elif l.startswith("f "): F.append(l.split()[1:4])
V = np.array(V, float); F = np.array(F, int) - 1
D = np.load(f"{out}/D.npy") * 100
fig, ax = plt.subplots(1, 3, figsize=(15, 8))
for a, (x, dep, s, t) in zip(ax, [(0, 2, 1, "front"), (2, 0, -1, "side"), (0, 2, -1, "back")]):
    tri = V[F]; zc = tri[:, :, dep].mean(1) * s; o = np.argsort(zc)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    sh = 0.35 + 0.6 * np.abs(n[:, dep]); col = plt.cm.viridis(np.clip(D[F].mean(1)[o] / 3, 0, 1))[:, :3] * sh[o, None]
    xx = tri[o][:, :, x] * (s if x == 0 else 1)
    a.add_collection(PolyCollection(np.stack([xx, tri[o][:, :, 1]], -1), facecolors=col, edgecolors="none"))
    a.set_aspect("equal"); a.autoscale(); a.set_title(t); a.axis("off")
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 3)); fig.colorbar(sm, ax=ax, shrink=.6, label="D (cm, 바깥 변위)")
fig.savefig(f"{out}/canonical_views.png", dpi=70); print("ok")
