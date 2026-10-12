"""피팅 결과를 RGB 위에 겹쳐 그림 (얼굴 모자이크). usage: python render_fit.py OUT_DIR [data]"""
import sys, numpy as np, cv2, torch
sys.path.insert(0, "E:/walking_r3d/personal_mesh")
from fit_personal import Body, load_stand4, load_session, rd, raster, vnormals, DEV
out = sys.argv[1]; data = sys.argv[2] if len(sys.argv) > 2 else "stand4"
views = load_stand4() if data == "stand4" else load_session(data)
fb = np.load(f"{out}/body_fit.npz"); names = list(map(str, fb["view_names"]))
body = Body(); T = lambda x: torch.tensor(np.asarray(x, np.float32), device=DEV)
d = nrm = None
import os
if os.path.exists(f"{out}/D.npy") and "--noD" not in sys.argv:
    d, nrm = T(np.load(f"{out}/D.npy")), T(np.load(f"{out}/canon_normals.npy"))
with torch.no_grad():
    V, J = body.forward(T(fb["betas"]), T(fb["global_orient"]), T(fb["body_pose"]), T(fb["transl"]), d, nrm)
V, J = V.cpu(), J.cpu()
tiles = []
for k, nm in enumerate(names):
    vw = [v for v in views if v["name"] == nm][0]; m = vw["mask"]
    x0c, y0c, x1c, y1c = vw["crop"]; im = np.ascontiguousarray(rd(vw["rgb"])[y0c:y1c, x0c:x1c])
    ys, xs = np.nonzero(m); top = ys.min(); h = ys.max() - top; band = ys < top + .15 * h
    x0, x1 = xs[band].min() - 30, xs[band].max() + 30; y0, y1 = max(0, top - 30), int(top + .17 * h)
    im[y0:y1, x0:x1] = cv2.resize(cv2.resize(im[y0:y1, x0:x1], (8, 10)), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    s = raster(V[k].numpy(), body.faces.cpu().numpy(), vw["K"], m.shape)
    im[s] = (0.45 * im[s] + 0.55 * np.array([255, 140, 60])).astype(np.uint8)
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cnt, -1, (0, 230, 255), 3)
    iou = (s & m).sum() / (s | m).sum()
    cv2.putText(im, f"IoU {iou:.2f}", (30, 90), 0, 2.4, (0, 0, 255), 6)
    ys2, xs2 = np.nonzero(m | s)
    tiles.append(cv2.resize(im[max(0, ys2.min() - 60):ys2.max() + 40, max(0, xs2.min() - 200):xs2.max() + 200], (300, 560)))
sel = tiles if len(tiles) <= 8 else [tiles[i] for i in np.linspace(0, len(tiles) - 1, 8).astype(int)]
cv2.imwrite(f"{out}/fit_overlay{'_D' if d is not None else ''}.jpg", np.hstack(sel))
print("saved")
