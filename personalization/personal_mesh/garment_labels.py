"""SMPL 정점마다 하의(반바지) / 상의 / 기타(피부·신발·머리) 판정.

피팅된 개인 메쉬를 Sapiens 부위 라벨이 저장된 프레임들에 투영하고,
카메라를 향하면서 관측 depth 와 5 cm 안으로 맞는 정점(= 가려지지 않은 정점)만 그 픽셀의 라벨에 투표한다.
  Sapiens goliath: 12 = Lower_Clothing, 22 = Upper_Clothing, 그 밖의 사람 라벨 = 기타
투표가 없는 정점은 canonical 메쉬에서 가장 가까운 투표 정점을 따른다.
손이 반바지 위에 놓인 프레임 때문에 손이 하의로 잡히지 않도록, 하의는 골반·다리 부위 안에서만 인정한다.

SMPL 부위 경계(예: hips 는 티셔츠 아랫단까지 포함)는 실제 옷 경계와 다르므로 고정 부위 대신 이 판정을 쓴다.

usage: python garment_labels.py SESSION_DIR OUT_DIR(=fit_personal 출력) FID [FID ...]
출력: OUT_DIR/garment_class.npy (6890, 0=하의 1=상의 2=기타), OUT_DIR/lower_cloth_vertices.npy (bool)
"""
import sys, json
import numpy as np, cv2, torch
from fit_personal import Body, vnormals, SEG_JSON

sess, out = sys.argv[1], sys.argv[2]; fids = [int(x) for x in sys.argv[3:]]
P = np.load(f"{out}/personal_mesh.npz"); fb = np.load(f"{out}/body_fit.npz"); names = list(map(str, fb["view_names"]))
K = json.load(open(f"{sess}/00_extract/intrinsics.json")); K = K.get("_default", K)
body = Body(); T = lambda x: torch.tensor(np.asarray(x, np.float32))
d, nrm = T(P["D"]), T(P["canon_normals"])
votes = np.zeros((6890, 3))
for f in fids:
    key = f"WALK_side_{f:05d}"
    if key not in names:
        print("건너뜀 (피팅에 없는 프레임):", key); continue
    k = names.index(key)
    with torch.no_grad():
        V, _ = body.forward(T(fb["betas"]), T(fb["global_orient"][k:k + 1]), T(fb["body_pose"][k:k + 1]),
                            T(fb["transl"][k:k + 1]), d, nrm)
        N = vnormals(V, body.faces)[0].cpu().numpy(); V = V[0].cpu().numpy()
    lab = cv2.imdecode(np.fromfile(f"{sess}/01_depth/work/mask/{key}_seg_label.png", np.uint8), -1)
    dep = np.load(f"{sess}/01_depth/work/out/{key}_person_metric.npy")
    u = np.round(K["fx"] * V[:, 0] / V[:, 2] + K["cx"]).astype(int); w = np.round(K["fy"] * V[:, 1] / V[:, 2] + K["cy"]).astype(int)
    ok = (u >= 0) & (u < lab.shape[1]) & (w >= 0) & (w < lab.shape[0]) & ((N * V).sum(1) < 0)
    zo = np.zeros(6890); zo[ok] = dep[w[ok], u[ok]]; ok &= (zo > 0) & (np.abs(V[:, 2] - zo) < 0.05)
    L = np.full(6890, -1); L[ok] = lab[w[ok], u[ok]]
    votes[:, 0] += (L == 12); votes[:, 1] += (L == 22); votes[:, 2] += ok & (L != 12) & (L != 22) & (L > 0)
    print(key, "보이는 정점", int(ok.sum()))
has = votes.sum(1) > 0; cls = np.full(6890, -1); cls[has] = votes[has].argmax(1)
with torch.no_grad():
    Vc = body.shaped(T(P["betas"]))[0].cpu().numpy()
hv = np.where(has)[0]
for i in np.where(~has)[0]:
    cls[i] = cls[hv[np.argmin(np.linalg.norm(Vc[hv] - Vc[i], axis=1))]]
seg = json.load(open(SEG_JSON))
legs = np.unique(np.concatenate([seg[p] for p in ["hips", "leftUpLeg", "rightUpLeg", "leftLeg", "rightLeg", "spine"]]))
lower = (cls == 0) & np.isin(np.arange(6890), legs)
np.save(f"{out}/garment_class.npy", cls); np.save(f"{out}/lower_cloth_vertices.npy", lower)
print("하의 정점", int(lower.sum()), {p: round(float(lower[seg[p]].mean()), 2) for p in ["hips", "leftUpLeg", "rightUpLeg", "leftLeg"]})
