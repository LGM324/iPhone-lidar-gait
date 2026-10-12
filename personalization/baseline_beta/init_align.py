"""II-8 init 의 transl 을 방향마다 다시 정렬 (II-10 Stage A 의 이동 한도 안으로 들어오게).
x, y: 투영한 SMPL 범위(bbox) 중심/머리 꼭대기를 사람 마스크에 맞춤.  z: SMPL 앞면(카메라 쪽 5%)을 depth 앞면 5% 에 맞춤.
좌표: 피팅 코드와 같은 y-flip 프레임 (관측 (x,-y,z), project_y_up).

사용: python init_align.py SRC_init_params.npz DST_init_params.npz --work STAND_WORK --smpl-model-dir MODELS
  STAND_WORK = S1 노트북 산출 폴더 (mask/{view}_seg_label.png, out/{view}_person_metric.npy)
"""
import argparse
import numpy as np, cv2, torch, smplx

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--work", required=True)
ap.add_argument("--smpl-model-dir", required=True)
args = ap.parse_args()
W = args.work

A = dict(np.load(args.src, allow_pickle=True)); names = [str(n) for n in A["view_names"]]
sm = smplx.create(args.smpl_model_dir, model_type="smpl", gender="neutral", num_betas=10)
T = lambda x: torch.tensor(np.asarray(x, np.float32)).reshape(1, -1)
tr = A["transl_init"].copy()
for i, key in enumerate(names):
    K = A["K"][i]; mask = cv2.imdecode(np.fromfile(f"{W}/mask/{key}_seg_label.png", np.uint8), -1) > 0
    dep = np.load(f"{W}/out/{key}_person_metric.npy"); zobs = dep[mask & (dep > 0)]
    ys, xs = np.nonzero(mask); mb = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
    V0 = sm(betas=T(A["betas_shared_init"]), body_pose=T(A["body_pose_init"][i]), global_orient=T(A["global_orient_init"][i])).vertices[0].detach().numpy()
    t = tr[i].astype(float)
    for it in range(6):
        V = V0 + t
        t[2] += np.percentile(zobs, 5) - np.percentile(V[:, 2], 5)
        V = V0 + t
        u = K[0, 0] * V[:, 0] / V[:, 2] + K[0, 2]; w = K[1, 2] - K[1, 1] * V[:, 1] / V[:, 2]
        du = (mb[0] + mb[2]) / 2 - (u.min() + u.max()) / 2
        dv = mb[1] - w.min()        # 머리 꼭대기를 맞춤 (발은 신발 때문에 마스크가 더 아래)
        z = np.median(V[:, 2]); t[0] += du * z / K[0, 0]; t[1] -= dv * z / K[1, 1]
    print(key, "Δtransl (m)", np.round(t - tr[i], 3))
    tr[i] = t
A["transl_init"] = tr.astype(np.float32)
np.savez(args.dst, **A); print("saved", args.dst)
