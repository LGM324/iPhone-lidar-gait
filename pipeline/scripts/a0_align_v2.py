"""
A0_align_v2.1.py  —  Phase 3/4 A0(정렬) v2.1 (사용자 결정 반영)

v2 → v2.1 변경:
  [side] 별도 라벨 파일 X. depth 안에서 z-ordering 즉시 계산 (y-median 상하 분할 → 하부 z-median 좌우 분할).
         near=오른다리(카메라 가까운 z 작음). Sapiens 좌우 미사용 (스펙).
  [RTMW] LAMBDA_KP2D=0 으로 off. yaw 방어는 two-init flip test + GATE2 로 대체.
         depth-only 게이트 결과 보고 yaw under-constrained 확인되면 그때 켬.

v2 (인용):
  [1] 재-anchor depth (global_a=0.486, LiDAR pool 검증). 1.099 폐기.
  [2] E_D torso 가중 (W_TORSO=1.0, W_LEG=0.3).
  [3] E_J = yaw 구속 (지금 off).
  [4] two-init flip test.
  [5] GATE3 = 재-anchor depth 시간축 안정(torso z-ext std / height std).
  [6] E_D = 카메라에서 보이는 표면만.

불변:
  - betas STAND freeze / body_pose = 4D freeze / near+upper 만 / 단방향 Huber 3cm
  - near = 오른다리 (카메라 사람 오른쪽 촬영). GATE2 near→right≈1.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import walkcfg  # [colab_kit] 경로/세션 설정
import os
import json
import numpy as np
import torch

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
FROZEN_NPZ = walkcfg.FROZEN_NPZ  # [colab_kit]
SMPL_MODEL = walkcfg.SMPL_MODEL  # [colab_kit]
DEPTH_SEQ  = walkcfg.DEPTH_SEQ  # [colab_kit]
VIDEO_SEQ  = walkcfg.VIDEO_SEQ  # [colab_kit]
# side: 내부 z-ordering 계산 (별도 파일 없음)
# kp2d: RTMW-x off (LAMBDA_KP2D=0)
CAM_INTR   = walkcfg.CAM_INTR  # [colab_kit]
FOURDH_SEQ = walkcfg.FOURDH_SEQ  # [colab_kit]
OUT_DIR    = walkcfg.A0_DIR  # [colab_kit]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ITERS = 100     # (수렴 확인됨: 이전 250 iter run 이 사용자 육안 정합 통과)
LR = 0.02
MAX_OBS_PTS = 15000   # cdist 병목 완화. 사람 전체 실루엣 커버 유지 (SMPLify/PROX 관례).
HUBER_DELTA_M = 0.03

W_TORSO = 1.0
W_LEG   = 0.3
LAMBDA_DEPTH = 1.0
LAMBDA_KP2D  = 0.0     # off (depth-only A0). yaw 방어 = two-init flip + GATE2. 필요 시 켬.

GATE3_TORSO_STD_CM  = 1.69
GATE3_HEIGHT_STD_CM = 1.44

SIDE_UPPER, SIDE_FAR, SIDE_NEAR = 0, 1, 2
J_LHIP, J_RHIP = 1, 2
J_LKNEE, J_RKNEE = 4, 5
J_LANKLE, J_RANKLE = 7, 8
J_NECK, J_LSH, J_RSH = 12, 16, 17


# ----------------------------------------------------------------------------
def load_frozen_betas(npz_path):
    f = np.load(npz_path, allow_pickle=True)
    betas = f["betas"].astype(np.float32)
    assert betas.shape == (10,), betas.shape
    return betas, float(f["target_height_m"])


def backproject(depth, K):
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.reshape(-1)
    valid = z > 0
    x = (us.reshape(-1) - cx) / fx * z
    y = (vs.reshape(-1) - cy) / fy * z
    pts = np.stack([x, y, z], 1)[valid].astype(np.float32)
    return pts, np.where(valid)[0]


# [side v2.3] 분리 판정 stats (main 끝에서 요약 print). 호출부 변경 없이 집계만.
SIDE_STATS = {"n_frames": 0, "n_separated": 0, "far_pts_sum": 0, "far_pts_frames": 0}


def side_from_zorder(pts):
    """[side v2.3] per-frame adaptive 좌우 분리.
    측정 근거:
      side-view treadmill → z(카메라 depth) = 사람 좌우축(far/near).
      far(왼)가 z로 분리되는 프레임(다리 벌어진 국면)에선 near/far 나눠 far 제외.
      분리 안 되는 프레임(double-support, 두 다리 겹침)에선 전부 near.
    핵심: 이 수정은 far를 A0에 '넣는' 게 아니라, far가 near로 '새는' 것을 막는 것.
          A0 데이터항은 여전히 near+upper만 사용.
    임계는 물리(다리 두께 ~8cm)에서 온 값. 분리율 % 결과에 맞춰 조정 금지.
    """
    SEP_MIN_CM   = 8.0     # 두 z-클러스터 중심 거리 >= 다리 두께면 두 다리
    FAR_FRAC_MIN = 0.15    # 뒤(far) 클러스터 점 비율 하한 (튄 점 배제)

    y = pts[:, 1]
    is_upper = y < float(np.median(y))
    side = np.full(len(pts), SIDE_UPPER, dtype=np.int64)

    leg_mask = ~is_upper
    leg_idx = np.where(leg_mask)[0]
    SIDE_STATS["n_frames"] += 1
    if len(leg_idx) < 50:
        side[leg_mask] = SIDE_NEAR
        return side

    z = pts[leg_idx, 2].astype(np.float64)

    # --- 1D 2-means (k=2) ---
    c = np.array([np.percentile(z, 15), np.percentile(z, 85)], dtype=np.float64)
    for _ in range(50):
        lab = (np.abs(z[:, None] - c[None]).argmin(1))
        newc = np.array([z[lab == k].mean() if (lab == k).any() else c[k]
                         for k in (0, 1)])
        if np.allclose(newc, c):
            break
        c = newc

    lo, hi = (0, 1) if c[0] <= c[1] else (1, 0)   # lo=near(z작음), hi=far(z큼)
    sep_cm = abs(c[1] - c[0]) * 100.0
    frac_hi = float((lab == hi).mean())            # 뒤(far) 클러스터 비율

    separated = (sep_cm >= SEP_MIN_CM) and (frac_hi >= FAR_FRAC_MIN)

    if separated:
        near_leg = leg_idx[lab == lo]
        far_leg  = leg_idx[lab == hi]
        side[near_leg] = SIDE_NEAR      # z 작음 = 오른다리 = near (A0 사용)
        side[far_leg]  = SIDE_FAR       # z 큼 = 왼다리 = far (A0 데이터항서 제외)
        SIDE_STATS["n_separated"]   += 1
        SIDE_STATS["far_pts_sum"]   += int(len(far_leg))
        SIDE_STATS["far_pts_frames"] += 1
    else:
        side[leg_mask] = SIDE_NEAR      # 안 갈림(두 다리 겹침) → 전부 near

    return side


def vertex_normals(verts, faces):
    v = verts
    fn = torch.cross(v[faces[:, 1]] - v[faces[:, 0]],
                     v[faces[:, 2]] - v[faces[:, 0]], dim=1)
    vn = torch.zeros_like(v)
    for i in range(3):
        vn.index_add_(0, faces[:, i], fn)
    return vn / (vn.norm(dim=1, keepdim=True) + 1e-8)


def project(joints3d, K):
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    x, y, z = joints3d[:, 0], joints3d[:, 1], joints3d[:, 2].clamp(min=1e-3)
    return torch.stack([fx * x / z + cx, fy * y / z + cy], dim=1)


def huber(a, delta):
    return torch.where(a < delta, 0.5 * a**2, delta * (a - 0.5 * delta))


def energy(smpl, betas_t, body_pose, go, tr, pts, w_pt, faces,
           kp2d, kp_conf, K):
    """E_A0 = λ_D·E_D(visible, data->model, Huber, 가중) + λ_J·E_J(near 2D).
       LAMBDA_KP2D=0 이면 E_J 스킵 (depth-only)."""
    out = smpl(betas=betas_t.reshape(1, -1), body_pose=body_pose,
               global_orient=go.reshape(1, 3), transl=tr.reshape(1, 3))
    verts = out.vertices[0]
    joints = out.joints[0]

    # E_D: 카메라에서 보이는 vertex만 (PROX visible-surface 근사)
    vn = vertex_normals(verts, faces)
    visible = vn[:, 2] < 0.0        # OpenCV 규약: +z=forward. 카메라 향한 vertex = normal.z<0
    vv = verts[visible]
    d = []
    for i in range(0, pts.shape[0], 4096):
        d.append(torch.cdist(pts[i:i + 4096], vv).min(1).values)
    dist = torch.cat(d)
    E_D = (w_pt * huber(dist, HUBER_DELTA_M)).sum() / (w_pt.sum() + 1e-8)

    if LAMBDA_KP2D > 0 and kp2d is not None and kp_conf is not None:
        # E_J: near 2D keypoint reprojection (yaw 핀)
        proj = project(joints, K)          # (J,2) px
        diff = ((proj - kp2d).norm(dim=1)) * kp_conf
        E_J = diff.sum() / (kp_conf.sum() + 1e-8)
        E_J = E_J / max(K["fx"], 1.0)      # px→정규화
        E = LAMBDA_DEPTH * E_D + LAMBDA_KP2D * E_J
    else:
        E = LAMBDA_DEPTH * E_D              # depth-only

    return E, dist, joints


def yaw180(go_aa):
    import cv2
    R, _ = cv2.Rodrigues(go_aa.astype(np.float64))
    Rf = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], float) @ R
    return cv2.Rodrigues(Rf)[0].reshape(3).astype(np.float32)


def align_one_init(smpl, betas_t, body_pose, init_go, pts, w_pt, faces,
                   kp2d, kp_conf, K):
    go = torch.tensor(init_go, device=DEVICE, requires_grad=True)
    tr = torch.tensor(pts.detach().cpu().numpy().mean(0),
                      device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([go, tr], lr=LR)
    for _ in range(N_ITERS):
        opt.zero_grad()
        E, dist, joints = energy(smpl, betas_t, body_pose, go, tr,
                                 pts, w_pt, faces, kp2d, kp_conf, K)
        E.backward()
        opt.step()
    with torch.no_grad():
        E, dist, joints = energy(smpl, betas_t, body_pose, go, tr,
                                 pts, w_pt, faces, kp2d, kp_conf, K)
    return dict(go=go.detach().cpu().numpy(), tr=tr.detach().cpu().numpy(),
                E=float(E), dist=dist.detach().cpu().numpy(),
                joints=joints.detach().cpu().numpy())


def align_frame(smpl, betas_t, body_pose, init_go, pts_np, side_np,
                faces, kp2d, kp_conf, K):
    mask = (side_np == SIDE_NEAR) | (side_np == SIDE_UPPER)
    pts_all = pts_np[mask]
    side_all = side_np[mask]
    # 관측 서브샘플 (cdist 속도, 정합 정확도 유지)
    if len(pts_all) > MAX_OBS_PTS:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pts_all), MAX_OBS_PTS, replace=False)
        pts_all = pts_all[idx]
        side_all = side_all[idx]
    pts = torch.tensor(pts_all, device=DEVICE)
    w = np.where(side_all == SIDE_UPPER, W_TORSO, W_LEG).astype(np.float32)
    w_pt = torch.tensor(w, device=DEVICE)

    A = align_one_init(smpl, betas_t, body_pose, init_go,
                       pts, w_pt, faces, kp2d, kp_conf, K)
    B = align_one_init(smpl, betas_t, body_pose, yaw180(init_go),
                       pts, w_pt, faces, kp2d, kp_conf, K)
    best, other = (A, B) if A["E"] <= B["E"] else (B, A)
    best["flip_margin"] = abs(A["E"] - B["E"])
    return best


# ----------------------------------------------------------------------------
def gate1_residual(dist_m):
    cm = dist_m * 100
    print(f"[GATE1] residual median={np.median(cm):.2f}cm "
          f"mean={cm.mean():.2f}cm p90={np.percentile(cm, 90):.2f}cm")


def gate2_side(pts_np, side_np, joints, flip_margin):
    rleg = joints[[J_RHIP, J_RKNEE, J_RANKLE]]
    lleg = joints[[J_LHIP, J_LKNEE, J_LANKLE]]

    def frac_right(m):
        p = pts_np[m]
        if len(p) == 0:
            return float("nan")
        dr = np.linalg.norm(p[:, None] - rleg[None], axis=2).min(1)
        dl = np.linalg.norm(p[:, None] - lleg[None], axis=2).min(1)
        return float((dr < dl).mean())

    nr = frac_right(side_np == SIDE_NEAR)
    fr = frac_right(side_np == SIDE_FAR)
    print(f"[GATE2] near->right={nr:.2f}(기대~1) far->right={fr:.2f}(기대~0) "
          f"flip_margin={flip_margin:.4f}")
    if nr < 0.7:
        print("        [WARN] near가 오른다리에 안 붙음 - flip test/side 라벨 재확인")
    if flip_margin < 0.02:
        print("        [WARN] 두 init 잔차 차 작음 - facing 애매(측면 특성)")


def gate3_temporal(torso_ext_cm_list, height_cm_list):
    te = np.array(torso_ext_cm_list); ht = np.array(height_cm_list)
    print(f"[GATE3] torso z-extent std={te.std():.2f}cm (기준 {GATE3_TORSO_STD_CM}) | "
          f"height std={ht.std():.2f}cm (기준 {GATE3_HEIGHT_STD_CM})")
    if te.std() > GATE3_TORSO_STD_CM * 1.3:
        print("        [WARN] torso 두께 시간축 흔들림 - 재-anchor 재확인")


def torso_z_extent_cm(pts_np, side_np):
    p = pts_np[side_np == SIDE_UPPER]
    return float((p[:, 2].max() - p[:, 2].min()) * 100) if len(p) > 20 else np.nan


def save_overlay_ply(pts_np, side_np, best, smpl, betas_t, body_pose, path):
    """관측 (near+upper) 점 = 색A / posed SMPL vertex = 색B."""
    mask = (side_np == SIDE_NEAR) | (side_np == SIDE_UPPER)
    obs = pts_np[mask]
    with torch.no_grad():
        out = smpl(betas=betas_t.reshape(1, -1), body_pose=body_pose,
                   global_orient=torch.tensor(best["go"], device=DEVICE).reshape(1, 3),
                   transl=torch.tensor(best["tr"], device=DEVICE).reshape(1, 3))
    verts = out.vertices[0].detach().cpu().numpy()
    N = len(obs) + len(verts)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {N}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p in obs:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} 220 40 40\n")   # 관측 = 빨강
        for v in verts:
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} 40 140 220\n")  # SMPL = 파랑


# ----------------------------------------------------------------------------
def main():
    import smplx
    os.makedirs(OUT_DIR, exist_ok=True)
    betas, target_h = load_frozen_betas(FROZEN_NPZ)
    betas_t = torch.tensor(betas, device=DEVICE)
    smpl = walkcfg.make_smpl(SMPL_MODEL).to(DEVICE)
    faces = torch.tensor(smpl.faces.astype(np.int64), device=DEVICE)
    with open(CAM_INTR) as f:
        K = json.load(f)
        if "_default" in K:
            K = K["_default"]

    # === 필수 입력 = 4D-Humans θ(t). side/kp2d 는 내부 처리 (v2.1) ===
    if FOURDH_SEQ is None:
        print("[BLOCK] FOURDH_SEQ 미확정 - 4D-Humans hmr2_walking_track.npz "
              "(fids/global_orient/body_pose) 경로 상단 CONFIG 에 지정하고 재실행하세요.")
        return

    fourdh = np.load(FOURDH_SEQ, allow_pickle=True)
    # hmr2_walking_track_v1 schema: fids/global_orient_aa/body_pose_aa
    fourdh_fids = None
    for k in ("fids", "frame", "frames", "fid"):
        if k in fourdh.files:
            fourdh_fids = np.asarray(fourdh[k])
            break
    fourdh_go = np.asarray(
        fourdh["global_orient_aa"] if "global_orient_aa" in fourdh.files else fourdh["global_orient"])
    fourdh_bp = np.asarray(
        fourdh["body_pose_aa"] if "body_pose_aa" in fourdh.files else fourdh["body_pose"])
    # body_pose_aa: (T, 23, 3) → (T, 69) reshape (SMPL forward 규격)
    if fourdh_bp.ndim == 3:
        fourdh_bp = fourdh_bp.reshape(fourdh_bp.shape[0], -1)
    print(f"[4DH] loaded fids [{int(fourdh_fids.min())}~{int(fourdh_fids.max())}] "
          f"N={len(fourdh_fids)}  go={fourdh_go.shape}  bp={fourdh_bp.shape}")

    import glob, re
    files = sorted(glob.glob(os.path.join(DEPTH_SEQ, "WALK_side_*_person_metric.npy")))
    frame_ids = [int(re.search(r"(\d+)_person_metric", os.path.basename(f)).group(1))
                 for f in files]
    if walkcfg.FRAME_LIMIT > 0:                      # 빠른 확인 모드: 앞쪽 N 프레임만
        files, frame_ids = files[:walkcfg.FRAME_LIMIT], frame_ids[:walkcfg.FRAME_LIMIT]
        print(f"[quick] 앞쪽 {len(files)} 프레임만 처리 (WALK_FRAME_LIMIT)")

    # depth fid → 4D-Humans row 매핑
    if fourdh_fids is not None:
        fid_to_row = {int(f): i for i, f in enumerate(fourdh_fids)}
    else:
        # fids 없으면 순서대로 가정 (안전하게 depth fid 첫 값부터 채움)
        fid_to_row = {f: i for i, f in enumerate(frame_ids[:len(fourdh_go)])}

    ply_dir = os.path.join(OUT_DIR, "overlay_ply")
    os.makedirs(ply_dir, exist_ok=True)

    results, torso_ext, heights = [], [], []
    for t_idx, (t, fp) in enumerate(zip(frame_ids, files)):
        row = fid_to_row.get(t)
        if row is None or row >= len(fourdh_go):
            print(f"  [skip] fid {t}: 4D-Humans 매칭 없음")
            continue
        depth = np.load(fp)
        pts, pix = backproject(depth, K)
        s = side_from_zorder(pts)                       # [v2.1] 내부 z-ordering

        bp = torch.tensor(fourdh_bp[row], device=DEVICE).float().reshape(1, -1)
        init_go = np.asarray(fourdh_go[row], np.float32).reshape(3)

        # kp2d/kp_conf: LAMBDA_KP2D=0 이라 실제 사용 X. dummy 넘김.
        J = 24
        kp2d    = torch.zeros((J, 2), device=DEVICE)
        kp_conf = torch.zeros((J,),   device=DEVICE)

        best = align_frame(smpl, betas_t, bp, init_go,
                           pts, s, faces, kp2d, kp_conf, K)

        print(f"--- fid {t:05d} (row {row}) ---")
        gate1_residual(best["dist"])
        gate2_side(pts, s, best["joints"], best["flip_margin"])
        torso_ext.append(torso_z_extent_cm(pts, s))
        vid_p = os.path.join(VIDEO_SEQ, f"WALK_side_{t:05d}.npz")
        if os.path.exists(vid_p):
            vid = np.load(vid_p)
            heights.append(float(vid["height_m"]) * 100)

        # overlay .ply : 관측 (near+upper) 색A + posed SMPL vertex 색B
        save_overlay_ply(pts, s, best, smpl, betas_t, bp,
                         os.path.join(ply_dir, f"a0_{t:05d}.ply"))

        results.append(dict(frame=t, global_orient=best["go"], transl=best["tr"],
                            E=best["E"],
                            residual_cm=float(np.median(best["dist"]) * 100),
                            flip_margin=best["flip_margin"]))

        # [증분 저장] 10프레임마다 npz 갱신 -> 중단 시 GATE 재구성 가능
        if (t_idx % 10 == 0) or (t_idx == len(files) - 1):
            np.savez(os.path.join(OUT_DIR, "A0_pose_per_frame_v2.npz"),
                     **{k: np.array([r[k] for r in results])
                        for k in ["frame", "global_orient", "transl", "E",
                                  "residual_cm", "flip_margin"]})

    gate3_temporal(torso_ext, heights)
    np.savez(os.path.join(OUT_DIR, "A0_pose_per_frame_v2.npz"),
             **{k: np.array([r[k] for r in results])
                for k in ["frame", "global_orient", "transl", "E",
                          "residual_cm", "flip_margin"]})

    # ---- [side v2.3] 분리 판정 검증 요약 ----
    nf, ns = SIDE_STATS["n_frames"], SIDE_STATS["n_separated"]
    if nf > 0:
        ratio = 100.0 * ns / nf
        avg_far = (SIDE_STATS["far_pts_sum"] / SIDE_STATS["far_pts_frames"]) \
                  if SIDE_STATS["far_pts_frames"] > 0 else 0.0
        print("=" * 62)
        print(f"[side v2.3 stats] 분리 프레임: {ns}/{nf} ({ratio:.1f}%)  "
              f"| 참고용 (타겟 아님)")
        print(f"[side v2.3 stats] 분리 프레임 avg far pts: {avg_far:.0f}점")
        print(f"[side v2.3 stats] SEP_MIN_CM=8.0, FAR_FRAC_MIN=0.15 (물리 근거, 조정 금지)")
        print("=" * 62)
    print(f"[A0 v2] done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
