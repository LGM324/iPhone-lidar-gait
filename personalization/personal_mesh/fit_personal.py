"""개인 메쉬 피팅: 여러 방향 정자세 프레임 → SMPL β (공유) + 프레임별 자세 → SMPL+D (옷·신발 표면).

좌표: 처음부터 끝까지 OpenCV 카메라 (x 오른쪽, y 아래, z 앞). 4D-Humans 출력과 같은 규약이라 변환 없음.

설계 근거 (E 드라이브 이전 시행착오 정리):
  - 실루엣을 주 항으로 (이전 가중 0.02~0.05 → IoU 0.46)
  - depth 는 카메라에서 보이는 면만, 5 cm truncated, 피부는 양방향 / 옷·머리·신발은 "몸이 옷보다 앞에 오면 벌점" 한쪽만
  - β 성분별 정규화 + |β| 한계 도달 시 실패 처리 (이전 6~8/10 가 ±3 에 붙음)
  - 자세 사전값은 4D-Humans 초기값 쪽으로 (0 쪽으로 끌면 몸이 T 자세로 펴짐)
  - D 는 법선 방향 숫자 하나, 영역별 범위 (피부 ±1 cm, 옷 0~3 cm, 신발 0~3 cm), 바탕 피팅이 점검을 통과한 뒤에만
  - 경험적 보정 상수 없음. 방향마다 자기 카메라에서 직접 잔차 계산 (합친 점군 사용 안 함)
"""
import argparse, json, os, sys, glob
import numpy as np, cv2, torch
import smplx
from smplx.lbs import batch_rodrigues, batch_rigid_transform, vertices2joints, blend_shapes

torch.set_num_threads(max(1, os.cpu_count() - 1))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_COLAB_ASSETS = "/content/drive/MyDrive/walking_0729/assets"
ASSETS = os.environ.get("WALK_ASSETS") or (_COLAB_ASSETS if os.path.isdir(_COLAB_ASSETS) else "E:/walking_r3d/colab_kit/assets")
SMPL_DIR = f"{ASSETS}/models"
SEG_JSON = f"{ASSETS}/smpl_vert_segmentation.json"
TORSO_PARTS = ["spine", "spine1", "spine2", "hips"]

# ---------------------------------------------------------------- 영역 (SMPL 부위 → 피부/옷/신발/머리)
REGION_PARTS = {
    "skin":  ["leftLeg", "rightLeg", "leftForeArm", "rightForeArm", "leftHand", "rightHand",
              "leftHandIndex1", "rightHandIndex1", "neck"],
    "cloth": ["hips", "spine", "spine1", "spine2", "leftShoulder", "rightShoulder",
              "leftArm", "rightArm", "leftUpLeg", "rightUpLeg"],
    "shoe":  ["leftFoot", "rightFoot", "leftToeBase", "rightToeBase"],
    "head":  ["head"],
}
D_RANGE = {"skin": (-0.01, 0.01), "cloth": (0.0, 0.03), "shoe": (0.0, 0.03), "head": (0.0, 0.02)}   # m (기본값, --d_cloth 등으로 변경)


def vertex_regions(nv=6890):
    seg = json.load(open(SEG_JSON))
    reg = np.full(nv, "cloth", object)
    for r, parts in REGION_PARTS.items():
        for p in parts:
            reg[seg[p]] = r
    return reg


# ---------------------------------------------------------------- 데이터
def rd(p, f=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(p, np.uint8), f)


def crop_view(vw, margin=80):
    """사람 영역만 남겨 메모리 절약. K 의 주점을 옮겨 좌표는 그대로 유지."""
    m = vw["mask"]; ys, xs = np.nonzero(m); H, W = m.shape
    x0, y0 = max(0, xs.min() - margin), max(0, ys.min() - margin)
    x1, y1 = min(W, xs.max() + margin), min(H, ys.max() + margin)
    vw["mask"] = np.ascontiguousarray(m[y0:y1, x0:x1]); vw["depth"] = np.ascontiguousarray(vw["depth"][y0:y1, x0:x1])
    K = vw["K"].copy(); K[0, 2] -= x0; K[1, 2] -= y0; vw["K"] = K; vw["crop"] = (int(x0), int(y0), int(x1), int(y1))
    return vw


def load_stand4():
    """기존 STAND 4방향 (0420/Stand). 시험용."""
    S = "E:/walking_r3d/0420/Stand"; W = f"{S}/STAND_Lidar_DAv2_align_all/DAv2필터링_refine_tailcut/work"
    raw = np.load("E:/walking_r3d/0420/inputs_local/init_params_raw/hmr2_stand_4view_raw.npz")
    Kall = np.load("E:/walking_r3d/0420/outputs/ii8_stand/init_params.npz")
    names = [str(n) for n in raw["view_names"]]
    views = []
    for i, key in enumerate(names):
        mask = rd(f"{W}/mask/{key}_seg_label.png", -1) > 0
        views.append(dict(name=key, K=Kall["K"][list(map(str, Kall["view_names"])).index(key)].astype(np.float64),
                          depth=np.load(f"{W}/out/{key}_person_metric.npy").astype(np.float32), mask=mask,
                          rgb=f"{S}/[Result]00_extract/rgb/{key.split('_')[-1]}.jpg",
                          go=raw["global_orient_aa_per_view"][i], bp=raw["body_pose_aa_per_view"][i].reshape(-1),
                          beta=raw["betas_per_view"][i]))
    return [crop_view(v) for v in views]


def load_session(sess_dir, fids=None):
    """보행 파이프라인 NB1~NB4 를 돌린 세션 (예: 0729_stand_turn)."""
    K = json.load(open(f"{sess_dir}/00_extract/intrinsics.json")); K = K.get("_default", K)
    Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]], np.float64)
    seg = np.load(f"{sess_dir}/01_depth/work/analysis/seg_pass1.npz")
    H, W = seg["shape"]; idx = {str(k): i for i, k in enumerate(seg["keys"])}
    fd = np.load(f"{sess_dir}/03_hmr2/hmr2_walking_track.npz", allow_pickle=True)
    row = {int(f): i for i, f in enumerate(fd["fids"])}
    bp_all = np.asarray(fd["body_pose_aa"]).reshape(len(fd["fids"]), -1)
    views = []
    for f in sorted(row):
        if fids is not None and f not in fids:
            continue
        key = f"WALK_side_{f:05d}"
        dp = f"{sess_dir}/01_depth/work/out/{key}_person_metric.npy"
        if key not in idx or not os.path.exists(dp):
            continue
        mask = np.unpackbits(seg["masks"][idx[key]])[:H * W].reshape(H, W).astype(bool)
        views.append(dict(name=key, K=Km, depth=np.load(dp).astype(np.float32), mask=mask,
                          rgb=f"{sess_dir}/00_extract/rgb/{f:05d}.jpg",
                          go=fd["global_orient_aa"][row[f]], bp=bp_all[row[f]], beta=fd["betas_per_frame"][row[f]]))
        crop_view(views[-1])
    return views


# ---------------------------------------------------------------- SMPL (+D) forward: 관절은 β 만으로, 정점에만 D
class Body:
    def __init__(self):
        m = smplx.create(SMPL_DIR, model_type="smpl", gender="neutral", num_betas=10)
        m = m.to(DEV)
        self.faces = torch.tensor(m.faces.astype(np.int64), device=DEV)
        self.vt, self.sd, self.pd = m.v_template, m.shapedirs, m.posedirs
        self.Jr, self.par, self.W = m.J_regressor, m.parents, m.lbs_weights

    def shaped(self, beta):
        return self.vt[None] + blend_shapes(beta[None], self.sd)            # (1,V,3)

    def forward(self, beta, go, bp, tr, d=None, nrm=None):
        """go (B,3), bp (B,69), tr (B,3), d (V,) 법선 방향 변위, nrm (V,3) canonical 법선."""
        B = go.shape[0]
        vs = self.shaped(beta).expand(B, -1, -1)
        J = vertices2joints(self.Jr, vs)                                     # D 와 무관한 관절
        if d is not None:
            vs = vs + (d[:, None] * nrm)[None]
        pose = torch.cat([go, bp], 1)
        R = batch_rodrigues(pose.reshape(-1, 3)).reshape(B, -1, 3, 3)
        pf = (R[:, 1:] - torch.eye(3, device=R.device)).reshape(B, -1)
        vp = vs + torch.matmul(pf, self.pd).reshape(B, -1, 3)
        Jp, A = batch_rigid_transform(R, J, self.par)
        T = torch.matmul(self.W, A.reshape(B, 24, 16)).reshape(B, -1, 4, 4)
        vh = torch.cat([vp, torch.ones_like(vp[..., :1])], 2)
        v = torch.matmul(T, vh[..., None])[..., :3, 0]
        return v + tr[:, None], Jp + tr[:, None]


def vnormals(v, f):
    """v (B,V,3) → 정점 법선 (B,V,3)."""
    a, b, c = v[:, f[:, 0]], v[:, f[:, 1]], v[:, f[:, 2]]
    fn = torch.cross(b - a, c - a, dim=-1)
    n = torch.zeros_like(v)
    for k in range(3):
        n.index_add_(1, f[:, k], fn)
    return n / (n.norm(dim=-1, keepdim=True) + 1e-9)


# ---------------------------------------------------------------- 관측 준비 (방향마다)
def prep_obs(vw, n_pts=2500, n_mask=1200, seed=0):
    rng = np.random.default_rng(seed)
    m = vw["mask"]; K = vw["K"]; H, W = m.shape
    dt_out = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)   # 마스크 밖 거리(px)
    ys, xs = np.nonzero(m)
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = np.concatenate([q.reshape(-1, 2) for q in cnt])
    ci = rng.choice(len(c), min(len(c), n_mask // 2), replace=False)
    ii = rng.choice(len(ys), min(len(ys), n_mask // 2), replace=False)
    mpts = np.concatenate([c[ci].astype(np.float32), np.c_[xs[ii], ys[ii]].astype(np.float32)])
    z = vw["depth"]; vv, uu = np.nonzero((z > 0) & m)
    j = rng.choice(len(vv), min(len(vv), n_pts), replace=False)
    zz = z[vv[j], uu[j]]
    P = np.c_[(uu[j] - K[0, 2]) / K[0, 0] * zz, (vv[j] - K[1, 2]) / K[1, 1] * zz, zz].astype(np.float32)
    T_ = lambda x, **k: torch.tensor(x, device=DEV, **k)
    return dict(dt=T_(dt_out)[None, None], depth=T_(np.where(m, z, 0).astype(np.float32))[None, None],
                mpts=T_(mpts), P=T_(P), K=T_(K, dtype=torch.float32), HW=(H, W),
                bbox=np.array([xs.min(), ys.min(), xs.max(), ys.max()], float), zmed=float(np.median(zz)))


def project(v, K):
    z = v[..., 2].clamp(min=1e-3)
    return torch.stack([K[0, 0] * v[..., 0] / z + K[0, 2], K[1, 1] * v[..., 1] / z + K[1, 2]], -1), z


def sample(img, uv, HW):
    H, W = HW
    g = torch.stack([uv[..., 0] / (W - 1) * 2 - 1, uv[..., 1] / (H - 1) * 2 - 1], -1).reshape(1, -1, 1, 2)
    return torch.nn.functional.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True).reshape(-1)


def sample_nn(img, uv, HW):
    H, W = HW
    g = torch.stack([uv[..., 0] / (W - 1) * 2 - 1, uv[..., 1] / (H - 1) * 2 - 1], -1).reshape(1, -1, 1, 2)
    return torch.nn.functional.grid_sample(img, g, mode="nearest", padding_mode="zeros", align_corners=True).reshape(-1)


# ---------------------------------------------------------------- 손실
LOOSE = dict(w=1.0, tol=0.03)   # 맨몸 단계: 옷·머리·신발 쪽 마스크 채우기 가중치 / 허용 간격 (--loose_cov_w, --loose_tol)


def view_losses(v, n, ob, reg_masks, stage, trunc=0.05):
    """v,n (V,3) 한 방향의 posed 정점/법선."""
    uv, z = project(v, ob["K"])
    f = ob["K"][0, 0]
    # 1) 실루엣 바깥: 마스크 밖으로 나간 정점 (px → m 환산)
    out = sample(ob["dt"], uv, ob["HW"]) * z / f
    L_out = out.clamp(max=0.10).mean()
    # 2) 실루엣 채우기: 마스크 점 → 가장 가까운 투영 정점
    dmin, imin = torch.cdist(ob["mpts"], uv).min(1)
    dm = dmin * ob["zmed"] / f
    loose = ~reg_masks["skin"][imin]                       # 옷/머리/신발 쪽 마스크는 몸보다 넓어도 됨
    cloth = reg_masks["cloth"][imin]
    if stage == "body":
        # 머리카락·신발: 몸보다 2~3 cm 크므로 tol 만큼 여유를 두고 채움
        dm = torch.where(loose, (dm - LOOSE["tol"]).clamp(min=0), dm)
        # 옷(티셔츠·반바지): 몸이 채울 필요 없음 (옷 아래 몸, 바깥은 L_out 이 막음). w=1 이면 이전과 같음
        dm = torch.where(cloth, LOOSE["w"] * dm, dm)
    L_cov = dm.clamp(max=0.10).mean()
    # 3) depth: 카메라를 향한 정점만, 관측 z 와 비교
    vis = (n * v).sum(-1) < 0
    zo = sample_nn(ob["depth"], uv, ob["HW"])
    ok = vis & (zo > 0)
    r = z - zo
    tight = reg_masks["skin"] if stage == "body" else torch.ones_like(reg_masks["skin"])
    e_t = r.abs().clamp(max=trunc)                          # 피부(또는 D 단계 전체): 양방향
    e_l = (-r).clamp(min=0, max=trunc)                      # 옷: 몸이 관측 표면보다 앞이면 벌점
    e = torch.where(tight, e_t, e_l)
    L_dep = (e * ok).sum() / ok.sum().clamp(min=1)
    # 4) 관측 점 → 가장 가까운 보이는 정점 (피부 정점이 가장 가까운 점만, D 단계는 전부)
    vv = v[vis]
    d3, i3 = torch.cdist(ob["P"], vv).min(1)
    use = reg_masks["skin"][vis][i3] if stage == "body" else torch.ones_like(d3, dtype=torch.bool)
    L_p2m = (d3.clamp(max=trunc) * use).sum() / use.sum().clamp(min=1)
    return dict(out=L_out, cov=L_cov, dep=L_dep, p2m=L_p2m)


W_BODY = dict(out=4.0, cov=2.0, dep=1.0, p2m=1.0, pose=0.02, beta=0.002, height=1.0)
BETA_W = torch.tensor([0.1, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1.0], device=DEV)          # β0(키·크기) 는 느슨, 나머지 조임


def canon_height(body, beta):
    v = body.shaped(beta)[0]
    return v[:, 1].max() - v[:, 1].min()


# ---------------------------------------------------------------- 초기 위치 (마스크 bbox + depth 앞면)
def init_transl(body, beta, vw, ob, torso_idx, half_thick=0.08):
    """z: 몸통 정점 중앙값 = 관측 depth 중앙값 + 몸 반두께 (팔 방향이 틀린 초기 자세에도 안정).
    x, y: 투영 범위 중심 / 머리 꼭대기를 마스크에 맞춤."""
    with torch.no_grad():
        v0, _ = body.forward(beta, torch.tensor(vw["go"], dtype=torch.float32, device=DEV)[None],
                             torch.tensor(vw["bp"], dtype=torch.float32, device=DEV)[None], torch.zeros(1, 3, device=DEV))
    v0 = v0[0].cpu().numpy().astype(float); K = vw["K"]; mb = ob["bbox"]
    t = np.zeros(3)
    for _ in range(8):
        t[2] = ob["zmed"] + half_thick - np.median(v0[torso_idx, 2])
        V = v0 + t
        u = K[0, 0] * V[:, 0] / V[:, 2] + K[0, 2]; w = K[1, 1] * V[:, 1] / V[:, 2] + K[1, 2]
        zc = np.median(V[:, 2])
        t[0] += ((mb[0] + mb[2]) / 2 - (u.min() + u.max()) / 2) * zc / K[0, 0]
        t[1] += (mb[1] - w.min()) * zc / K[1, 1]                       # 머리 꼭대기 맞춤
    return t


# ---------------------------------------------------------------- 점검 (실루엣 IoU / bbox / 방향)
def raster(v, f, K, HW):
    u = K[0, 0] * v[:, 0] / v[:, 2] + K[0, 2]; w = K[1, 1] * v[:, 1] / v[:, 2] + K[1, 2]
    s = np.zeros(HW, np.uint8)
    for t in f:
        cv2.fillConvexPoly(s, np.c_[u[t], w[t]].astype(np.int32), 1)
    return s.astype(bool)


def gates(body, beta, go, bp, tr, views, d=None, nrm=None):
    with torch.no_grad():
        V, J = body.forward(beta, go, bp, tr, d, nrm)
    V, J = V.cpu(), J.cpu()
    F = body.faces.cpu().numpy(); rows = []
    for k, vw in enumerate(views):
        s = raster(V[k].numpy(), F, vw["K"], vw["mask"].shape); m = vw["mask"]
        iou = (s & m).sum() / (s | m).sum()
        ys, xs = np.nonzero(s); ym, xm = np.nonzero(m)
        hr = (ys.max() - ys.min()) / (ym.max() - ym.min())
        dx = ((xs.min() + xs.max()) - (xm.min() + xm.max())) / 2 / (ym.max() - ym.min())
        head_up = bool(J[k, 15, 1] < J[k, 7, 1])                        # OpenCV: 머리가 발목보다 위(작은 y)
        rows.append(dict(view=vw["name"], iou=round(float(iou), 3), height_ratio=round(float(hr), 3),
                         center_dx=round(float(dx), 3), head_up=head_up))
    return rows, V


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="stand4", help="stand4 | 세션 폴더 경로")
    ap.add_argument("--out", default="E:/walking_r3d/personal_mesh/out_stand4")
    ap.add_argument("--iters_a", type=int, default=150)
    ap.add_argument("--iters_b", type=int, default=400)
    ap.add_argument("--iters_d", type=int, default=400)
    ap.add_argument("--target_h", type=float, default=1.81, help="맨발 키(m). 0 이면 키 항 없음")
    ap.add_argument("--holdout", default="", help="피팅에서 뺄 view 이름 일부 (검증용)")
    ap.add_argument("--skip_d", action="store_true")
    ap.add_argument("--d_cloth", type=float, default=0.03)
    ap.add_argument("--d_shoe", type=float, default=0.03)
    ap.add_argument("--w_lap", type=float, default=50.0)
    ap.add_argument("--batch", type=int, default=8, help="반복마다 쓰는 view 수 (0 = 전부). 메모리 절약")
    ap.add_argument("--d_fid_min", type=int, default=0, help="D 단계에 쓸 프레임 fid 하한 (예: 1470 = STAND 만)")
    ap.add_argument("--val_fids", default="", help="검증용으로 뺄 fid 목록 (쉼표). 피팅에 안 쓰고 끝에 평가")
    ap.add_argument("--loose_cov_w", type=float, default=1.0, help="맨몸 단계에서 옷 쪽 마스크를 채우는 힘 (0 = 옷 아래 몸, 날씬)")
    ap.add_argument("--loose_tol", type=float, default=0.03)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D_RANGE["cloth"] = (0.0, a.d_cloth); D_RANGE["shoe"] = (0.0, a.d_shoe)
    LOOSE["w"], LOOSE["tol"] = a.loose_cov_w, a.loose_tol

    views = load_stand4() if a.data == "stand4" else load_session(a.data)
    hold = [vw for vw in views if a.holdout and a.holdout in vw["name"]]
    vset = {int(x) for x in a.val_fids.split(",") if x.strip()}
    fid_of = lambda vw: int(vw["name"].split("_")[-1]) if vw["name"].split("_")[-1].isdigit() else -1
    val = [vw for vw in views if fid_of(vw) in vset]
    views = [vw for vw in views if vw not in hold and vw not in val]
    print(f"[data] {len(views)} views (+{len(hold)} holdout, {len(val)} 검증 프레임)")
    body = Body(); F = body.faces
    reg = vertex_regions(); regm = {r: torch.tensor(reg == r, device=DEV) for r in REGION_PARTS}
    seg = json.load(open(SEG_JSON)); torso_idx = np.unique(np.concatenate([seg[p] for p in TORSO_PARTS]))
    obs = [prep_obs(vw, seed=i) for i, vw in enumerate(views)]
    for vw in views:
        vw.pop("depth", None)                                  # 원본 depth 해제 (obs 에 복사본)
    nV = len(views); rng = np.random.default_rng(0)

    def pick():
        return np.arange(nV) if (a.batch <= 0 or a.batch >= nV) else rng.choice(nV, a.batch, replace=False)
    print(f"[device] {DEV}  batch {a.batch if 0 < a.batch < nV else nV}/{nV}")

    TT = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=DEV)
    beta = TT(np.median([vw["beta"] for vw in views], 0))
    go = TT(np.stack([vw["go"] for vw in views]))
    bp0 = TT(np.stack([vw["bp"] for vw in views]))
    tr = TT(np.stack([init_transl(body, beta, vw, ob, torso_idx) for vw, ob in zip(views, obs)]))
    rows, _ = gates(body, beta, go, bp0, tr, views)
    print("[init]", rows)
    assert all(r["head_up"] for r in rows), "초기 자세가 거꾸로 — 좌표 규약 확인"

    beta = beta.clone().requires_grad_(); go = go.clone().requires_grad_(); tr = tr.clone().requires_grad_()
    bp = bp0.clone().requires_grad_()

    def body_step(params, iters, lr, free_beta):
        opt = torch.optim.Adam(params, lr=lr)
        for it in range(iters):
            opt.zero_grad()
            ix = pick(); it_ = torch.tensor(ix, device=DEV)
            V, _ = body.forward(beta, go[it_], bp[it_], tr[it_])
            N = vnormals(V, F)
            Ls = [view_losses(V[j], N[j], obs[k], regm, "body") for j, k in enumerate(ix)]
            L = {k: torch.stack([l[k] for l in Ls]).mean() for k in Ls[0]}
            tot = sum(W_BODY[k] * L[k] for k in L)
            tot = tot + W_BODY["pose"] * ((bp[it_] - bp0[it_]) ** 2).sum(1).mean()
            if free_beta:
                tot = tot + W_BODY["beta"] * ((beta * BETA_W) ** 2).sum()
                if a.target_h > 0:
                    tot = tot + W_BODY["height"] * (canon_height(body, beta) - a.target_h) ** 2
            tot.backward(); opt.step()
            if it % 50 == 0 or it == iters - 1:
                print(f"  it {it:4d} tot {tot.item():.4f} " + " ".join(f"{k} {v.item()*100:.2f}" for k, v in L.items())
                      + f" | h {canon_height(body, beta).item():.3f} |β|max {beta.abs().max().item():.2f}", flush=True)

    print("[stage A] 방향·위치만"); body_step([go, tr], a.iters_a, 0.01, False)
    print("[stage B] + β, 자세"); body_step([go, tr, beta, bp], a.iters_b, 0.005, True)

    rows, Vb = gates(body, beta, go, bp, tr, views)
    print("[gate body]"); [print("  ", r) for r in rows]
    fail = []
    if beta.abs().max() > 2.5: fail.append("|β| 한계 근접")
    if not all(r["head_up"] for r in rows): fail.append("거꾸로 선 view")
    if any(not (0.90 <= r["height_ratio"] <= 1.03) for r in rows): fail.append("실루엣 키 비율 0.90~1.03 밖 (마스크엔 머리카락·신발 포함)")
    res = dict(beta=beta.detach().cpu().numpy().tolist(), canon_height=float(canon_height(body, beta)), gates_body=rows, fail=fail)
    np.savez(f"{a.out}/body_fit.npz", betas=beta.detach().cpu().numpy(), global_orient=go.detach().cpu().numpy(),
             body_pose=bp.detach().cpu().numpy(), transl=tr.detach().cpu().numpy(), view_names=[vw["name"] for vw in views])

    if fail:
        print("[STOP] 바탕 피팅 점검 실패 → D 단계로 가지 않음:", fail)
    elif not a.skip_d:
        print("[stage D] 법선 방향 변위 (β·자세 고정)")
        with torch.no_grad():
            vsh = body.shaped(beta.detach())
            nrm = vnormals(vsh, F)[0]
        lo = torch.tensor([D_RANGE[r][0] for r in reg], dtype=torch.float32, device=DEV)
        hi = torch.tensor([D_RANGE[r][1] for r in reg], dtype=torch.float32, device=DEV)
        mid = (0 - lo) / (hi - lo); raw = torch.logit(mid.clamp(1e-3, 1 - 1e-3)).clone().requires_grad_()
        # 정점 이웃 (Laplacian)
        e = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = torch.cat([e, e.flip(1)])
        deg = torch.zeros(6890, device=DEV).index_add_(0, e[:, 0], torch.ones(len(e), device=DEV))
        opt = torch.optim.Adam([raw], lr=0.05)
        b_, g_, p_, t_ = beta.detach(), go.detach(), bp.detach(), tr.detach()
        d_ix = np.array([k for k, vw in enumerate(views) if fid_of(vw) >= a.d_fid_min])
        print(f"  D 단계 view: {len(d_ix)}/{nV} (fid >= {a.d_fid_min})")

        def pick_d():
            return d_ix if (a.batch <= 0 or a.batch >= len(d_ix)) else rng.choice(d_ix, a.batch, replace=False)
        for it in range(a.iters_d):
            opt.zero_grad()
            d = lo + (hi - lo) * torch.sigmoid(raw)
            ix = pick_d(); it_ = torch.tensor(ix, device=DEV)
            V, _ = body.forward(b_, g_[it_], p_[it_], t_[it_], d, nrm); N = vnormals(V, F)
            Ls = [view_losses(V[j], N[j], obs[k], regm, "D") for j, k in enumerate(ix)]
            L = {k: torch.stack([l[k] for l in Ls]).mean() for k in Ls[0]}
            lap = d - torch.zeros(6890, device=DEV).index_add_(0, e[:, 0], d[e[:, 1]]) / deg
            tot = 4 * L["out"] + 2 * L["cov"] + L["dep"] + L["p2m"] + a.w_lap * (lap ** 2).mean()
            tot.backward(); opt.step()
            if it % 50 == 0 or it == a.iters_d - 1:
                print(f"  it {it:4d} tot {tot.item():.4f} " + " ".join(f"{k} {v.item()*100:.2f}" for k, v in L.items())
                      + f" | D cm: " + " ".join(f"{r} {d[regm[r]].median().item()*100:.2f}/{d[regm[r]].abs().max().item()*100:.2f}" for r in regm), flush=True)
        d = (lo + (hi - lo) * torch.sigmoid(raw)).detach()
        rowsD, _ = gates(body, b_[None][0], g_[d_ix], p_[d_ix], t_[d_ix], [views[k] for k in d_ix], d, nrm)
        print("[gate D]"); [print("  ", r) for r in rowsD]
        at_bound = float(((d - lo).abs() < 1e-3).float().mean() + ((hi - d).abs() < 1e-3).float().mean())
        skin_med = float(d[regm["skin"]].abs().median())
        res.update(gates_D=rowsD, D_at_bound_frac=at_bound, skin_D_median_cm=skin_med * 100,
                   D_stats={r: [float(d[regm[r]].median() * 100), float(d[regm[r]].abs().max() * 100)] for r in regm})
        if skin_med > 0.01: print("[WARN] 피부 D 중앙값 > 1 cm — 바탕 피팅 재확인")
        d, nrm, b_ = d.cpu(), nrm.cpu(), b_.cpu()
        np.save(f"{a.out}/D.npy", d.numpy()); np.save(f"{a.out}/canon_normals.npy", nrm.numpy())
        # 보행 파이프라인용 개인 메쉬 파일 (scripts/personal_smpl.py 가 읽음)
        np.savez(f"{a.out}/personal_mesh.npz", betas=b_.numpy().astype(np.float32), D=d.numpy().astype(np.float32),
                 canon_normals=nrm.numpy().astype(np.float32), target_height_m=np.float32(canon_height(body, b_.to(DEV)).item()),
                 source=np.array(a.data))
        with torch.no_grad():
            vc = (body.shaped(b_.to(DEV))[0].cpu() + d[:, None] * nrm).numpy()
        with open(f"{a.out}/personal_canonical.obj", "w") as fo:
            for p in vc: fo.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
            for t in F.cpu().numpy() + 1: fo.write(f"f {t[0]} {t[1]} {t[2]}\n")
    if val:
        vobs = [prep_obs(vw, seed=100 + i) for i, vw in enumerate(val)]
        for vw in val:
            vw.pop("depth", None)
        b0 = beta.detach()
        dD = torch.tensor(np.load(f"{a.out}/D.npy"), device=DEV) if os.path.exists(f"{a.out}/D.npy") else None
        nD = torch.tensor(np.load(f"{a.out}/canon_normals.npy"), device=DEV) if dD is not None else None
        out_rows = []
        for vw, ob in zip(val, vobs):
            row = dict(view=vw["name"])
            for tag, dd in (("SMPL", None), ("SMPL+D", dD)):
                if tag == "SMPL+D" and dd is None:
                    continue
                stage = "D" if dd is not None else "body"
                g1 = TT(vw["go"])[None].clone().requires_grad_(); p0 = TT(vw["bp"])[None]; p1 = p0.clone().requires_grad_()
                t1 = TT(init_transl(body, b0, vw, ob, torso_idx))[None].requires_grad_()
                for params, n_it in (([g1, t1], 100), ([g1, t1, p1], 200)):
                    o2 = torch.optim.Adam(params, lr=0.005)
                    for _ in range(n_it):
                        o2.zero_grad()
                        V1, _ = body.forward(b0, g1, p1, t1, dd, nD)
                        L1 = view_losses(V1[0], vnormals(V1, F)[0], ob, regm, stage)
                        (4 * L1["out"] + 2 * L1["cov"] + L1["dep"] + L1["p2m"] + W_BODY["pose"] * ((p1 - p0) ** 2).sum()).backward()
                        o2.step()
                r1, Vf = gates(body, b0, g1.detach(), p1.detach(), t1.detach(), [vw], dd, nD)
                with torch.no_grad():
                    V = Vf[0].to(DEV); n = vnormals(V[None], F)[0]
                    uv, z = project(V, ob["K"]); zo = sample_nn(ob["depth"], uv, ob["HW"])
                    ok = ((n * V).sum(-1) < 0) & (zo > 0); rr = (z - zo)[ok]; inl = rr.abs() < 0.05
                row[tag] = dict(iou=r1[0]["iou"], depth_mae_cm=round(float(rr[inl].abs().mean() * 100), 2),
                                depth_inlier=round(float(inl.float().mean()), 3))
            out_rows.append(row); print("[val]", row, flush=True)
        res["validation"] = out_rows
        for tag in ("SMPL", "SMPL+D"):
            xs = [r[tag] for r in out_rows if tag in r]
            if xs:
                res[f"validation_mean_{tag}"] = {k: round(float(np.mean([x[k] for x in xs])), 3) for k in xs[0]}
                print(f"[val 평균] {tag}: {res[f'validation_mean_{tag}']}")
    json.dump(res, open(f"{a.out}/report.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("[saved]", a.out)


if __name__ == "__main__":
    main()
