"""한 방향을 빼고 만든 체형(β, D)을, 뺀 방향에 맞춰 보고 맨몸 SMPL vs SMPL+D 를 비교.
체형은 고정, 뺀 방향의 몸 방향·위치·자세만 실루엣 + depth 로 맞춤 (두 경우 같은 절차).
usage: python eval_holdout.py HOLDOUT_DIR VIEW_KEYWORD"""
import sys, json, numpy as np, torch
sys.path.insert(0, "E:/walking_r3d/personal_mesh")
from fit_personal import (Body, load_stand4, vertex_regions, REGION_PARTS, prep_obs, init_transl,
                          view_losses, vnormals, gates, W_BODY, project, sample_nn)

out, kw = sys.argv[1], sys.argv[2]
vw = [v for v in load_stand4() if kw in v["name"]][0]
body = Body(); F = body.faces; reg = vertex_regions(); regm = {r: torch.tensor(reg == r) for r in REGION_PARTS}
fb = np.load(f"{out}/body_fit.npz"); beta = torch.tensor(fb["betas"])
D = torch.tensor(np.load(f"{out}/D.npy")); nrm = torch.tensor(np.load(f"{out}/canon_normals.npy"))
ob = prep_obs(vw, seed=99)


def fit_and_eval(d):
    stage = "D" if d is not None else "body"
    go = torch.tensor(vw["go"], dtype=torch.float32)[None].clone().requires_grad_()
    bp0 = torch.tensor(vw["bp"], dtype=torch.float32)[None]; bp = bp0.clone().requires_grad_()
    tr = torch.tensor(init_transl(body, beta, vw, ob), dtype=torch.float32)[None].requires_grad_()
    for params, iters in (([go, tr], 100), ([go, tr, bp], 200)):
        opt = torch.optim.Adam(params, lr=0.005)
        for _ in range(iters):
            opt.zero_grad()
            V, _ = body.forward(beta, go, bp, tr, d, nrm if d is not None else None)
            L = view_losses(V[0], vnormals(V, F)[0], ob, regm, stage)
            tot = 4 * L["out"] + 2 * L["cov"] + L["dep"] + L["p2m"] + W_BODY["pose"] * ((bp - bp0) ** 2).sum()
            tot.backward(); opt.step()
    rows, Vf = gates(body, beta, go.detach(), bp.detach(), tr.detach(), [vw], d, nrm if d is not None else None)
    # 공통 depth 오차: 보이는 정점 vs 관측 depth (양방향, 전 영역, 5 cm 이내만)
    V = Vf[0]; n = vnormals(Vf, F)[0]; uv, z = project(V, ob["K"]); zo = sample_nn(ob["depth"], uv, ob["HW"])
    ok = ((n * V).sum(-1) < 0) & (zo > 0); r = (z - zo)[ok]
    inl = r.abs() < 0.05
    d3 = torch.cdist(ob["P"], V[(n * V).sum(-1) < 0]).min(1).values
    return dict(iou=rows[0]["iou"], depth_mae_cm=float(r[inl].abs().mean() * 100), depth_inlier=float(inl.float().mean()),
                pts_to_mesh_median_cm=float(d3.median() * 100))


res = {"view": vw["name"], "SMPL(β)": fit_and_eval(None), "SMPL+D": fit_and_eval(D)}
print(json.dumps(res, ensure_ascii=False))
json.dump(res, open(f"{out}/holdout_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
