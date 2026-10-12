"""개인 메쉬 SMPL (SMPL + 법선 방향 변위 D). smplx SMPL 과 같은 호출법·출력(.vertices, .joints, .faces).

- 관절은 β 만으로 계산 (D 는 옷·신발 표면이라 뼈 위치를 바꾸면 안 됨)
- 정점에는 canonical 에서 D·n 을 더한 뒤 같은 LBS 로 자세를 입힘 → 옷·신발이 뼈를 따라 움직임
- 개인 메쉬 파일 (npz): betas (10,), D (6890,) m, canon_normals (6890,3)  ← personal_mesh/fit_personal.py 출력
"""
from types import SimpleNamespace

import numpy as np
import torch
import smplx
from smplx.lbs import batch_rodrigues, batch_rigid_transform, vertices2joints, blend_shapes


class PersonalSMPL(torch.nn.Module):
    def __init__(self, model_path, personal_npz):
        super().__init__()
        self.base = smplx.create(model_path, model_type="smpl", gender="neutral", num_betas=10)
        p = np.load(personal_npz)
        self.register_buffer("D", torch.tensor(p["D"], dtype=torch.float32))
        self.register_buffer("nrm", torch.tensor(p["canon_normals"], dtype=torch.float32))
        self.faces = self.base.faces
        self.personal_betas = torch.tensor(p["betas"], dtype=torch.float32).reshape(1, 10)

    def forward(self, betas=None, body_pose=None, global_orient=None, transl=None, **_):
        b = self.base
        dev = b.v_template.device
        if betas is None:
            betas = self.personal_betas.to(dev)
        B = max(t.shape[0] for t in (betas, body_pose, global_orient) if t is not None)
        betas = betas.expand(B, -1)
        go = global_orient if global_orient is not None else torch.zeros(B, 3, device=dev)
        bp = body_pose if body_pose is not None else torch.zeros(B, 69, device=dev)
        bp = bp.reshape(B, -1)
        vs = b.v_template[None] + blend_shapes(betas, b.shapedirs)
        J = vertices2joints(b.J_regressor, vs)
        vs = vs + (self.D[:, None] * self.nrm)[None]
        R = batch_rodrigues(torch.cat([go.reshape(B, 3), bp], 1).reshape(-1, 3)).reshape(B, -1, 3, 3)
        pf = (R[:, 1:] - torch.eye(3, device=dev)).reshape(B, -1)
        vp = vs + torch.matmul(pf, b.posedirs).reshape(B, -1, 3)
        Jp, A = batch_rigid_transform(R, J, b.parents)
        T = torch.matmul(b.lbs_weights, A.reshape(B, 24, 16)).reshape(B, -1, 4, 4)
        vh = torch.cat([vp, torch.ones_like(vp[..., :1])], 2)
        v = torch.matmul(T, vh[..., None])[..., :3, 0]
        joints = b.vertex_joint_selector(v, Jp)        # smplx 와 같은 45개 (24 + 정점 기반 추가 관절)
        if transl is not None:
            v = v + transl.reshape(B, 1, 3)
            joints = joints + transl.reshape(B, 1, 3)
        return SimpleNamespace(vertices=v, joints=joints)


def check_equivalence(model_path, personal_npz):
    """D=0 이면 smplx SMPL 과 같은 결과인지 확인 (단위 시험)."""
    m = PersonalSMPL(model_path, personal_npz)
    m.D.zero_()
    ref = smplx.create(model_path, model_type="smpl", gender="neutral", num_betas=10)
    g = torch.Generator().manual_seed(0)
    be, bp, go, tr = (torch.randn(2, 10, generator=g) * .5, torch.randn(2, 69, generator=g) * .2,
                      torch.randn(2, 3, generator=g) * .5, torch.randn(2, 3, generator=g))
    a = m(betas=be, body_pose=bp, global_orient=go, transl=tr)
    r = ref(betas=be, body_pose=bp, global_orient=go, transl=tr)
    return float((a.vertices - r.vertices).abs().max()), float((a.joints - r.joints).abs().max())
