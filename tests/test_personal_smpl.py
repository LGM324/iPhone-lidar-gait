"""personal_smpl.PersonalSMPL: 개인 메쉬 (SMPL + 법선 변위 D) 의 핵심 성질."""
import numpy as np
import torch

from conftest import SMPL_DIR, needs_smpl


def _personal_npz(tmp_path, D, betas=None):
    p = tmp_path / "pm.npz"
    nrm = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (6890, 1))
    np.savez(p, betas=(np.zeros(10) if betas is None else betas).astype(np.float32),
             D=D.astype(np.float32), canon_normals=nrm, target_height_m=np.float32(1.8))
    return str(p)


@needs_smpl
def test_zero_displacement_equals_smplx(tmp_path):
    """D = 0 이면 smplx SMPL 과 정점·관절이 완전히 같아야 함 (기준 결과 재현성)."""
    from personal_smpl import check_equivalence
    dv, dj = check_equivalence(SMPL_DIR, _personal_npz(tmp_path, np.zeros(6890)))
    assert dv < 1e-5 and dj < 1e-5


@needs_smpl
def test_displacement_moves_surface_not_joints(tmp_path):
    """D 는 옷·신발 표면이므로 정점만 움직이고 관절(뼈) 위치는 그대로여야 함."""
    import smplx
    from personal_smpl import PersonalSMPL
    D = np.full(6890, 0.03)
    m = PersonalSMPL(SMPL_DIR, _personal_npz(tmp_path, D))
    ref = smplx.create(SMPL_DIR, model_type="smpl", gender="neutral", num_betas=10)
    g = torch.Generator().manual_seed(1)
    kw = dict(betas=torch.randn(1, 10, generator=g) * .5, body_pose=torch.randn(1, 69, generator=g) * .2,
              global_orient=torch.randn(1, 3, generator=g) * .3)
    a, r = m(**kw), ref(**kw)
    assert float((a.joints[:, :24] - r.joints[:, :24]).abs().max()) < 1e-5
    moved = (a.vertices - r.vertices).norm(dim=-1)
    assert float(moved.min()) > 0.02 and float(moved.max()) < 0.04
