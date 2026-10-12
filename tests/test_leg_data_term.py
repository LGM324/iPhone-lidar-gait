"""leg_data_term: 다리 피팅 데이터 항 (양방향 / 하의 한쪽)."""
import torch

from leg_data_term import huber, leg_data_energy


def _sphere(n=2000, r=0.1, z=2.0):
    """간단한 닫힌 표면 (구): 정점과 바깥 법선. 카메라는 원점, +z 를 봄."""
    g = torch.Generator().manual_seed(0)
    d = torch.randn(n, 3, generator=g); d = d / d.norm(dim=1, keepdim=True)
    v = d * r + torch.tensor([0.0, 0.0, z])
    return v, d


def test_off_equals_two_sided_huber():
    """한쪽 항을 끄면 (cloth_mask=None) 기존 공식: 점 → 가장 가까운 '카메라를 향한' 정점 거리의 Huber 평균."""
    v, n = _sphere()
    g = torch.Generator().manual_seed(1)
    pts = v[n[:, 2] < 0][:300] + torch.randn(300, 3, generator=g) * 0.01
    vv = v[n[:, 2] < 0]
    expected = huber(torch.cdist(pts, vv).min(1).values, 0.03).mean()
    assert torch.allclose(leg_data_energy(pts, v, n, 0.03, None), expected)


def test_one_sided_signs():
    """옷 정점: 옷이 몸 위에 떠 있으면(바깥 3 cm) 벌점 ≈ 0, 몸이 옷을 뚫으면(안쪽 1 cm) 벌점 > 0."""
    v, n = _sphere()
    front = torch.where(n[:, 2] < -0.5)[0][:200]
    mask = torch.ones(len(v), dtype=torch.bool)
    outside = v[front] + n[front] * 0.03
    inside = v[front] - n[front] * 0.01
    e_out = float(leg_data_energy(outside, v, n, 0.03, mask, tol=0.06))
    e_in = float(leg_data_energy(inside, v, n, 0.03, mask, tol=0.06))
    e_two = float(leg_data_energy(outside, v, n, 0.03, None))
    assert e_out < 1e-6
    assert e_in > 1e-5
    assert e_two > 1e-4                       # 같은 점이라도 양방향이면 끌어당김


def test_far_beyond_tol_pulls_weakly():
    """옷이 허용 간격(6 cm)보다 멀면 약하게 당김 (벌점 > 0 이지만 같은 거리 양방향보다 작음)."""
    v, n = _sphere()
    front = torch.where(n[:, 2] < -0.5)[0][:200]
    mask = torch.ones(len(v), dtype=torch.bool)
    far = v[front] + n[front] * 0.09
    e_one = float(leg_data_energy(far, v, n, 0.03, mask, tol=0.06))
    e_two = float(leg_data_energy(far, v, n, 0.03, None))
    assert 0 < e_one < e_two
