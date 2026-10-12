"""다리 피팅 데이터 항 (near / far 공용).

기본 (지금까지): 관측 점 → 가장 가까운 '카메라를 향한' 정점 거리에 Huber. 모든 점을 양방향으로 끌어당김.

옷 영역 한쪽 항 (WALK_CLOTH_ONESIDED=1):
  가장 가까운 정점이 하의(반바지) 정점이면, 법선 방향 부호거리 sd = (p - v)·n 으로
    sd < 0        : 몸이 옷 표면을 뚫고 카메라 쪽으로 나옴 → 벌점
    0 ≤ sd ≤ tol  : 옷이 몸 위에 떠 있음 (정상)          → 벌점 없음
    sd > tol      : 몸이 옷에서 너무 멀어짐              → 약하게 끌어당김
  맨살·신발 정점은 기존과 같이 양방향.
  근거: BUFF (Zhang et al. CVPR 2017) 옷 아래 몸, DoubleFusion (Yu et al. CVPR 2018) 2겹 구조.
  헐렁한 반바지는 걸을 때 모양이 바뀌므로 몸을 그 표면에 끌어당기면 허벅지 방향이 끌려감.
"""
import torch


def huber(a, delta):
    a = a.abs()
    return torch.where(a < delta, 0.5 * a ** 2, delta * (a - 0.5 * delta))


def nearest_visible(pts, v, vn, chunk=4096):
    """pts (P,3), v/vn (V,3) → (거리, 보이는 정점 중 인덱스, 보이는 정점 마스크)."""
    vis = vn[:, 2] < 0.0
    vv = v[vis]
    ds, ids = [], []
    for i in range(0, pts.shape[0], chunk):
        d, j = torch.cdist(pts[i:i + chunk], vv).min(1)
        ds.append(d); ids.append(j)
    return torch.cat(ds), torch.cat(ids), vis


def leg_data_energy(pts, v, vn, delta, cloth_mask=None, tol=0.06, far_w=0.3):
    """반환: 데이터 항 (스칼라). cloth_mask (V,) bool = 하의 정점 (None 이면 기존 방식)."""
    d, idx, vis = nearest_visible(pts, v, vn)
    if cloth_mask is None:
        return huber(d, delta).mean()
    vv, nn, cm = v[vis], vn[vis], cloth_mask[vis]
    c = cm[idx]
    sd = ((pts - vv[idx]) * nn[idx]).sum(-1)
    e_cloth = huber(torch.relu(-sd), delta) + far_w * huber(torch.relu(sd - tol), delta)
    return torch.where(c, e_cloth, huber(d, delta)).mean()
