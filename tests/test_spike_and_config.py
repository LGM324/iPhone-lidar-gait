"""튄 프레임 판정 규칙과 결과 폴더 꼬리표 (옵션별로 기준 결과를 덮어쓰지 않는지)."""
import importlib
import os

import numpy as np


def test_spike_rules():
    """무릎 > 85° 또는 앞뒤 2 프레임 중앙값과 30° 이상 차이 → 튐. 정상 보행 곡선은 걸리지 않아야 함."""
    from far_spike_refit import spike_mask
    t = np.arange(84)
    knee = 35 + 28 * np.sin(2 * np.pi * t / 39)                # 정상 보행 (7 ~ 63°)
    ok = np.ones(len(t), bool)
    assert not spike_mask(t, knee, ok).any()
    k2 = knee.copy(); k2[10] = 114.0; k2[50] = knee[50] + 40   # 불가능한 각도, 갑작스러운 튐
    bad = spike_mask(t, k2, ok)
    assert bad[10] and bad[50] and bad.sum() == 2


def test_spike_ignores_unfit_frames():
    from far_spike_refit import spike_mask
    t = np.arange(10); knee = np.full(10, 20.0); knee[4] = 120.0
    ok = np.ones(10, bool); ok[4] = False                       # 피팅 안 한(가림) 프레임은 판정 대상 아님
    assert not spike_mask(t, knee, ok).any()


def _cfg(**env):
    for k in ("WALK_PERSONAL", "WALK_FIT_TAG", "WALK_CLOTH_ONESIDED", "WALK_FRAME_LIMIT"):
        os.environ.pop(k, None)
    os.environ.update(env)
    import walkcfg
    return importlib.reload(walkcfg)


def test_result_folder_tags():
    assert _cfg().FIT_TAG == ""                                                      # 기준 결과
    assert _cfg(WALK_PERSONAL="x.npz").FIT_TAG == "_personal"
    assert _cfg(WALK_PERSONAL="x.npz", WALK_CLOTH_ONESIDED="1", WALK_FRAME_LIMIT="84").FIT_TAG == "_personal_c1_q84"
    assert _cfg(WALK_CLOTH_ONESIDED="1").FIT_TAG == "_c1"
    c = _cfg()
    assert c.cloth_mask() is None and c.FROZEN_NPZ.endswith("final_single_smpl_params.npz")
