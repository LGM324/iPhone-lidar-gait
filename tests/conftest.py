"""공용 설정: pipeline/scripts 를 import 경로에 넣고, Drive 가 없어도 walkcfg 가 폴더를 만들지 않게 함.

SMPL 모델이 필요한 테스트는 환경변수 SMPL_MODEL_DIR (안에 smpl/SMPL_NEUTRAL.pkl) 가 없으면 건너뜀.
SMPL 모델은 라이선스상 저장소에 포함하지 않음 (https://smpl.is.tue.mpg.de 에서 직접 받기).
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline", "scripts"))
os.environ.setdefault("WALK_ROOT", os.path.join(ROOT, "_no_drive_"))   # 존재하지 않는 경로 → makedirs 안 함

SMPL_DIR = os.environ.get("SMPL_MODEL_DIR", "")
needs_smpl = pytest.mark.skipif(not os.path.exists(os.path.join(SMPL_DIR, "smpl", "SMPL_NEUTRAL.pkl")),
                                reason="SMPL_MODEL_DIR 에 smpl/SMPL_NEUTRAL.pkl 이 없음 (라이선스상 미포함)")
