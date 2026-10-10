"""
walkcfg.py — Colab 파이프라인 공용 경로/세션 설정 (모든 스크립트가 여기서 경로를 읽음)

환경변수:
  WALK_ROOT     드라이브 루트 (기본 /content/drive/MyDrive/walking_0729)
  WALK_SESSION  세션 이름    (기본 0729_walk)  예: 0729_walk, 0729_full
  WALK_GAIT_FRAMES  1 보행주기 프레임 수 (NB6 가 자동 측정해 넣음, 기본 42 = 4/20 값)

세션 폴더 구조 ({ROOT}/sessions/{SESSION}/):
  00_extract/   rgb/ depth_npy/ conf_npy/ intrinsics.json meta.json motion_envelope.npz
  01_depth/     inputs/intrinsics.json  work/{video,out,mask,crop,analysis}   (구 walking_pipeline)
  03_hmr2/      hmr2_walking_track.npz
  04_fit/       A0/ near/ far/ ankle/
  05_knee/      knee_basis_full.* far_thigh_gate_full.* far_knee_final.* knee_final.*
  06_imu/       IMU 검증 결과
"""
import json
import os

ROOT = os.environ.get("WALK_ROOT", "/content/drive/MyDrive/walking_0729")
SESSION = os.environ.get("WALK_SESSION", "0729_walk")

ASSETS = f"{ROOT}/assets"
SMPL_MODEL = f"{ASSETS}/models"                      # 안에 smpl/SMPL_NEUTRAL.pkl
FROZEN_NPZ = f"{ASSETS}/final_single_smpl_params.npz"  # STAND ii13 rawdepth_w05_leftcorr β (고정)
SEG_JSON = f"{ASSETS}/smpl_vert_segmentation.json"

SESS = f"{ROOT}/sessions/{SESSION}"
EXTRACT_DIR = f"{SESS}/00_extract"
CAM_INTR = f"{EXTRACT_DIR}/intrinsics.json"

DEPTH_BASE = f"{SESS}/01_depth"
DEPTH_SEQ = f"{DEPTH_BASE}/work/out"     # WALK_side_{fid:05d}_person_metric.npy
VIDEO_SEQ = f"{DEPTH_BASE}/work/video"   # WALK_side_{fid:05d}.npz
GLOBAL_JSON = f"{DEPTH_BASE}/work/analysis/global_const_fit.json"

FOURDH_SEQ = f"{SESS}/03_hmr2/hmr2_walking_track.npz"

FIT_DIR = f"{SESS}/04_fit"
A0_DIR = f"{FIT_DIR}/A0"
A0_NPZ = f"{A0_DIR}/A0_pose_per_frame_v2.npz"
NEAR_DIR = f"{FIT_DIR}/near"
NEAR_F2 = f"{NEAR_DIR}/depthfit_near_leg_lam0_f2.npz"
NEAR_F3 = f"{NEAR_DIR}/depthfit_near_leg_lam0_f3.npz"
FAR_DIR = f"{FIT_DIR}/far"
FAR_F2 = f"{FAR_DIR}/depthfit_far_leg_lam0_far2.npz"
ANKLE_DIR = f"{FIT_DIR}/ankle"

KNEE_DIR = f"{SESS}/05_knee"
IMU_OUT = f"{SESS}/06_imu"

GAIT_FRAMES = float(os.environ.get("WALK_GAIT_FRAMES", "42"))


def global_a():
    """NB3-① extract_pool_fit.py 가 쓴 global_const_fit.json 의 chosen_global_a.
    (4/20 하드코딩 0.486 대신 세션별 값을 사용)"""
    if not os.path.exists(GLOBAL_JSON):
        raise SystemExit(f"[walkcfg] {GLOBAL_JSON} 없음 → 먼저 extract_pool_fit.py 실행")
    with open(GLOBAL_JSON, encoding="utf-8") as f:
        return float(json.load(f)["chosen_global_a"])


if os.path.isdir(ROOT):   # 드라이브가 마운트된 경우에만 출력 폴더 생성
    for _d in (A0_DIR, NEAR_DIR, FAR_DIR, ANKLE_DIR, KNEE_DIR, IMU_OUT,
               f"{DEPTH_BASE}/work/analysis"):
        os.makedirs(_d, exist_ok=True)
