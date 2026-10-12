# 보행 파이프라인 (A ~ E, Google Colab)

원래 로컬에서 단계별 스크립트로 돌리던 파이프라인을 **Colab 만으로** 다시 실행하도록 옮긴 것입니다. 원본 스크립트의 로직은 그대로 두고, 경로 상수만 [`scripts/walkcfg.py`](scripts/walkcfg.py) 를 읽게 바꿨습니다.

## 노트북

| NB | 런타임 | 하는 일 | 산출 (`sessions/{SESSION}/`) |
|---|---|---|---|
| NB0 setup_folder | CPU | Drive 폴더 생성·업로드 확인 (최초 1회) | — |
| NB1 extract_r3d | CPU | `.r3d` 전체 스캔 (움직임 곡선, 미리보기) → 구간 추출 (`START`, `N`, `STEP`) | `00_extract/` |
| NB2 sapiens_depth | GPU | Sapiens Seg → Depth 1B → LiDAR anchor → 꼬리 제거 + Normal 정제 (청크 처리, 이어하기) | `01_depth/work/{out,video}` |
| NB3 reanchor | GPU 권장 | 세션 전역 `a` (fixed-effects) → 전 프레임 재적용 (원본은 `.bak`) | `01_depth/work/analysis/` |
| NB4 4dhumans_init | GPU | 4D-Humans (HMR 2.0) 프레임별 초기 자세 | `03_hmr2/` |
| NB5 smpl_fit | GPU | A0 몸 정렬 → 가까운(오른) 다리 → 먼(왼) 다리, λ_prior = 0 | `04_fit/{A0,near,far}` |
| NB6 knee_timeseries | CPU | 보행주기 측정 → 측정 근거 → 가림 판정 → 보간 | `05_knee/` |
| NB7 imu_validation | CPU | IMU 무릎각 · 동기화 · 비교 | `06_imu/` |
| NB8 visualize | CPU | 단계별 그림·영상 일괄 생성 | `viz/` |
| **NB_RUN_ALL** | GPU | NB1 ~ NB7 의 셀을 순서대로 실행, 단계마다 결과 파일 확인 (`START_AT` 으로 이어하기). NB5 뒤 튄 프레임 재피팅 자동 실행 | — |
| **NB_EXPERIMENTS** | GPU | 체형(β) × 표면(D) × 데이터 항 16 조합을 같은 조건으로 실행 → 요약표·주효과 → 미리 정한 기준으로 최선 조합 선택 → 250 프레임 확인 | `experiments/` |

각 NB 끝의 `시각화` 셀은 그 단계의 그림 (`viz/NB{k}/`) 과 영상 (`viz/videos/`) 을 만듭니다.

## 스크립트

| 단계 | 스크립트 |
|---|---|
| 설정 | `walkcfg.py` (`WALK_ROOT`, `WALK_SESSION` 환경변수로 경로 결정) |
| A | `extract_session.py` |
| B | `extract_pool_fit.py` (전역 a), `phase4_reanchor.py` |
| C | `a0_align_v2.py` (몸 정렬 + 좌우 다리 구분 `side_from_zorder`), `depthfit_near_leg_lam0.py`, `depthfit_far_leg_lam0.py`, `08_ankle_refit.py` (선택) |
| D | `knee_measurement_basis.py`, `far_thigh_npts_gate.py`, `far_thigh_infill.py`, `far_thigh_infill_v2.py` |
| E | `imu_validate.py` |
| 개인 메쉬·옷 | `personal_smpl.py` (SMPL+D, 관절은 β 만으로), `leg_data_term.py` (하의 한쪽 데이터 항), `far_spike_refit.py` (튄 프레임 재피팅) |
| 시각화 | `viz_stages.py`, `viz_video.py`, `viz_depth.py`, `viz_pipeline.py`, `depth_viewer.py` (Open3D, 로컬) |

## `NB_RUN_ALL` 주요 옵션 (환경변수로 스크립트에 전달)

| 옵션 | 환경변수 | 효과 | 결과 폴더 꼬리표 |
|---|---|---|---|
| `PERSONAL_MESH` | `WALK_PERSONAL` | 개인 메쉬 (β + D) 사용 | `_personal` |
| `CLOTH_ONESIDED` | `WALK_CLOTH_ONESIDED` | 하의 정점은 "몸이 옷 밖으로 나오면 벌점" 만 (허용 6 cm) | `_c1` |
| `QUICK_FRAMES` | `WALK_FRAME_LIMIT` | 앞쪽 N 프레임만 (빠른 확인) | `_q{N}` |
| `LOCAL_CACHE` | `WALK_DEPTH_SEQ`, `WALK_VIDEO_SEQ` | NB5 부터 depth/video 를 Colab 로컬 복사본에서 읽음 | — |

꼬리표가 붙은 결과는 별도 폴더(`04_fit_personal_c1_q84` 등)에 저장되어 기준 결과를 덮어쓰지 않습니다. 옵션을 모두 끄면 이전과 같은 결과가 나옵니다 (데이터 항: 이전 공식과 차이 0 확인, `personal_smpl`: D = 0 이면 smplx 와 정점·관절 차이 0).

## 이식 검증 (2026-10-02, 로컬)

- D 단계: 4/20 입력으로 기존 산출과 **전 키 일치**.
- C 단계 (CPU, 4/20 두 프레임): A0 위치 차 ≤ 4 mm, 잔차 동일, 무릎각 15.83° / 10.32° 로 기존 값과 일치.
- `extract_session.py`: 4/20 `.r3d` 에서 기존 추출본과 비트 단위 일치.
- `imu_validate.py`: 실제 IMU + 합성 영상 (offset 30.000 s, bias +3°, noise 2°) → offset 30.000 s 정확히 복원.

## 주의

- **Drive 용량**: NB2 + NB3 ≈ 40 ~ 45 MB / 프레임 (250 프레임 ≈ 11 GB). NB3 마지막 셀로 `.bak` 삭제 가능.
- **60 fps 영상**은 `STEP = 2`. 프레임 단위 파라미터(창 길이 등)가 30 fps 에서 검증됐기 때문.
- **IMU**: Xsens MTw 실제 100 Hz. 정지 기립 2 초로 장축 캘리브레이션. 골반 센서가 없어 무릎만 비교.
- 같은 세션 이름으로 구간을 바꿔 다시 돌리면 이전 프레임 파일이 남아 NB3 에 섞입니다 → **구간을 바꾸면 세션 이름도 바꾸기**.
