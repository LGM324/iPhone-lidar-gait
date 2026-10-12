# iPhone LiDAR 측면 영상 한 대로 보행 하체 관절각 추정

**Single side-view iPhone LiDAR + RGB → personalized SMPL → knee-angle time series (treadmill gait)**

<p align="center"><img src="docs/media/fit.gif" width="860" alt="좌우 다리 구분, SMPL 투영, 옆·위에서 본 depth 와 SMPL"></p>

> [English](README_en.md)
>
> 개인 프로젝트 (2026.03 – 진행 중) · 촬영, 데이터 수집, 파이프라인 설계·구현·검증 모두 단독 수행
> 재활 로봇 보조 응용을 목표로, 마커 없이 **스마트폰 한 대**로 보행 중 무릎·고관절·발목 각도를 얻는 것이 목표입니다.
> 2026년 3월부터 로컬에서 개발했고, 2026년 10월 Google Colab 으로 이식하면서 GitHub 로 옮겼습니다 (그 이전 이력은 [DECISIONS.md](docs/DECISIONS.md) 에 기록).

### 결과

| 항목 | 옷 처리 전 | **최종** | 조건 |
|---|---|---|---|
| 무릎각 vs IMU, 카메라 쪽 다리 | 3.40° (r 0.989) | **3.76°** (r 0.993) | 7/29 트레드밀 3 km/h, 250 프레임 (보행 약 6 주기) |
| 무릎각 vs IMU, 가려지는 쪽 다리 | 7.67° (r 0.945) | **4.79°** (r 0.971) | 같은 250 프레임, 측정 239 · 가림 보간 11 |
| 두 다리 평균 RMSE | 5.54° | **4.27°** | 16 조합 실험에서 미리 정한 기준으로 고른 조합 |
| 개인 메쉬, 피팅에 안 쓴 9장 (8방향) | IoU 0.838 · depth 1.59 cm | **IoU 0.925 · depth 1.22 cm** | 맨몸 SMPL → SMPL+D |

<p align="center"><img src="docs/figures/imu_knee_250.png" width="900" alt="IMU 대비 무릎각 250 프레임, 기존과 최종"></p>

> 피험자 1명(본인), 세션 1개. IMU 자체도 광학 모션캡처 대비 2 – 5° 오차가 보고되므로, 위 값은 "IMU 와의 일치도" 입니다. 이 저장소는 결과보다 **과정과 판단 근거**를 정리한 것입니다.

---

## 목차
1. [왜 어려운가](#1-왜-어려운가)
2. [파이프라인 한눈에 보기](#2-파이프라인-한눈에-보기)
3. [단계별 설명](#3-단계별-설명)
4. [문제 해결 사례](#4-문제-해결-사례)
5. [현재 상태](#5-현재-상태)
6. [실행 방법 (Google Colab)](#6-실행-방법-google-colab)
7. [저장소 구조](#7-저장소-구조)
8. [데이터·라이선스](#8-데이터라이선스)

모든 선택지에서 무엇을 시험했고 왜 그것을 골랐는지는 [**docs/DECISIONS.md**](docs/DECISIONS.md) 에 따로 정리했습니다.

> 용어: **NB** = 노트북(Google Colab `.ipynb`). 파이프라인 단계마다 NB1 ~ NB7 로 번호를 붙였습니다.

---

## 1. 왜 어려운가

| 문제 | 내용 | 이 프로젝트의 대응 |
|---|---|---|
| 카메라 한 대 | 측면에서 찍으면 **반대쪽 다리(far leg)가 주기적으로 가려짐** | 가려진 구간을 판정하고, 측정/보간을 프레임마다 라벨로 구분 |
| LiDAR 품질 | iPhone LiDAR 256×192, 노이즈·프레임 간 깜빡임 | 단안 depth 모델(Sapiens)의 모양 + LiDAR 의 절대 거리를 결합 |
| 학습 기반 3D 자세 | 프레임마다 체형·자세가 출렁임 | 개인 체형은 따로 한 번 측정해 고정, 자세는 depth 측정으로 다시 맞춤 |
| 정답 없음 | 마커 모캡 없음 | Xsens IMU 6개 동시 측정으로 검증 |

## 2. 파이프라인 한눈에 보기

<p align="center"><img src="docs/figures/pipeline.svg" width="900" alt="파이프라인 구조도"></p>

| 단계 | 하는 일 | 코드 |
|---|---|---|
| **0 · 개인 메쉬** | 보행 때와 같은 옷·신발로 한 바퀴 돈 영상 → SMPL 체형 β + 옷·신발 표면 D (한 번, 이후 고정) | [`personalization/`](personalization) |
| **A · 추출** | `.r3d` → RGB / LiDAR depth / confidence, 보행 구간 선택 | NB1 |
| **B · 사람 depth** | Sapiens 상대 depth → LiDAR 로 미터 변환 → 세션 전역 scale 재보정 | NB2, NB3 |
| **C · SMPL 하체 fit** | 4D-Humans 초기 자세 → 몸 위치 정렬 → 가까운 다리 → 먼 다리 (반바지는 한쪽 데이터 항) → 튄 프레임 재피팅 | NB4, NB5 |
| **D · 무릎각** | 측정 근거 태깅 → 가려진 구간 보간 → 값·불확실도·라벨 | NB6 |
| **E · IMU 검증** | Xsens MTw 6센서 무릎각과 시간 동기화 후 비교 | NB7 |

## 3. 단계별 설명

### 0 · 개인 메쉬 (체형 + 옷·신발)

보행 영상에서 매 프레임 체형을 추정하면 다리 길이가 출렁이고, 맨몸 SMPL 을 옷 입은 depth 에 맞추면 뼈가 옷 쪽으로 끌려갑니다. 그래서 **보행 때와 같은 옷·신발로 한 바퀴(8방향) 돈 30 fps 영상**에서 정지 프레임 77장을 자동으로 골라, 체형 β 와 옷·신발 표면 D 를 가진 **개인 메쉬(SMPL+D)** 를 한 번 만들고 보행 내내 고정합니다.

<p align="center"><img src="docs/figures/personal_overlay.jpg" width="900" alt="개인 메쉬를 돌기 영상 사진에 겹친 모습"></p>

- 1단계: β 공유 + 프레임별 자세, 실루엣 + 보이는 면 depth, 옷 부분은 "몸이 옷 안쪽" 만 → **점검** (방향 · |β| · 실루엣 키) → 2단계: 법선 방향 변위 D (맨살 ±1, 상의 0 – 5, 신발 0 – 4 cm)
- **피팅에 쓰지 않은 9장 (8방향)** 에서 맨몸 SMPL → SMPL+D: 실루엣 IoU 0.838 → **0.925**, depth 오차 1.59 → **1.22 cm**
- 비교 기준 체형 **b0** = 정자세 4방향 4D-Humans β 평균 + 키 보정 ([재현 코드](personalization/baseline_beta)).
- 자세한 과정: [**personalization/README.md**](personalization/README.md)

### A · 추출

<p align="center"><img src="docs/figures/nb1_extract.png" width="760" alt="RGB, LiDAR depth, confidence"></p>

Record3D `.r3d` (zip) 에서 프레임별 RGB 1440×1920, LiDAR 192×256, confidence 를 꺼냅니다. 60 fps 영상은 프레임 기준 파라미터가 30 fps 에서 검증됐기 때문에 2 프레임마다 1장(STEP = 2)으로 맞춥니다.

### B · 사람 depth (Sapiens + LiDAR)

<p align="center"><img src="docs/figures/nb2_depth.png" width="900" alt="Sapiens 마스크, 상대 depth, 미터 depth, LiDAR 와 옆 단면 비교"></p>

- Sapiens 1B 로 사람 마스크·상대 depth `s`·normal 을 구하고, LiDAR 고신뢰 점으로 `z = a·s + b` 를 RANSAC + MAD 로 맞춥니다.
- 프레임마다 `a` 를 따로 맞추면 `a` 가 0.2 ~ 0.8 로 출렁이고, 그만큼 **몸통 두께가 프레임마다 압축·팽창**합니다 (상관 r = 0.867). 그래서 세션 전체에서 `a` 하나를 정해 다시 적용합니다.

<p align="center"><img src="docs/figures/nb3_reanchor.png" width="760" alt="프레임별 a 와 전역 a, 몸통 두께 재보정 전후, 거리 보존"></p>

전역 `a` 재적용 후 몸통 두께 std 2.73 → 1.36 cm, 사람까지 거리는 최대 0.39 cm 만 변함 (7/29, 3 km/h). 전역 `a` 를 구하는 방식 자체에서 생긴 문제는 [4-1](#4-1-전역-scale-a-가-치우친-이유-심슨-역설).

### C · SMPL 하체 fit

<p align="center"><img src="docs/figures/nb4_init.png" width="760" alt="검출 상자, 4D-Humans 초기 자세"></p>

1. **4D-Humans (HMR 2.0)** 로 프레임별 초기 자세. 체형은 기준 β (b0) 로 고정하고, 신발·상의 표면 D 는 0 단계 개인 메쉬에서 가져옴 (16 조합 실험으로 정한 구성, [4-6](#4-6-옷-처리-변수를-하나씩-떼어-원인-찾기)).
2. **A0 몸 정렬**: 몸 전체 위치·방향만 depth 에 맞춤 (몸통 가중 1.0, 다리 0.3).
3. **좌우 다리 구분**: 측면 촬영이라 카메라 거리 z 가 곧 사람의 좌우 축. 하체 점의 z 를 1D 2-means 로 나누고, 두 무리 간격이 8 cm(다리 두께) 미만이면 "겹침"으로 보고 나누지 않음.
4. **가까운 다리 → 먼 다리** 순서로 각각 고관절·무릎·발목 9-DOF 를 depth 표면에 맞춤 (Huber 3 cm, 카메라에서 보이는 면만, **자세 prior 없이 측정만으로**).
5. **옷 처리**: 헐렁한 반바지 정점은 "몸이 옷 밖으로 나오면 벌점" 한쪽 항만 적용 ([4-6](#4-6-옷-처리-변수를-하나씩-떼어-원인-찾기)). 맨살·신발은 양방향.
6. **튄 프레임 재피팅**: 무릎 > 85° 또는 앞뒤보다 30° 이상 튀면, 앞뒤 정상 프레임 자세를 초기값으로 다시 피팅 ([4-7](#4-7-피팅이-튄-프레임-다시-맞추기)).

<p align="center"><img src="docs/figures/nb5_fit.png" width="900" alt="좌우 구분, SMPL 투영, 옆 단면, 위에서 본 다리"></p>

### D · 무릎각 시계열

<p align="center"><img src="docs/figures/nb6_knee.png" width="760" alt="먼 허벅지 관측 점 수, 부위별 잔차, 최종 무릎각과 라벨"></p>

- 먼 허벅지의 관측 점 수가 기준(N_low) 아래로 떨어지는 구간 = **가려짐** (위 그래프의 빨간 띠).
- 짧은 가림은 앞뒤 측정값으로 보간하고, 정점이 가림 안에 갇히는 경우는 `INTERPOLATED_HIGH_CURV` 로 따로 표시.
- 모든 프레임에 **값 · 라벨(MEASURED / INTERPOLATED / INTERPOLATED_HIGH_CURV) · 보간 불확실도 σ** 를 같이 저장. 7/29 3 km/h 250 프레임: MEASURED 218 / INTERPOLATED 14 / HIGH_CURV 18.

<p align="center"><img src="docs/media/knee.gif" width="760" alt="무릎각 시계열 커서 애니메이션"></p>

### E · IMU 검증

- Xsens MTw Awinda 6개 (양쪽 허벅지·정강이·발, 100 Hz). 정지 기립 2 초로 센서 → 분절 장축을 캘리브레이션해 SMPL 과 같은 정의의 무릎각을 계산.
- 시간 동기화: 영상·IMU 활동량 상호상관 (평활 창 1/2/4 s 로 강건성 확인) + **점프 착지 순간** 앵커 + 무릎각 곡선 미세 정렬. 7/29 3 km/h 세션: offset −16.31 s, 무릎각 곡선 상관 0.98 (모든 비교 조건에서 ±0.01 s 안으로 같음).
- IMU 자체의 안정성: 같은 세션에서 보행 주기별 최대 굴곡 편차 ±0.8°, 정지 구간 ±1.6°, 드리프트 없음.

| 250 프레임, 최종 | RMSE | bias | bias 제거 RMSE | 상관 | 굴곡 범위 SMPL / IMU |
|---|---|---|---|---|---|
| 오른다리 (카메라 쪽) | 3.76° | +3.05° | 2.20° | 0.993 | 56.3 / 61.1° |
| 왼다리 (가려지는 쪽) | 4.79° | −0.54° | 4.76° | 0.971 | 77.1 / 61.2° |

- **오른다리**: 곡선 모양은 잘 맞고(bias 를 빼면 2.20°), 오차 대부분은 디딤기에 무릎을 일정하게 약 3° 더 굽힌 치우침. 옷 처리 전(3.40°)보다 0.36° 나빠짐 — 상의 D 의 영향 (아래 [4-6](#4-6-옷-처리-변수를-하나씩-떼어-원인-찾기)).
- **왼다리**: 치우침은 거의 없어졌지만, 한 프레임만 튀는 오차가 남음 (예: 5698 에서 IMU 6.7° 에 26°, 앞뒤 프레임은 정상). 튐 판정 기준(30°) 아래라 걸리지 않은 경우로, 최대 오차 19.4° 의 원인.

## 4. 문제 해결 사례

### 4-1. 전역 scale `a` 가 치우친 이유: 심슨 역설

<p align="center"><img src="docs/figures/fe_global_a.png" width="900" alt="프레임 안 기울기와 프레임 사이 기울기, 단일 b 와 fixed-effects 비교"></p>

전 프레임의 (s, z) 점을 그냥 모아 `a` 하나, `b` 하나로 맞추니 `a` = 0.385 로, 프레임별 값의 중앙값(0.50)보다 한참 낮았습니다. 보행 중 사람이 카메라 쪽으로 오가며 **프레임마다 b(거리)가 다른데**, 그 차이가 프레임 사이에서 반대 방향의 기울기를 만들어 전체 기울기를 끌어내린 것입니다.
→ **a 공통 + b 프레임별 (fixed-effects)** 로 바꿔 `a` = 0.476. 그 결과 좌우 다리 구분 실패 70 → 25 프레임, 먼 다리 MEASURED 72 → 90%.

### 4-2. 자세 prior 를 없앤 이유 (λ = 0)

자세 prior 세기 0.02 / 0.002 / 0 을 비교하고, 잔차·각도 폭주·jitter·저관측 프레임 네 가지 안전 기준을 걸었습니다. prior 없이도 네 기준을 모두 통과해 (1주기 42 프레임, 발산 0, 잔차 median 1.52 → 1.43 cm) **측정 주도**로 통일했습니다.

### 4-3. 보간 버그를 인위 가림 실험으로 잡기

먼 다리 보간에서 Hermite 끝점 기울기를 창 중앙 기울기로 잘못 쓰고 있었습니다. 정답이 있는 **가까운 다리를 일부러 가리는 실험**으로 편향을 재니 +1.24° (정점이 경계일 때 +2.34°). 2차 fit 끝점 미분으로 고친 뒤 +0.40°, "정점이 가림 안에 갇힘" 판정은 4건 → 2건 (절반이 버그가 만든 허상).

### 4-4. 다리 길이가 틀려서 무릎이 굽나?

다리를 거의 편 디딤기 구간에서 무릎각이 IMU 보다 크게 나와, "개인 체형의 다리가 길어서 무릎을 굽혀 맞추는 것"이라는 가설을 세웠습니다. 디딤기 148 프레임에서 SMPL 발바닥과 depth 발바닥 높이를 비교하니 SMPL 발이 오히려 **2.5 cm 위**(신발 밑창 두께)였고, 높이 차와 편향의 상관은 −0.08. 또 무릎을 2° → 8° 굽혀도 고관절~발목 거리는 2 mm 밖에 줄지 않아, cm 단위 길이 오차로는 몇 도의 편향을 설명할 수 없습니다 → **가설 기각**. 이후 실험에서 **헐렁한 반바지 표면이 허벅지를 끌어당기는 효과**가 디딤기 과굴곡을 키우는 요인임을 확인했습니다 ([4-6](#4-6-옷-처리-변수를-하나씩-떼어-원인-찾기)). 16 조합 실험에서는 하의 한쪽 항이 두 다리의 RMSE 를 모두 줄였습니다.

### 4-5. 깊이 모델 고르기

Depth Pro, PromptDA, Depth Anything v2, Prior-Depth-Anything, Sapiens 를 정자세 4방향에서 키(실측 1.81 m)와 시간축 깜빡임으로 비교했습니다. PromptDA 계열은 키 1.86 – 2.23 m 로 과대, DAv2 는 키 jump std 0.99 cm (기준 0.5 cm 초과) → Sapiens + LiDAR 채택. 자세한 표는 [DECISIONS.md D3](docs/DECISIONS.md).

### 4-6. 옷 처리: 변수를 하나씩 떼어 원인 찾기

<p align="center"><img src="docs/figures/ablation_c0_c5.png" width="820" alt="C0 – C5 비교"></p>

개인 메쉬를 보행에 넣자 **가려지는 왼다리는 좋아졌는데 카메라 쪽 오른다리는 나빠졌습니다.** 한 번에 하나씩 바꿔 IMU 무릎각과 비교했습니다 (002 세션 84 프레임, 잠정).

| 단계 | 바꾼 것 | 확인한 것 |
|---|---|---|
| C0 → C1 | 개인 메쉬 (옷·신발 D) | 왼 8.0 → 6.4°, 오른 3.7 → 4.1° |
| C1 → C2 | 맨몸만 날씬하게 | 오른다리 더 나빠짐 → **"통통한 체형 탓" 기각** |
| C1 → C3 | 반바지 D 만 0 | 오른 디딤기 **과굴곡 +6.4°**: 얇은 허벅지가 반바지 표면으로 끌려감 |
| C3 → C4 | 반바지에 한쪽 데이터 항 | 왼다리 5.1° (최고), 오른 디딤기 과굴곡 +6.4 → +4.2° |
| C0 → C5 | 기존 β 에 한쪽 항만 | 오른 3.7 → **3.0°**, 왼 8.0 → 6.6° → 한쪽 항은 두 다리 모두 개선 |

<p align="center"><img src="docs/figures/cloth_data_term.svg" width="820" alt="양방향 vs 하의 한쪽 데이터 항"></p>

헐렁한 옷은 걸을 때 모양이 바뀌므로 고정 모양으로 넣어도, 빼고 양방향으로 끌어도 뼈 방향이 틀어집니다. 옷 아래 몸을 추정하는 연구들([BUFF](https://arxiv.org/abs/1703.04454), depth 카메라 한 대의 [DoubleFusion](https://arxiv.org/abs/1804.06023))처럼 **"몸은 옷 안쪽에만 있으면 된다"** 는 한쪽 항으로 바꾼 것이 핵심이었습니다.

**16 조합 완전 요인 실험 (β 2 × 표면 D 4 × 데이터 항 2).** 하나씩 바꾸는 실험으로는 요인끼리 섞인 효과를 가를 수 없어서, 모든 조합을 같은 조건(84 프레임)으로 돌렸습니다. 선택 기준(두 다리 RMSE 평균 최소)은 실험 전에 고정했고, 고른 조합을 250 프레임으로 다시 확인했습니다.

<p align="center"><img src="docs/figures/factorial_16.png" width="900" alt="16 조합 RMSE 히트맵"></p>
<p align="center"><img src="docs/figures/main_effects.png" width="680" alt="요인별 주효과"></p>

| 요인 | 오른다리 | 왼다리 | 해석 |
|---|---|---|---|
| 하의 한쪽 항 (양방향 → 한쪽) | −0.85° | **−1.93°** | 가장 큰 요인, 두 다리 모두 개선 |
| 개인 β (기준 → 개인) | +0.49° | +0.25° | 개인 β 가 오히려 나쁨. 옷 입은 채 측정해 옷 두께 일부가 체형으로 들어간 것으로 봄 (T자세 허리 둘레 기준 β 89 cm → 개인 β 99 cm) |
| 신발 D | −0.33° | −0.85° | 두 다리 모두 개선 |
| 상의·머리 D | **+1.11°** | −0.57° | 오른다리는 나빠지고 왼다리는 좋아지는 **맞교환** |

- 최선 = **기준 β + 신발·상의 D + 하의 한쪽 항** (평균 4.10°, 84 프레임). 오른다리만 보면 기준 β + 신발 D + 한쪽 항(2.74°)이 가장 좋지만 왼다리가 5.96° 로, 미리 정한 기준에 따라 평균으로 골랐습니다.
- 250 프레임 확인: 오른 3.76°, 왼 4.79°, 평균 4.27° → 84 프레임 결과(4.10°)와 비슷하게 유지.
- "개인화" 의 이득은 체형 β 가 아니라 **신발·상의 표면과 옷 처리 데이터 항** 에서 나왔습니다. 피험자 한 명이고 이 사람의 체형이 평균에 가까울 수 있어, "개인 β 가 필요 없다" 로 일반화하지는 않습니다.
- 한계: 반복 실행이 없어 0.3° 안쪽의 차이는 우연일 수 있음. 조합을 고른 84 프레임이 250 프레임에 포함돼 있어 완전히 독립된 확인은 아님. 신발 영역에 발목·양말 일부가 포함되고, IMU 스트랩이 신발로 분류되며 정강이 위 스트랩 두께는 모델에 없음 (오른다리 디딤기 치우침의 후보 원인, 검증하지 않음).

### 4-7. 피팅이 튄 프레임 다시 맞추기

관측 점은 충분한데 먼 다리 무릎이 114°, 123° 로 접힌 프레임이 나왔습니다 (보행 최대 굴곡은 60 – 70°). IMU 를 보지 않고 **물리적 범위만으로** 판정합니다: 무릎 > 85°, 또는 앞뒤 2 프레임 중앙값과 30° 이상 차이 (30 fps 에서 한 프레임 만에 튀었다 돌아오는 것은 불가능). 걸린 프레임은 버리지 않고 **앞뒤 정상 프레임 자세를 보간해 초기값으로 다시 피팅** 합니다. 114° → 45°, 123° → 21° 로 정상화됐고, 다시 피팅해도 튀면 그때 가림과 같이 보간합니다.

## 5. 현재 상태

- [x] 0 · 개인 메쉬 (돌기 영상 → SMPL+D, 피팅에 안 쓴 9장 검증)
- [x] A ~ D 파이프라인, Google Colab 으로 이식 (연구실 PC 없이 전 과정 재실행)
- [x] 옷 처리 (하의 한쪽 데이터 항) · 튄 프레임 재피팅
- [x] 체형·표면·옷 처리 16 조합 실험 → 최선 조합 250 프레임 확인
- [x] 2026-07-29 데이터 수집: iPhone 영상 + Xsens MTw 6센서, 트레드밀 1 / 3 / 6 km/h
- [x] IMU 무릎각 계산·동기화 모듈
- [x] IMU 대비 무릎각 정확도 (3 km/h, 250 프레임: 오른 3.76°, 왼 4.79°)
- [ ] 1 / 6 km/h 세션, 다른 피험자
- [ ] 무릎각 → 관절 회전 정리, OpenSim IK 연결

> 이 저장소의 수치는 피험자 1명(본인), 3 km/h 세션 하나의 결과입니다. 다른 속도·사람에서의 정확도는 아직 주장하지 않습니다.

## 6. 실행 방법 (Google Colab)

1. 이 저장소를 Google Drive `MyDrive/walking_0729/` 로 복사 (`pipeline/scripts`, `pipeline/notebooks` → `scripts/`, `notebooks/`).
2. `assets/` 에 **직접 받은** SMPL 모델을 두고, 개인 메쉬는 [`NB_PERSONAL`](personalization/personal_mesh/NB_PERSONAL.ipynb) 로 만듭니다 (아래 [데이터·라이선스](#8-데이터라이선스)).
3. `raw/` 에 Record3D `.r3d`, (선택) Xsens 내보내기 `.txt` 를 둡니다.
4. **`NB_RUN_ALL.ipynb`** 하나로 NB1 → NB7 을 순서대로 실행하거나, NB1 ~ NB7 을 하나씩 실행합니다. GPU 런타임 (T4 이상) 필요. 주요 옵션: `PERSONAL_MESH` (개인 메쉬), `CLOTH_ONESIDED` (하의 한쪽 항), `QUICK_FRAMES` (앞쪽 N 프레임만 빠른 확인). 여러 조건 비교는 [`NB_EXPERIMENTS`](pipeline/notebooks/NB_EXPERIMENTS.ipynb).

```
MyDrive/walking_0729/
├── assets/     models/smpl/SMPL_NEUTRAL.pkl, personal_mesh_*.npz (β + D), smpl_vert_segmentation.json, smpl_lower_cloth_vertices.npy
├── raw/        *.r3d, imu/*.txt
├── scripts/    ← pipeline/scripts
├── notebooks/  ← pipeline/notebooks
└── sessions/   (자동 생성) {SESSION}/00_extract … 06_imu, viz/
```

개인 체형 단계는 [`personalization/README.md`](personalization/README.md) 참고.

**로컬 환경과 테스트**
```bash
pip install -r requirements.txt
python -m pytest tests                                   # SMPL 모델 없이도 실행 (모델 필요한 2개는 건너뜀)
SMPL_MODEL_DIR=/path/to/models python -m pytest tests     # 전체 (smpl/SMPL_NEUTRAL.pkl)
```
테스트: 개인 메쉬가 D = 0 일 때 SMPL 과 정점·관절이 같은지, D 가 뼈 위치를 바꾸지 않는지, 데이터 항을 끄면 이전 공식과 같은지, 하의 한쪽 항의 부호, 튄 프레임 판정 규칙, 옵션별 결과 폴더 분리.

## 7. 저장소 구조

```
├── personalization/          0 · 개인 메쉬 (돌기 영상 → SMPL+D)
│   ├── personal_mesh/        정지 프레임 선택, fit_personal (β → 점검 → D), 옷 영역 판정, 검증, NB_PERSONAL
│   └── baseline_beta/        기준 체형 b0 재현 (4D-Humans β 평균 → 키 보정)
├── tests/                    pytest (개인 메쉬·데이터 항·튐 판정·설정)
├── requirements.txt
├── pipeline/
│   ├── notebooks/            NB0 – NB8, NB_RUN_ALL, NB_EXPERIMENTS (Colab)
│   └── scripts/              walkcfg.py (경로·옵션) + 단계별 스크립트 + personal_smpl / leg_data_term / far_spike_refit + 시각화
└── docs/
    ├── DECISIONS.md          선택지별 근거 기록 (+ DECISIONS_2026-10_personal_mesh.md: 개인 메쉬·옷 처리·튐 재피팅)
    ├── figures/, media/      README 그림 (얼굴 모자이크 처리)
```

## 8. 데이터·라이선스

- **포함하지 않은 것**: SMPL 모델 파일 (재배포 금지 — [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) 에서 직접 받아야 함), 원본 영상·depth·IMU 데이터, 개인 메쉬 파일.
- 그림·영상의 얼굴은 모두 모자이크 처리했습니다.
- 사용한 외부 모델: [Sapiens](https://github.com/facebookresearch/sapiens) (CC BY-NC 4.0), [4D-Humans](https://github.com/shubham-goel/4D-Humans) (MIT), [detectron2](https://github.com/facebookresearch/detectron2) (Apache 2.0), [SMPL](https://smpl.is.tue.mpg.de) (연구용 라이선스). 각 라이선스를 따릅니다.
- 이 저장소의 코드: [MIT](LICENSE).

## 기술 스택

Python · PyTorch · NumPy / SciPy · OpenCV · SMPL / smplx · Meta Sapiens (Seg / Depth / Normal 1B) · 4D-Humans (HMR 2.0) · detectron2 · Open3D · Record3D (iPhone LiDAR) · Xsens MTw Awinda · Google Colab · RANSAC / robust 통계 · 최적화 기반 모델 피팅
