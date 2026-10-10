# 0 · 개인 체형 (STAND 4방향 → SMPL β)

보행 분석에 쓸 **SMPL 체형 β (10개)** 를 정자세 4방향 촬영에서 한 번 구합니다. 여기서 나온 `final_single_smpl_params.npz` 를 보행 파이프라인의 `assets/` 에 넣으면 NB4 ~ NB6 이 β 를 고정값으로 씁니다.

<p align="center"><img src="../docs/figures/stand_personalization.png" width="900" alt="STAND 4방향 RGB, 실루엣 비교, depth 점과 최종 SMPL"></p>

## 촬영

- 트레드밀 위 정자세, 카메라 고정, 사람이 90° 씩 돌아 **앞 · 오른쪽 · 뒤 · 왼쪽** 4방향 (Record3D, iPhone LiDAR).
- 방향마다 흔들림 없는 프레임 1장씩 사용 (예: front 00720, right 00880, back 01021, left 01104).

## 흐름

```
STAND RGB + LiDAR (4방향)
  │  S1_stand_sapiens_depth_colab.ipynb    Sapiens Seg/Depth/Normal + LiDAR anchor → 방향별 미터 depth + 부위 라벨
  │  S2_stand_4dhumans_init_colab.ipynb    4D-Humans → 방향별 초기 SMPL (init_params_raw)
  ▼
ii8_init            초기 SMPL 을 미터 depth 좌표로 옮김                         → init_params.npz
ii10_fit            4방향 공동 fit: β 공유, 자세·방향·위치는 방향별
                    Stage 0 키 사전보정(1.81 m) → A 위치 → B ~ D depth 표면 맞춤 → final_smpl_params.npz
ii12_canonical_fusion  돌아선 4방향 결과를 yaw 로 되돌려 한 좌표계로 (바닥 높이 맞춤)  → 단일 후보
ii13_single_refine  SMPL 하나로 4방향 동시 정제 (β 학습률 0 = 고정)
                    + 하체 raw depth 잔차 (가중 0.5) + 왼쪽 뷰 보정                 → final_single_smpl_params.npz ★
```

진단 도구: `ii15_depth_error_audit` (depth 잔차), `ii17_silhouette_iou_audit` (부위별 실루엣 IoU — 왼쪽 뷰 정렬 문제를 찾은 도구), `ii14` / `ii16` (오버레이·Open3D 뷰어).

## 실행 (로컬, CPU 가능)

`personalization/` 에서 패키지로 실행합니다. 경로는 예시입니다.

```bash
python -m smpl_pipeline.ii8_init \
  --raw-hmr2-dir data/stand/init_params_raw --depth-dir data/stand/depth/out \
  --k-json data/stand/intrinsics.json --smpl-model-dir models \
  --output-dir outputs/ii8_stand --depth-y-axis down --rgb-dir data/stand/rgb --gender neutral

python -m smpl_pipeline.ii10_fit \
  --init-npz outputs/ii8_stand/init_params.npz --depth-dir data/stand/depth/out \
  --partseg-dir data/stand/depth --k-json data/stand/intrinsics.json --smpl-model-dir models \
  --output-dir outputs/ii10 --rgb-dir data/stand/rgb --depth-y-axis down --gender neutral --device cpu \
  --allow-png-label-fallback --precalibrate-beta-height --target-height-m 1.81 --height-prior-weight 1.0

python -m smpl_pipeline.ii12_canonical_fusion \
  --input-dir outputs/ii10 --output-dir outputs/ii12 --smpl-model-dir models

python -m smpl_pipeline.ii13_single_refine \
  --input-dir outputs/ii10 --ii12-dir outputs/ii12 --output-dir outputs/ii13 --smpl-model-dir models \
  --cam2world data/stand/cam2world.npy --iters 220 --w-height 2.0 --w-body-prior 0.05 \
  --w-orient-prior 0.01 --w-transl-prior 0.01 \
  --raw-depth-weight 0.5 --raw-depth-region lower_body_no_shoe \
  --depth-dir data/stand/depth/out --partseg-dir data/stand/depth --allow-png-label-fallback \
  --view-corrections-json config/ii17_left_view_correction_rawdepth_w05.json
```

최종 실행값은 각 단계 출력의 `final_config.yaml` / `*_report.json` 에 기록됩니다 (위 값은 최종 버전 보고서에서 옮김).

## 결과와 한계

| 단계 | 4방향 target 평균 거리 | 하체 z 평균 | 점→메쉬 | 몸통·다리 IoU |
|---|---|---|---|---|
| ii13 yaw-only | 0.132 m | 0.153 m | 0.055 m | 0.332 |
| + raw depth 잔차 | 0.133 m | 0.118 m | 0.055 m | 0.336 |
| + 왼쪽 뷰 보정 (최종) | 0.087 m | 0.111 m | 0.034 m | 0.455 |

- **왼쪽 뷰 보정** (yaw +10°, x −0.25 m, y −0.08 m) 은 실루엣 IoU 붕괴(0.003)를 보고 찾은 **경험값**입니다. 회전 중심 정렬이 진짜 원인일 가능성이 남아 있습니다.
- 줄자 실측은 어깨(견봉 간 39 cm, T자세)와 골반(ASIS 간 27.5 cm)만 있고 다리 길이 실측은 없습니다. 3D 키포인트 기반 길이와 비교하면 허벅지는 1 ~ 2 cm, 정강이는 약 3 cm 차이 (관절 정의 차이 포함).
- 이 단계는 β 만 넘깁니다. 보행 영상과 촬영 세션이 다르므로 옷·신발 차이는 남습니다.
