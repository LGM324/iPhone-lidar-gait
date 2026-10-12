# 기준 체형 β (b0) 재현

보행 피팅의 기준 체형 **b0** 는 정자세 4방향(앞·오른쪽·뒤·왼쪽) 사진에서 만든 값입니다. 만드는 과정은 두 단계뿐입니다.

1. **4D-Humans 초기값** — 4방향 사진마다 4D-Humans (HMR 2.0) 가 낸 β 를, 다른 방향과 크게 다른 값을 빼고(중앙값 거리 + 3·MAD) 평균합니다 (`ii8_init.py`, `robust_beta_mean`).
2. **키 보정** — 4방향 자세 그대로의 SMPL 메쉬 높이(정점 y 최대 − 최소) 평균이 실측 키 1.81 m 가 되도록 β 만 조정합니다 (`ii10_fit.py` Stage 0, Adam lr 0.005, 100회).
   손실 = 10 · (평균 높이 − 1.81)² + 0.01 · ‖β‖²

그 뒤 depth 에 맞추는 단계(A – D)에서도 β 는 0.03 이하로만 바뀝니다. 즉 **b0 = 4D-Humans 평균 체형 + 키 보정** 입니다. 자세가 들어간 높이를 1.81 m 에 맞추므로 T자세(정자세 직립) SMPL 키는 177.6 cm 가 됩니다.

개인 메쉬(β + 옷·신발 D)와 비교할 때 이 b0 를 "개인화하지 않은 기준" 으로 씁니다 ([../README.md](../README.md), [DECISIONS](../../docs/DECISIONS_2026-10_personal_mesh.md)).

## 재현 결과

이 폴더의 코드로 다시 돌린 β 와, 보행 실험에 쓴 b0 비교:

| | β₀ – β₇ | β₈, β₉ | T자세 키 | 허벅지 L / R | 정강이 L / R |
|---|---|---|---|---|---|
| 4D-Humans 초기값 (보정 전) | \|β\| ≤ 0.24 | | 173.9 cm (자세 높이) | | |
| 재현 | 0.413 … 0.314 | −0.099, 0.089 | 177.4 cm | 39.36 / 40.14 cm | 41.90 / 41.95 cm |
| b0 (보행 실험) | 0.431 … 0.344 | 0.020, 0.172 | 177.6 cm | 39.43 / 40.22 cm | 42.00 / 42.05 cm |

- β₀ – β₇ 차이 ≤ 0.03, β₈·β₉ 는 0.12 / 0.08 차이 (영향이 작은 고차 성분). 무릎각을 정하는 **허벅지·정강이 길이 차이는 0.1 cm 이내**, 키 차이 0.2 cm.
- 재현 4방향 자세 높이: 1.781 / 1.799 / 1.813 / 1.816 m (평균 1.802 m).

## 실행

입력은 S1 (Sapiens 마스크·LiDAR depth), S2 (4D-Humans 4방향) 노트북 산출입니다.

| 단계 | 하는 일 | 파일 |
|---|---|---|
| S1 | 정자세 4방향 Sapiens 마스크 + 미터 depth | [`notebooks/S1_stand_sapiens_depth_colab.ipynb`](notebooks/S1_stand_sapiens_depth_colab.ipynb) |
| S2 | 4D-Humans 4방향 β · 자세 | [`notebooks/S2_stand_4dhumans_init_colab.ipynb`](notebooks/S2_stand_4dhumans_init_colab.ipynb) |
| II-8 | β 평균 + 위치 초기값, 4D-Humans 방향을 관측 좌표(y 반전)로 변환 | `smpl_pipeline/ii8_init.py` |
| 정렬 | 방향마다 위치를 마스크·depth 앞면에 맞춤 | `init_align.py` |
| II-10 | Stage 0 키 보정 → A – D depth 피팅 | `smpl_pipeline/ii10_fit.py` |

```bash
WORK=<S1 산출>/work   # mask/, out/ 포함
python -m smpl_pipeline.ii8_init --raw-hmr2-dir <S2 산출> --depth-dir $WORK/out \
    --k-json <intrinsics.json> --smpl-model-dir <models> --output-dir out/ii8 \
    --depth-y-axis down --gender neutral --device cpu
python init_align.py out/ii8/init_params.npz out/ii8a/init_params.npz --work $WORK --smpl-model-dir <models>
python -m smpl_pipeline.ii10_fit --init-npz out/ii8a/init_params.npz --depth-dir $WORK/out --partseg-dir $WORK \
    --k-json <intrinsics.json> --smpl-model-dir <models> --output-dir out/ii10a \
    --depth-y-axis down --gender neutral --device cpu --allow-png-label-fallback \
    --precalibrate-beta-height --target-height-m 1.81 --height-prior-weight 1.0 --foot-pose-prior-mult 4.0
```

결과: `out/ii10a/final_smpl_params.npz` 의 `betas_shared_final`. 보행 파이프라인에는 `assets/final_single_smpl_params.npz` 의 `betas` 로 넣습니다.

`smpl_pipeline/` 의 II-11 ~ II-17 은 확인용 진단(미리보기, depth 잔차, 실루엣 IoU) 스크립트입니다.
