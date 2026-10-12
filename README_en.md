# Lower-limb gait angles from a single side-view iPhone LiDAR video

**Single side-view iPhone LiDAR + RGB → personalized SMPL → knee-angle time series (treadmill gait)**

<p align="center"><img src="docs/media/fit.gif" width="860" alt="Left/right leg split, SMPL projection, depth and SMPL seen from the side and from above"></p>

> [한국어](README.md)
>
> Personal project (2026.03 – ongoing). I did the capture, data collection, and the pipeline design, implementation, and validation myself.
> The goal is to get knee, hip, and ankle angles during walking from **one smartphone**, without markers, for use with rehabilitation robots.
> Developed locally from March 2026 and moved to GitHub when the pipeline was ported to Google Colab in October 2026. Earlier history is in [DECISIONS.md](docs/DECISIONS.md) (Korean).

### Results

| Item | Before cloth handling | **Final** | Conditions |
|---|---|---|---|
| Knee angle vs IMU, camera-side leg | 3.40° (r 0.989) | **3.76°** (r 0.993) | 2026-07-29 treadmill 3 km/h, 250 frames (about 6 gait cycles) |
| Knee angle vs IMU, occluded leg | 7.67° (r 0.945) | **4.79°** (r 0.971) | Same 250 frames: 239 measured, 11 interpolated through occlusion |
| Mean RMSE of both legs | 5.54° | **4.27°** | Configuration chosen by a criterion fixed before the 16-combination experiment |
| Personal mesh, 9 held-out frames (8 directions) | IoU 0.838 · depth 1.59 cm | **IoU 0.925 · depth 1.22 cm** | Plain SMPL → SMPL+D |

<p align="center"><img src="docs/figures/imu_knee_250.png" width="900" alt="Knee angle vs IMU over 250 frames, before and final (figure labels in Korean)"></p>

> One subject (myself), one session. IMUs themselves differ from optical motion capture by 2 – 5° in published comparisons, so these numbers measure **agreement with the IMU**, not absolute accuracy. This repository documents the **process and the reasoning behind each decision** more than the final number.

Figures carry Korean labels. Detailed write-ups ([personalization/README.md](personalization/README.md), [DECISIONS](docs/DECISIONS.md)) are in Korean.

---

## Contents
1. [Why it is hard](#1-why-it-is-hard)
2. [Pipeline at a glance](#2-pipeline-at-a-glance)
3. [Stages](#3-stages)
4. [Problems solved](#4-problems-solved)
5. [Status](#5-status)
6. [How to run (Google Colab)](#6-how-to-run-google-colab)
7. [Repository layout](#7-repository-layout)
8. [Data and licenses](#8-data-and-licenses)

> **NB** = notebook (Google Colab `.ipynb`). Pipeline stages are numbered NB1 – NB7.

---

## 1. Why it is hard

| Problem | Details | Approach here |
|---|---|---|
| One camera | Filmed from the side, **the far leg is periodically hidden** | Detect occlusion; label every frame as measured or interpolated |
| LiDAR quality | iPhone LiDAR is 256×192, noisy, and flickers between frames | Combine the shape from a monocular depth model (Sapiens) with the absolute distance from LiDAR |
| Learned 3D pose | Body shape and pose jitter from frame to frame | Measure body shape once and fix it; refit the pose to the depth measurements |
| No ground truth | No marker-based mocap | Validate against 6 Xsens IMUs recorded at the same time |

## 2. Pipeline at a glance

<p align="center"><img src="docs/figures/pipeline.svg" width="900" alt="Pipeline diagram"></p>

| Stage | What it does | Code |
|---|---|---|
| **0 · Personal mesh** | A turn-around video in the same clothes and shoes as the walk → SMPL shape β + clothing/shoe surface offsets D (once, then fixed) | [`personalization/`](personalization) |
| **A · Extract** | `.r3d` → RGB / LiDAR depth / confidence; choose the walking segment | NB1 |
| **B · Person depth** | Sapiens relative depth → metric via LiDAR → session-wide scale re-fit | NB2, NB3 |
| **C · SMPL lower-body fit** | 4D-Humans initial pose → body alignment → near leg → far leg (one-sided data term for shorts) → refit of spiking frames | NB4, NB5 |
| **D · Knee angle** | Tag measurement support → interpolate occluded spans → value, uncertainty, label | NB6 |
| **E · IMU validation** | Time-sync with knee angles from 6 Xsens MTw sensors and compare | NB7 |

## 3. Stages

### 0 · Personal mesh (body shape + clothes and shoes)

Estimating body shape per frame makes leg length fluctuate, and fitting a naked SMPL to clothed depth pulls the bones toward the clothes. So I recorded **a 30 fps turn-around video (8 directions) in the same clothes and shoes as the walk**, selected 77 still frames automatically, and built a **personal mesh (SMPL+D)** with shape β and clothing/shoe surface offsets D. It is built once and fixed for the whole walk.

<p align="center"><img src="docs/figures/personal_overlay.jpg" width="900" alt="Personal mesh overlaid on turn-around video frames"></p>

- Step 1: shared β + per-frame pose, fitted to the silhouette and the visible depth surface. For clothing, the only constraint is that the body stays inside it. → **Checks** (orientation, |β|, silhouette height). Step 2: per-vertex normal offsets D (skin ±1 cm, top 0 – 5 cm, shoes 0 – 4 cm).
- On **9 frames not used for fitting (8 directions)**, plain SMPL → SMPL+D: silhouette IoU 0.838 → **0.925**, depth error 1.59 → **1.22 cm**.
- The reference shape **b0** is the 4D-Humans β averaged over 4 standing views, then calibrated to the measured height ([reproduction code](personalization/baseline_beta)).
- Details (Korean): [**personalization/README.md**](personalization/README.md)

### A · Extract

<p align="center"><img src="docs/figures/nb1_extract.png" width="760" alt="RGB, LiDAR depth, confidence"></p>

Per-frame RGB (1440×1920), LiDAR (192×256), and confidence are read from the Record3D `.r3d` (zip). 60 fps videos are subsampled to every second frame (STEP = 2), because the frame-based parameters were validated at 30 fps.

### B · Person depth (Sapiens + LiDAR)

<p align="center"><img src="docs/figures/nb2_depth.png" width="900" alt="Sapiens mask, relative depth, metric depth, side cross-section vs LiDAR"></p>

- Sapiens 1B gives the person mask, relative depth `s`, and normals. High-confidence LiDAR points fit `z = a·s + b` with RANSAC + MAD.
- Fitting `a` per frame makes it swing between 0.2 and 0.8, and **torso thickness compresses and expands frame to frame** by the same amount (r = 0.867). So one `a` is estimated for the whole session and re-applied.

<p align="center"><img src="docs/figures/nb3_reanchor.png" width="760" alt="Per-frame a vs global a, torso thickness before/after, distance preserved"></p>

After re-applying the global `a`, torso-thickness std drops from 2.73 to 1.36 cm, while the distance to the person changes by at most 0.39 cm (2026-07-29, 3 km/h). A problem with how the global `a` was estimated is in [4-1](#4-1-why-the-global-scale-a-was-biased-simpsons-paradox).

### C · SMPL lower-body fit

<p align="center"><img src="docs/figures/nb4_init.png" width="760" alt="Detection box, 4D-Humans initial pose"></p>

1. **4D-Humans (HMR 2.0)** gives the initial pose per frame. Shape is fixed to the reference β (b0); the shoe and top surface offsets D come from the stage-0 personal mesh (the configuration chosen by the 16-combination experiment, [4-6](#4-6-clothing-isolating-the-variables)).
2. **A0 body alignment**: fit only global position and orientation to the depth (torso weight 1.0, legs 0.3).
3. **Left/right leg split**: in a side view, camera distance z is the person's left-right axis. Lower-body points are split by 1D 2-means on z; if the two clusters are closer than 8 cm (one leg's thickness), they are treated as overlapping and not split.
4. **Near leg, then far leg**: fit the 9 DOF of hip, knee, and ankle to the depth surface (Huber 3 cm, camera-visible surface only, **no pose prior, measurement only**).
5. **Clothing**: vertices under the loose shorts get a one-sided term that penalizes only the body coming out through the cloth ([4-6](#4-6-clothing-isolating-the-variables)). Skin and shoes stay two-sided.
6. **Refit of spiking frames**: if the knee exceeds 85° or jumps more than 30° from its neighbors, refit from the pose interpolated between neighboring good frames ([4-7](#4-7-refitting-frames-that-spiked)).

<p align="center"><img src="docs/figures/nb5_fit.png" width="900" alt="Leg split, SMPL projection, side cross-section, legs from above"></p>

### D · Knee-angle time series

<p align="center"><img src="docs/figures/nb6_knee.png" width="760" alt="Far-thigh observed point count, per-part residuals, final knee angle with labels"></p>

- When the number of observed far-thigh points drops below a threshold (N_low), the span is **occluded** (red bands above).
- Short occlusions are interpolated from neighboring measurements. When a peak falls inside an occlusion, the frame is labeled `INTERPOLATED_HIGH_CURV`.
- Every frame stores a **value, a label (MEASURED / INTERPOLATED / INTERPOLATED_HIGH_CURV), and an interpolation uncertainty σ**. For 2026-07-29 3 km/h, 250 frames: MEASURED 218 / INTERPOLATED 14 / HIGH_CURV 18.

<p align="center"><img src="docs/media/knee.gif" width="760" alt="Knee-angle time series with moving cursor"></p>

### E · IMU validation

- 6 Xsens MTw Awinda sensors (both thighs, shanks, and feet, 100 Hz). Two seconds of quiet standing calibrate the sensor-to-segment long axis, so the knee angle has the same definition as in SMPL.
- Time sync: cross-correlation of video and IMU activity (robustness checked with 1/2/4 s smoothing windows), a **jump-landing** anchor, and fine alignment of the knee-angle curves. For the 2026-07-29 3 km/h session: offset −16.31 s, knee-curve correlation 0.98 (identical within ±0.01 s across all compared configurations).
- IMU stability within the session: peak flexion varies ±0.8° across strides, ±1.6° in still segments, no drift.

| 250 frames, final | RMSE | Bias | RMSE without bias | r | Flexion range SMPL / IMU |
|---|---|---|---|---|---|
| Right leg (camera side) | 3.76° | +3.05° | 2.20° | 0.993 | 56.3 / 61.1° |
| Left leg (occluded) | 4.79° | −0.54° | 4.76° | 0.971 | 77.1 / 61.2° |

- **Right leg**: the curve shape matches well (2.20° once bias is removed). Most of the error is a constant ~3° of extra flexion during stance. It is 0.36° worse than before cloth handling (3.40°), an effect of the top-surface D (see [4-6](#4-6-clothing-isolating-the-variables)).
- **Left leg**: the bias is almost gone, but single-frame spikes remain (e.g., frame 5698: 26° vs IMU 6.7°, with normal neighbors). They stay below the 30° spike threshold and cause the 19.4° maximum error.

## 4. Problems solved

### 4-1. Why the global scale `a` was biased: Simpson's paradox

<p align="center"><img src="docs/figures/fe_global_a.png" width="900" alt="Within-frame vs between-frame slopes, single b vs fixed effects"></p>

Pooling (s, z) points from all frames and fitting one `a` and one `b` gave `a` = 0.385, well below the median of the per-frame values (0.50). During walking, the person moves toward and away from the camera, so **each frame has a different b (distance)**. That variation creates a between-frame slope in the opposite direction and drags the pooled slope down.
→ Switched to **a shared `a` + per-frame `b` (fixed effects)**: `a` = 0.476. Left/right split failures dropped from 70 to 25 frames, and far-leg MEASURED rose from 72 to 90%.

### 4-2. Why the pose prior was removed (λ = 0)

I compared pose-prior weights of 0.02, 0.002, and 0 against four safety criteria: residual, angle blow-up, jitter, and low-observation frames. With no prior, all four still passed (one cycle of 42 frames, 0 divergences, median residual 1.52 → 1.43 cm), so the fit is now **driven by measurement only**.

### 4-3. Catching an interpolation bug with artificial occlusion

Far-leg interpolation was using the mid-window slope as the Hermite endpoint slope. I measured the bias by **deliberately occluding the near leg, which has ground truth**: +1.24° (+2.34° when a peak sits at the boundary). After switching to the endpoint derivative of a quadratic fit, the bias became +0.40°. "Peak trapped inside occlusion" cases went from 4 to 2: half had been artifacts of the bug.

### 4-4. Is the knee bent because the leg length is wrong?

In stance, with the leg nearly straight, the knee angle came out larger than the IMU. My hypothesis was that the personal shape's legs were too long, so the fit bent the knee to compensate. Over 148 stance frames, the SMPL sole was actually **2.5 cm above** the depth sole (the shoe sole thickness), and the height difference correlated with the bias at only −0.08. Bending the knee from 2° to 8° shortens the hip–ankle distance by only 2 mm, so centimeter-level length errors cannot explain a bias of several degrees → **hypothesis rejected**. Later experiments showed that **the loose shorts surface pulling the thigh** is a factor in stance over-flexion ([4-6](#4-6-clothing-isolating-the-variables)). In the 16-combination experiment, the one-sided shorts term reduced RMSE for both legs.

### 4-5. Choosing a depth model

I compared Depth Pro, PromptDA, Depth Anything v2, Prior-Depth-Anything, and Sapiens on 4 standing views by height (measured 1.81 m) and temporal flicker. The PromptDA family overestimated height (1.86 – 2.23 m), and DAv2 had a height jump std of 0.99 cm (limit 0.5 cm) → chose Sapiens + LiDAR. Full table in [DECISIONS.md D3](docs/DECISIONS.md).

### 4-6. Clothing: isolating the variables

<p align="center"><img src="docs/figures/ablation_c0_c5.png" width="820" alt="C0 – C5 comparison"></p>

When the personal mesh was first used for walking, **the occluded left leg improved but the camera-side right leg got worse.** I changed one thing at a time and compared against the IMU knee angle (session 002, 84 frames).

| Step | Change | Finding |
|---|---|---|
| C0 → C1 | Personal mesh (clothing and shoe D) | Left 8.0 → 6.4°, right 3.7 → 4.1° |
| C1 → C2 | Slimmer naked body only | Right leg got worse → **"the body is too heavy" rejected** |
| C1 → C3 | Shorts D set to 0 | Right stance **over-flexion +6.4°**: the thin thigh is pulled toward the shorts surface |
| C3 → C4 | One-sided data term for shorts | Left 5.1° (best), right stance over-flexion +6.4 → +4.2° |
| C0 → C5 | One-sided term only, on the reference β | Right 3.7 → **3.0°**, left 8.0 → 6.6° → the one-sided term helps both legs |

<p align="center"><img src="docs/figures/cloth_data_term.svg" width="820" alt="Two-sided vs one-sided data term for the shorts"></p>

Loose clothes change shape while walking. Building them into the mesh with a fixed shape skews the bones, and so does removing them and pulling both ways. The key change was a one-sided term, **"the body only has to stay inside the clothes,"** as in work on estimating the body under clothing ([BUFF](https://arxiv.org/abs/1703.04454); [DoubleFusion](https://arxiv.org/abs/1804.06023) with a single depth camera).

**16-combination full factorial experiment (β 2 × surface D 4 × data term 2).** One-at-a-time changes cannot separate interacting factors, so every combination was run under the same conditions (84 frames). The selection criterion (lowest mean RMSE of both legs) was fixed before running, and the chosen combination was then checked on 250 frames.

<p align="center"><img src="docs/figures/factorial_16.png" width="900" alt="RMSE heatmap of 16 combinations"></p>
<p align="center"><img src="docs/figures/main_effects.png" width="680" alt="Main effect of each factor"></p>

| Factor | Right leg | Left leg | Interpretation |
|---|---|---|---|
| One-sided shorts term (two-sided → one-sided) | −0.85° | **−1.93°** | Largest factor; improves both legs |
| Personal β (reference → personal) | +0.49° | +0.25° | Personal β is slightly worse. Measured in clothes, part of the cloth thickness likely went into the shape (T-pose waist: reference β 89 cm, personal β 99 cm) |
| Shoe D | −0.33° | −0.85° | Improves both legs |
| Top/head D | **+1.11°** | −0.57° | **Trade-off**: worse right leg, better left leg |

- Best = **reference β + shoe and top D + one-sided shorts term** (mean 4.10°, 84 frames). For the right leg alone, reference β + shoe D + one-sided term is best (2.74°), but its left leg is 5.96°, so the pre-set mean criterion chose the former.
- 250-frame check: right 3.76°, left 4.79°, mean 4.27°, close to the 84-frame result (4.10°).
- The gain of "personalization" came from **the shoe and top surfaces and the clothing data term**, not from the body-shape β. With one subject whose shape may be close to average, I do not generalize this to "personal β is unnecessary."
- Limitations: no repeated runs, so differences under ~0.3° may be noise. The 84 frames used for selection are part of the 250 frames, so the check is not fully independent. The shoe region includes part of the ankle and sock; Sapiens labels the IMU straps as shoe, and strap thickness on the shank is not modeled (a candidate cause of the right-leg stance bias, not tested).

### 4-7. Refitting frames that spiked

Some frames had enough observed points but the far knee folded to 114° or 123° (peak walking flexion is 60 – 70°). They are flagged **from physical range only, without looking at the IMU**: knee > 85°, or more than 30° from the median of the 2 neighbors on each side (at 30 fps, a knee cannot jump and return within one frame). Flagged frames are not discarded; they are **refitted from the pose interpolated between neighboring good frames**. 114° → 45°, 123° → 21°. If a frame still spikes after the refit, it is interpolated like an occlusion.

## 5. Status

- [x] 0 · Personal mesh (turn-around video → SMPL+D, validated on 9 held-out frames)
- [x] Stages A – D, ported to Google Colab (whole pipeline reruns without the lab PC)
- [x] Clothing handling (one-sided data term) and refit of spiking frames
- [x] 16-combination experiment on shape, surface, and clothing term → best combination checked on 250 frames
- [x] 2026-07-29 data collection: iPhone video + 6 Xsens MTw sensors, treadmill 1 / 3 / 6 km/h
- [x] IMU knee-angle computation and time sync
- [x] Knee-angle agreement with IMU (3 km/h, 250 frames: right 3.76°, left 4.79°)
- [ ] 1 / 6 km/h sessions, other subjects
- [ ] Knee angle → joint rotations, OpenSim IK

> All numbers come from one subject (myself) and one 3 km/h session. I make no accuracy claims yet for other speeds or other people.

## 6. How to run (Google Colab)

1. Copy this repository to Google Drive `MyDrive/walking_0729/` (`pipeline/scripts`, `pipeline/notebooks` → `scripts/`, `notebooks/`).
2. Put an SMPL model **you downloaded yourself** in `assets/`, and build the personal mesh with [`NB_PERSONAL`](personalization/personal_mesh/NB_PERSONAL.ipynb) (see [Data and licenses](#8-data-and-licenses)).
3. Put the Record3D `.r3d` in `raw/`, and optionally the Xsens export `.txt`.
4. Run NB1 → NB7 with **`NB_RUN_ALL.ipynb`**, or run NB1 – NB7 one by one. Needs a GPU runtime (T4 or better). Main options: `PERSONAL_MESH` (personal mesh), `CLOTH_ONESIDED` (one-sided shorts term), `QUICK_FRAMES` (first N frames for a quick check). To compare configurations, use [`NB_EXPERIMENTS`](pipeline/notebooks/NB_EXPERIMENTS.ipynb).

```
MyDrive/walking_0729/
├── assets/     models/smpl/SMPL_NEUTRAL.pkl, personal_mesh_*.npz (β + D), smpl_vert_segmentation.json, smpl_lower_cloth_vertices.npy
├── raw/        *.r3d, imu/*.txt
├── scripts/    ← pipeline/scripts
├── notebooks/  ← pipeline/notebooks
└── sessions/   (created automatically) {SESSION}/00_extract … 06_imu, viz/
```

For the body-shape stage, see [`personalization/README.md`](personalization/README.md).

**Local environment and tests**
```bash
pip install -r requirements.txt
python -m pytest tests                                   # runs without the SMPL model (2 model tests are skipped)
SMPL_MODEL_DIR=/path/to/models python -m pytest tests     # full run (smpl/SMPL_NEUTRAL.pkl)
```
The tests check that:
- the personal mesh with D = 0 matches SMPL vertices and joints, and D does not move the bones;
- disabling the new data term reproduces the previous formula;
- the one-sided term has the right sign;
- the spike rule flags the right frames;
- each option writes to its own result folder.

## 7. Repository layout

```
├── personalization/          0 · personal mesh (turn-around video → SMPL+D)
│   ├── personal_mesh/        still-frame selection, fit_personal (β → checks → D), clothing labels, validation, NB_PERSONAL
│   └── baseline_beta/        reproduces the reference shape b0 (4D-Humans β average → height calibration)
├── tests/                    pytest (personal mesh, data term, spike rule, config)
├── requirements.txt
├── pipeline/
│   ├── notebooks/            NB0 – NB8, NB_RUN_ALL, NB_EXPERIMENTS (Colab)
│   └── scripts/              walkcfg.py (paths, options) + per-stage scripts + personal_smpl / leg_data_term / far_spike_refit + visualization
└── docs/
    ├── DECISIONS.md          rationale for each choice (+ DECISIONS_2026-10_personal_mesh.md: personal mesh, clothing, spike refit)
    ├── figures/, media/      README figures (faces blurred)
```

## 8. Data and licenses

- **Not included**: SMPL model files (no redistribution; download from [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de)), raw video, depth, and IMU data, and the personal mesh files.
- All faces in figures and videos are mosaicked.
- External models used: [Sapiens](https://github.com/facebookresearch/sapiens) (CC BY-NC 4.0), [4D-Humans](https://github.com/shubham-goel/4D-Humans) (MIT), [detectron2](https://github.com/facebookresearch/detectron2) (Apache 2.0), [SMPL](https://smpl.is.tue.mpg.de) (research license). Each is used under its own license.
- Code in this repository: [MIT](LICENSE).

## Tech stack

Python · PyTorch · NumPy / SciPy · OpenCV · SMPL / smplx · Meta Sapiens (Seg / Depth / Normal 1B) · 4D-Humans (HMR 2.0) · detectron2 · Open3D · Record3D (iPhone LiDAR) · Xsens MTw Awinda · Google Colab · RANSAC / robust statistics · optimization-based model fitting
