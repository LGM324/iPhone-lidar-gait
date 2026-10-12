# smpl_pipeline

## II-8 STAND SMPL Metric Initialization

This step converts a strict four-view STAND HMR2 / 4D-Humans raw bundle and
Part-I metric depth into an initializer for the later II-10 optimizer. It does
not run iterative fitting, point-cloud fusion, canonicalization, or depth edits.

```bash
python -m smpl_pipeline.ii8_init \
  --raw-hmr2-dir /path/to/init_params_raw \
  --depth-dir /path/to/part1_metric_depth/STAND \
  --k-json /path/to/K.json \
  --smpl-model-dir /path/to/smpl/models \
  --output-dir /path/to/outputs/ii8_stand \
  --depth-y-axis down \
  --rgb-dir /path/to/STAND/rgb_full \
  --gender neutral
```

Use `--depth-y-axis down` when Part-I depth/XYZ uses OpenCV-style positive Y
toward the image bottom. The script flips observed Y into SMPL's y-up camera
convention and records the choice in `init_params.npz` and diagnostics.

On this machine, the previous SMPL fitting environment can run this step:

```powershell
C:\Users\User\Desktop\walking_r3d\venv_smpl\Scripts\python.exe -m smpl_pipeline.ii8_init `
  --raw-hmr2-dir C:\Users\User\Desktop\walking_r3d\0420\inputs_local\init_params_raw `
  --depth-dir C:\Users\User\Desktop\walking_r3d\0420\Stand\STAND_Lidar_DAv2_align_all\DAv2필터링_refine_tailcut\work\out `
  --k-json C:\Users\User\Desktop\walking_r3d\0420\Stand\STAND_Lidar_DAv2_align_all\inputs\intrinsics.json `
  --smpl-model-dir <SMPL_MODEL_DIR> `
  --output-dir C:\Users\User\Desktop\walking_r3d\0420\outputs\ii8_stand `
  --depth-y-axis down `
  --rgb-dir C:\Users\User\Desktop\walking_r3d\0420\Stand\[Result]00_extract\rgb `
  --gender neutral
```

Required outputs:

- `init_params.npz`
- `init_diagnostics.json`
- `debug/*_ii8_debug.png` when `--rgb-dir` is provided

Before feeding the result into II-10, manually inspect debug overlays and verify
that RGB/depth dimensions and filename matching are correct.

## II-10 Depth Fitting

II-10 consumes `init_params.npz` and refines the four STAND views with:

- shared `betas`
- per-view `body_pose`
- per-view `global_orient`
- per-view `transl`

The first implementation uses projected SMPL vertex Z-depth consistency with
strong priors to II-8. It is a conservative fitting pass, not a final
silhouette-perfect lower-limb optimizer.

```powershell
C:\Users\User\Desktop\walking_r3d\venv_smpl\Scripts\python.exe -m smpl_pipeline.ii10_fit `
  --init-npz C:\Users\User\Desktop\walking_r3d\0420\outputs\ii8_stand\init_params.npz `
  --depth-dir C:\Users\User\Desktop\walking_r3d\0420\Stand\STAND_Lidar_DAv2_align_all\DAv2필터링_refine_tailcut\work\out `
  --partseg-dir C:\Users\User\Desktop\walking_r3d\0420\Stand\STAND_Lidar_DAv2_align_all\DAv2필터링_refine_tailcut\work `
  --k-json C:\Users\User\Desktop\walking_r3d\0420\Stand\STAND_Lidar_DAv2_align_all\inputs\intrinsics.json `
  --smpl-model-dir C:\Users\User\Desktop\walking_r3d\0420\models `
  --output-dir C:\Users\User\Desktop\walking_r3d\0420\outputs\ii10_stand `
  --rgb-dir C:\Users\User\Desktop\walking_r3d\0420\Stand\[Result]00_extract\rgb `
  --depth-y-axis down `
  --gender neutral `
  --device cpu `
  --allow-png-label-fallback
```

Use `--audit-only` before fitting to validate part labels, depth intersections,
and projected sample counts. Use `--max-stage A` to run only root alignment and
write `stage_a_smpl_params.npz`, `stage_a_height_report.json`, and
`stage_a_overlays/`. PNG label fallback is disabled by default; pass
`--allow-png-label-fallback` only for verified single-channel label-index PNGs.
Use `--precalibrate-beta-height --target-height-m 1.810` to run Stage 0 beta-only
height calibration before Stage A. `--height-prior-weight` remains `0.0` by
default and only affects Stage C/D when set nonzero.

Outputs:

- `final_smpl_params.npz`
- `final_config.yaml`
- `loss_history.csv`
- `fit_summary.json`
- `checkpoints/*.pt`
- `meshes/final_*.obj`
- `overlays/*_overlay.png`
- `regions/*_region_debug.png`
- `shoe_export/*_shoe_points.npy`
