# OpenWorldTactile_v7: 数据质量迭代

## Goal

让 HDF5 数据不只是能保存，而是稳定、连续、有接触结构，适合后续训练或分析。

V7 要解决的问题：

```text
V6 只证明单个 episode 能写入。
真实使用需要多 episode 稳定性、触觉可分辨性和参数配置。
```

V7 不解决的问题：

```text
不做真实牛顿标定。
不改变 HDF5 数据契约。
不把 sim_constitutive_force 宣称为真实力。
```

## Current State

V6 之后应具备：

```text
HDF5 episode 可写入
observations/tactile/fxyz shape 正确
metadata 正确
至少一个 episode 可复查
```

V7 在此基础上做批量质量分析。

## Key Tasks

任务：

```text
1. 批量采集多个 episode。
2. 统计 fxyz 分布。
3. 对比 grasp / preinsert / insert / recovery 阶段触觉差异。
4. 调整膜参数、splat 参数、接触速度。
5. 调整 marker 参数、camera 参数和 tracking 参数。
6. 形成 recommended_params.json。
7. 形成 quality_report.md。
```

建议最小批量：

```text
smooth_insert: 5 episodes
xy_offset: 5 episodes
edge_search: 5 episodes
partial_jam: 5 episodes
```

## Tunable Parameters

重点调参：

```text
normal_stiffness
normal_damping
shear_stiffness
shear_damping
friction_mu
front_segments_y
front_segments_z
tet_edge_length_r
splat_sigma_px
camera_width
camera_height
marker_spacing_mm
marker_radius_mm
marker_detection_threshold
marker_tracking_max_disp_px
shear_interpolation_radius_px
insert speed
grasp threshold
recovery motion speed
```

调参顺序：

```text
1. 先调 UIPC 稳定性。
2. 再调 fz 接触/非接触区分度。
3. 再调 fx/fy 方向性。
4. 再调纹理/空间分辨率。
5. 最后调输出尺度。
```

## Quality Metrics

必须统计：

```text
fxyz_nan_count
fxyz_max_abs
fz_contact_mean
fz_free_mean
fz_contact_to_free_ratio
fx_direction_consistency
fy_direction_consistency
max_conservation_error
empty_area_ratio
spike_frame_count
episode_success_rate
uipc_failure_count
marker_detection_count_mean
marker_detection_count_std
marker_tracking_loss_rate
marker_displacement_free_mean
marker_displacement_contact_mean
shear_confidence_mean
camera_surface_fz_consistency
```

关键质量判断：

```text
接触阶段 fz 明显高于空载。
fx/fy 在偏心或摩擦接触中有方向性。
marker tracking 在接触阶段稳定。
无接触 marker displacement 接近零。
不同 recovery case 有不同触觉模式。
无大面积空洞。
无随机尖峰。
连续 episode 不崩溃。
```

## Outputs

建议输出目录：

```text
/tmp/openworldtactile_uipc_v7_quality
```

输出文件：

```text
quality_report.md
quality_summary.json
recommended_params.json
sample_previews/
case_comparison_plots/
```

`recommended_params.json` 至少包含：

```text
normal_stiffness
normal_damping
shear_stiffness
shear_damping
friction_mu
front_segments_y
front_segments_z
tet_edge_length_r
splat_sigma_px
camera_width
camera_height
marker_spacing_mm
marker_radius_mm
marker_detection_threshold
marker_tracking_max_disp_px
shear_interpolation_radius_px
```

## Run Commands

批量采集：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_insert_v2_hdf5.py \
  --dataset_dir /tmp/openworldtactile_uipc_v7_dataset \
  --max_episodes 20
```

质量分析：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/analyze_openworldtactile_uipc_dataset_quality.py \
  --dataset_dir /tmp/openworldtactile_uipc_v7_dataset \
  --output_dir /tmp/openworldtactile_uipc_v7_quality
```

## Acceptance Criteria

V7 通过条件：

```text
至少 20 个 episode 可写入并可打开。
所有 episode fxyz 无 NaN。
连续 episode 无 UIPC 崩溃。
接触阶段 fz 明显高于空载阶段。
fx/fy 在偏心或摩擦阶段有方向性。
marker tracking loss rate 在可接受范围内。
shear_confidence 在接触区域足够高。
force map 无大面积空洞。
force map 无明显随机尖峰。
recommended_params.json 固定。
quality_report.md 明确记录参数和结论。
```

## Failure Criteria

V7 失败条件：

```text
多 episode 后 UIPC 不稳定。
触觉响应不可重复。
force map 大面积空洞。
接触和非接触无法区分。
fx/fy 方向完全不稳定。
marker tracking 大量丢失。
shear_confidence 长期过低。
recommended_params 无法收敛。
```

## Next Version Gate

进入 V8 前必须满足：

```text
数据质量稳定。
推荐参数固定。
HDF5 数据可用于后续训练或分析。
仍明确 force_units = sim_constitutive_force。
```
