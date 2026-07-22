# OpenWorldTactile_v6: HDF5 数据采集链路

## Goal

把 V5 的 fxyz 接入 HDF5 recorder，生成可用 episode 数据。

V6 要解决的问题：

```text
V5 只证明插孔 debug 时 fxyz 行为合理。
真正的数据集需要 HDF5 写入、帧同步、metadata 正确。
```

V6 不解决的问题：

```text
不做大规模数据质量优化。
不做真实力标定。
不改变 V5 的触觉来源。
```

## Current State

V5 之后应具备：

```text
插孔 episode 可运行
接触阶段 fxyz 非零
非接触阶段 fxyz 回落
SDF 不生成最终 fxyz
```

V6 在这个基础上接入 recorder。

## Key Changes

新增脚本：

```text
experiments/tactile-bench/OpenWorldTactile_insert_v2_hdf5.py
```

脚本职责：

```text
1. 复用 V5 插孔流程。
2. 接入 HDF5EpisodeRecorder。
3. 将每帧 fxyz 写入 recorder。
4. 保存 observations/tactile/fxyz。
5. 更新 tactile metadata。
6. 保存至少 1 个成功 episode。
```

## HDF5 Data Contract

触觉数据路径固定：

```text
observations/tactile/fxyz
```

shape 固定：

```text
T x 300 x 300 x 3
```

通道固定：

```text
channel_names = ["fx_local_y", "fy_local_z", "fz_local_x_normal_pressure"]
```

metadata 必须写：

```text
tactile_force_model = deformation_based_constitutive_uipc_camera_marker
tactile_contact_source = uipc_membrane_deformation_camera_observed_depth_rgb_marker
normal_source = camera_depth
shear_source = rgb_marker_tracking
tactile_sdf_generates_force = false
tactile_rgb_texture_used_for_force = true
tactile_rgb_used_for_normal_force = false
tactile_rgb_used_for_shear_force = true
tactile_visual_skin_enabled = false
tactile_uipc_conservative_splat_used = false
surface_reference_force_saved = optional
force_units = sim_constitutive_force
normalized = false
frame = tactile sensor local frame
```

建议同时保存或引用 debug 中间量：

```text
observations/tactile/compression_map
observations/tactile/shear_map
observations/tactile/shear_confidence
observations/tactile/marker_tracks
observations/tactile/marker_valid_mask
```

必须保留相机/触觉标定信息：

```text
T_world_tactile
T_camera_top_tactile
T_camera_wrist_tactile
T_link_tactile
T_tactile_link
corners_tactile
active_size_m
resolution
```

## Run Command

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_insert_v2_hdf5.py \
  --dataset_dir /tmp/openworldtactile_uipc_v6_dataset \
  --max_episodes 1
```

## Validation Commands

HDF5 检查脚本建议新增：

```text
experiments/tactile-bench/check_openworldtactile_uipc_hdf5.py
```

检查命令：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/check_openworldtactile_uipc_hdf5.py \
  --dataset_dir /tmp/openworldtactile_uipc_v6_dataset
```

## Acceptance Criteria

V6 通过条件：

```text
能保存 1 个成功 episode。
HDF5 文件可打开。
observations/tactile/fxyz 存在。
fxyz shape = T x 300 x 300 x 3。
fxyz 无 NaN。
相机、关节、触觉帧数一致。
metadata 不再包含旧 SDF 生成力描述。
metadata 明确记录 normal_source = camera_depth。
metadata 明确记录 shear_source = rgb_marker_tracking。
force_units = sim_constitutive_force。
```

## Failure Criteria

V6 失败条件：

```text
HDF5 写入失败。
帧数不同步。
metadata 仍宣称 SDF 生成力。
metadata 仍宣称 RGB 完全不参与力估计。
fxyz 全程为零。
fxyz 出现 NaN。
触觉数据路径或 shape 不符合约定。
```

## Next Version Gate

进入 V7 前必须满足：

```text
至少 1 个 HDF5 episode 完整可用。
HDF5 checker 通过。
metadata 与新的 UIPC camera-marker deformation force 模型一致。
```
