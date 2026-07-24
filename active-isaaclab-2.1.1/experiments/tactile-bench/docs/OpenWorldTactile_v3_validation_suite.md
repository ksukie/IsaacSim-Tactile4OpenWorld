# OpenWorldTactile_v3: 自动化验证套件

## Goal

把 V2.3 的 camera-observed fxyz 链路变成自动检查，不再只靠肉眼看 preview。

V3 要解决的问题：

```text
单次运行只能说明脚本能跑。
后续版本需要稳定的回归基准。
因此必须有一套固定 validation suite 来自动判断 pass/fail。
```

V3 不解决的问题：

```text
不接 Piper。
不写 HDF5。
不改变力估计算法。
不重新设计 marker tracking。
不做真实力标定。
```

## Current State

V2.3 之后应具备：

```text
openworldtactile_uipc_force.py
openworldtactile_camera_membrane.py
openworldtactile_marker_tracking.py
OpenWorldTactile_v2_3.py
camera depth -> fz 可用
RGB marker tracking -> fx/fy 可用
surface-reference force estimator 可作为对照
```

V3 在这个基础上批量运行固定膜 camera bench。

## Key Changes

新增验证脚本：

```text
experiments/tactile-bench/run_openworldtactile_uipc_validation_suite.py
```

脚本职责：

```text
1. 定义固定 validation cases。
2. 默认调用 OpenWorldTactile_v2_3.py 批量运行。
3. 收集每个 case 的 metadata.json、fxyz.npy、marker overlay 和 shear_map。
4. 自动计算 pass/fail。
5. 输出 validation_summary.json。
```

## Validation Cases

固定 case：

```text
empty_no_contact
indent_0p2mm
indent_0p5mm
indent_1p0mm
rub_positive_y
rub_negative_y
shape_sphere
shape_cylinder
shape_dots
shape_cross_lines
shape_wave1
shape_random
shape_texture_stamp
camera_depth_indent
marker_no_contact
marker_rub_positive_y
marker_rub_negative_y
marker_texture_stamp_rub
```

建议每个 case 使用较低分辨率 smoke 参数先跑通：

```text
tactile_width = 64
tactile_height = 64
front_segments_y = 8
front_segments_z = 8
thickness_segments = 2
```

稳定后再提高到正式参数：

```text
tactile_width = 300
tactile_height = 300
front_segments_y >= 96
front_segments_z >= 120
```

## Checks

每个 case 必须检查：

```text
fxyz.npy 存在
metadata.json 存在
fxyz shape 正确
fxyz 无 NaN
metadata 中 sdf_used_for_force = false
metadata 中 force_source = camera_depth_rgb_marker_tracking 或 camera_marker
metadata 中 normal_source = camera_depth
metadata 中 shear_source = rgb_marker_tracking
metadata 中 force_units = sim_constitutive_force
observed_rgb_sequence.mp4 或 marker_tracking_overlay.mp4 可生成
compression_map / shear_map shape 正确
```

跨 case 检查：

```text
empty_no_contact 的 max(abs(fxyz)) 接近零
indent_0p2mm / indent_0p5mm / indent_1p0mm 的 sum(fz) 单调增加
rub_positive_y 与 rub_negative_y 的 sum(fx) 符号相反
marker_no_contact 的 mean marker displacement 接近零
marker_rub_positive_y / marker_rub_negative_y 的 marker displacement 符号相反
camera depth fz 与 surface-reference fz 趋势一致
dots / cross_lines / wave1 / random / texture_stamp 的 preview_force.png 非空
texture_stamp 的 compression_map 或 fz preview 能看到非光滑结构
```

## Output Layout

建议输出目录：

```text
/tmp/openworldtactile_uipc_v3_validation
```

目录结构：

```text
/tmp/openworldtactile_uipc_v3_validation/
  validation_summary.json
  validation_summary.md
  case_outputs/
    empty_no_contact/
      fxyz.npy
      metadata.json
      preview_force.png
      marker_tracking_overlay.mp4
      shear_map.npy
    indent_0p2mm/
      fxyz.npy
      metadata.json
      preview_force.png
    ...
```

## Summary Schema

`validation_summary.json` 至少包含：

```text
overall_status
case_status
case_metrics
monotonic_fz_check
rub_sign_check
marker_detection_check
marker_displacement_zero_check
marker_rub_sign_check
camera_surface_consistency_check
nan_check
failed_reasons
```

`overall_status` 只允许：

```text
pass
fail
blocked
```

## Run Command

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/run_openworldtactile_uipc_validation_suite.py \
  --output_dir /tmp/openworldtactile_uipc_v3_validation \
  --mode smoke
```

正式高分辨率：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/run_openworldtactile_uipc_validation_suite.py \
  --output_dir /tmp/openworldtactile_uipc_v3_validation_full \
  --mode full
```

## Acceptance Criteria

V3 通过条件：

```text
所有 smoke case 完成。
所有 case 无 NaN。
空载接近零。
fz 随压入深度单调增加。
+Y / -Y rub 下 sum(fx) 符号反转。
marker detection 数量稳定。
无接触 marker displacement 接近零。
camera depth fz 与 surface-reference fz 趋势一致。
纹理 case preview 非空且无大片空洞。
texture_stamp 能显示物体纹理导致的局部形变结构。
validation_summary.json overall_status = pass。
```

## Failure Criteria

V3 失败条件：

```text
任一核心物理趋势失败。
任一 case 出现 NaN。
纹理 case 完全看不到结构。
marker tracking 大量丢失或误匹配。
+Y / -Y rub 的 fx 不反向。
camera depth fz 与 surface-reference fz 完全不一致。
任一 case 运行崩溃且无法自动记录 blocked 原因。
```

## Next Version Gate

进入 V4 前必须满足：

```text
V3 smoke suite 通过。
至少一个 full resolution case 通过。
marker overlay / shear_map artifacts 可复查。
validation_summary.json 可复现。
```
