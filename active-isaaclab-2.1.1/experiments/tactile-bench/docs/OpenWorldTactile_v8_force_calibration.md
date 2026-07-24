# OpenWorldTactile_v8: 真实力标定

## Goal

把 `sim_constitutive_force` 映射到真实力单位。

V8 要解决的问题：

```text
V1-V7 的力是仿真本构力。
它可以用于相对趋势和触觉结构，但不能直接宣称为牛顿。
V8 通过标定实验建立 sim force 到 Newton 的映射。
```

V8 不解决的问题：

```text
不重新设计 UIPC 膜。
不改变 HDF5 基本数据路径。
不重新定义 fx/fy/fz 通道。
```

## Current State

V7 之后应具备：

```text
稳定 HDF5 数据链路
推荐仿真参数
稳定 fxyz 输出
force_units = sim_constitutive_force
```

V8 在此基础上做真实力标定。

## Calibration Inputs

标定输入可以来自：

```text
known_normal_load
known_shear_load
真实力计数据
真实 OpenWorldTactile/GelSight 对照数据
标准砝码或已知压入力实验
```

最低要求：

```text
至少 3 个不同法向载荷水平。
至少 2 个不同切向方向。
每个载荷水平重复多次。
记录真实力和对应 fxyz。
```

## Calibration Tasks

任务：

```text
1. 设计法向标定实验。
2. 设计切向标定实验。
3. 收集真实力和仿真 fxyz 对应数据。
4. 拟合 sim_constitutive_force 到 Newton 的映射。
5. 评估误差。
6. 输出 calibration_params.json。
7. 更新 HDF5 metadata 标定字段。
```

建议先做简单线性标定：

```text
F_newton = scale * F_sim + bias
```

如果线性误差过大，再考虑：

```text
分通道 scale
分段线性
深度相关 scale
区域相关校正
```

## Outputs

建议输出目录：

```text
/tmp/openworldtactile_uipc_v8_calibration
```

输出文件：

```text
calibration_params.json
calibration_report.md
calibrated_fxyz_samples/
calibration_error_plots/
```

`calibration_params.json` 至少包含：

```text
calibration_version
source_dataset
normal_force_scale
normal_force_bias
shear_force_scale_y
shear_force_scale_z
shear_force_bias_y
shear_force_bias_z
valid_force_range
error_metrics
```

## Metadata Update

完成标定后，新数据或转换后数据可以写：

```text
force_units = calibrated_newton
calibration_version = v1
calibration_source = known_load_or_force_gauge
calibration_params_file = calibration_params.json
```

未完成标定的数据必须继续写：

```text
force_units = sim_constitutive_force
```

## Validation Metrics

必须评估：

```text
normal_force_mae
normal_force_rmse
normal_force_relative_error
shear_force_direction_accuracy
shear_force_mae
repeatability_error
depth_consistency_error
```

## Run Commands

标定拟合：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/fit_openworldtactile_uipc_force_calibration.py \
  --calibration_dataset /path/to/calibration_dataset \
  --output_dir /tmp/openworldtactile_uipc_v8_calibration
```

应用标定：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/apply_openworldtactile_uipc_force_calibration.py \
  --dataset_dir /path/to/sim_constitutive_dataset \
  --calibration_params /tmp/openworldtactile_uipc_v8_calibration/calibration_params.json \
  --output_dir /tmp/openworldtactile_uipc_v8_calibrated_dataset
```

## Acceptance Criteria

V8 通过条件：

```text
法向力误差在项目可接受范围内。
切向力方向正确。
不同压入深度尺度一致。
重复实验误差可接受。
calibration_params.json 可复用。
metadata 明确记录标定版本。
未标定数据和已标定数据能明确区分。
```

## Failure Criteria

V8 失败条件：

```text
sim force 与真实力无稳定关系。
切向力方向不可靠。
不同深度下尺度严重漂移。
重复实验误差过大。
metadata 无法区分标定/未标定数据。
```

## Final Gate

V8 完成后，完整链路达到：

```text
UIPC 软膜形变
-> deformation-based fxyz
-> 插孔 HDF5 数据集
-> 数据质量报告
-> 可选真实力标定
```

此时才允许把部分输出标记为：

```text
calibrated_newton
```
