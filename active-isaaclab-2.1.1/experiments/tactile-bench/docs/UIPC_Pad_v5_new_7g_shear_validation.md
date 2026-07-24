# V5 new 7g-Shear lateral validation

## 1. 目标

验证真实 UIPC 摩擦接触下的链路：

```text
Pad-local +Y/-Y tool motion
  -> simulation/membrane_sim_mesh tangential deformation
  -> frozen 7f displacement semantics
  -> frozen 7g F=KQ estimator
  -> signed Fx/Fy tactile response
```

本阶段只验证相对切向响应是否有效、方向是否正确、正反是否翻转，以及法向通道是否稳定。
不修改 7g 模型，不标定 Newton，不进入 7h。

## 2. 输入数据

物理工况固定为两个独立 run：

```text
Case A:
  normal indentation = 0.2 mm
  friction mu        = 0.3
  lateral motion     = pad +Y, 0.2 mm

Case B:
  normal indentation = 0.2 mm
  friction mu        = 0.3
  lateral motion     = pad -Y, 0.2 mm
```

每个 run 使用 `22 x 26` 已验证结构网格。流程为：法向加载、切向 ramp、切向 hold、保持当前
切向位置解除法向接触、无接触复位、恢复。正负方向分开运行，避免持续接触下的静摩擦锁定
污染工具轨迹。

冻结 7g 只读取：

```text
surface_deformation.npy
vertex_area.npy
front_surface_mask.npy
```

`commanded_lateral_mm.npy`、实际工具轨迹和 phase history 只用于验收，不参与力估计。

## 3. 输出数据

统一输出目录 `shear_validation/` 包含：

```text
force_pad_local.npy
tactile_force_channels.npy
shear_displacement.npy
shear_direction_error.json
shear_response_metrics.json
verdict.json
```

当前真实验证输出位于：

```text
/tmp/openworldtactile_uipc_v5_new_7g_shear_validation
```

## 4. 修改范围

新增文件：

```text
OpenWorldTactile_v5_new_7g_lateral_shear_deformation_probe.py
OpenWorldTactile_v5_new_7g_lateral_shear_validation.py
test_v5_new_7g_lateral_shear_validation.py
```

未修改：

```text
UIPC_Pad.usda
7f deformation contract
7g F=KQ estimator
7g gains and activation thresholds
link8 direct mount and calibrated Pad pose
```

## 5. 验收实验

### 5.1 正向 UIPC deformation

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7g_lateral_shear_deformation_probe.py \
  --headless \
  --no_save_camera_rgb \
  --lateral_direction positive \
  --pre_contact_frames 2 \
  --ramp_frames 4 \
  --level_hold_frames 6 \
  --lateral_ramp_frames 10 \
  --lateral_hold_frames 8 \
  --recovery_frames 4 \
  --gripper_settle_steps 2 \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7g_shear_positive \
  --workspace_dir /tmp/openworldtactile_uipc_v5_new_7g_shear_positive_ws \
  --fail_on_verdict_fail
```

### 5.2 反向 UIPC deformation

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7g_lateral_shear_deformation_probe.py \
  --headless \
  --no_save_camera_rgb \
  --lateral_direction negative \
  --pre_contact_frames 2 \
  --ramp_frames 4 \
  --level_hold_frames 6 \
  --lateral_ramp_frames 10 \
  --lateral_hold_frames 8 \
  --recovery_frames 4 \
  --gripper_settle_steps 2 \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7g_shear_negative \
  --workspace_dir /tmp/openworldtactile_uipc_v5_new_7g_shear_negative_ws \
  --fail_on_verdict_fail
```

### 5.3 冻结 7g estimator

分别把两个 probe 的 `surface_deformation.npy` 传给
`OpenWorldTactile_v5_new_7g_deformation_force_estimator.py`，参数保持冻结值，并使用前两帧
无接触记录计算相对基线。

```bash
python experiments/tactile-bench/OpenWorldTactile_v5_new_7g_deformation_force_estimator.py \
  --contract_dir /tmp/openworldtactile_uipc_v5_new_7f_contract_verified \
  --displacement_path /tmp/openworldtactile_uipc_v5_new_7g_shear_positive/surface_deformation.npy \
  --baseline_frame_count 2 \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7g_shear_positive_force \
  --fail_on_verdict_fail

python experiments/tactile-bench/OpenWorldTactile_v5_new_7g_deformation_force_estimator.py \
  --contract_dir /tmp/openworldtactile_uipc_v5_new_7f_contract_verified \
  --displacement_path /tmp/openworldtactile_uipc_v5_new_7g_shear_negative/surface_deformation.npy \
  --baseline_frame_count 2 \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7g_shear_negative_force \
  --fail_on_verdict_fail
```

### 5.4 合并验收

```bash
python \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7g_lateral_shear_validation.py \
  --positive_probe_dir /tmp/openworldtactile_uipc_v5_new_7g_shear_positive \
  --positive_force_dir /tmp/openworldtactile_uipc_v5_new_7g_shear_positive_force \
  --negative_probe_dir /tmp/openworldtactile_uipc_v5_new_7g_shear_negative \
  --negative_force_dir /tmp/openworldtactile_uipc_v5_new_7g_shear_negative_force \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7g_shear_validation \
  --fail_on_verdict_fail
```

## 6. 验收指标

### 6.1 有效切向响应

从法向 hold 的 Pad-local 切向输出估计噪声 `F_noise`。切向命令超过最大命令 10% 的帧定义为
active shear frame：

```text
shear_active_response_ratio
  = count(||Delta Ft|| > F_noise) / active_frame_count
```

要求严格大于 `95%`。

### 6.2 方向一致性

每个 active frame 计算：

```text
cos(theta) = Delta Ft dot d / (||Delta Ft|| * ||d||)
```

正向使用 `d=[+1,0]`，反向使用 `d=[-1,0]`。合并均值要求严格大于 `0.9`。

### 6.3 正反符号翻转

```text
sign_flip_rate
  = count(Delta Ft dot d > 0) / active_frame_count
```

要求严格大于 `95%`。

### 6.4 法向污染

使用 active shear frames 的 RMS 比值：

```text
normal_pollution = RMS(Delta Fn) / RMS(||Delta Ft||)
```

要求严格小于 `0.2`。

## 7. 通过标准与达标效果

四项指标必须同时通过，且正负 deformation probe 与冻结 7g estimator 的 source verdict 均为
PASS。

真实结果：

```text
positive actual lateral displacement       +0.199970 mm
negative actual lateral displacement       -0.199993 mm
minimum contact vertices during shear       9 / 9

shear_active_response_ratio                 1.000000
direction_cosine_similarity                 0.998433
sign_flip_rate                              1.000000
normal_pollution DeltaFz/DeltaFt             0.062568

positive release tangent                    0.0 TU
negative release tangent                    0.0 TU
```

结论：

```text
7g-Shear lateral validation = PASS
```

这证明冻结 7g estimator 在当前已验证工况下同时支持 normal tactile 与带符号 shear tactile。
输出仍是 relative tactile unit，不是 Newton。

## 8. 不通过时说明什么

```text
active response <= 95%:
  deformation 太弱、激活阈值不适合，或接触没有维持；不能进入 7h。

direction cosine <= 0.9:
  坐标轴、object_on_sensor 符号、顶点 correspondence 或摩擦拖动方向错误。

sign flip rate <= 95%:
  正反向通道未保留符号，或切向响应被法向/网格偏置主导。

normal pollution >= 0.2:
  工具法向轨迹不稳定、接触面积变化过大，或离散化产生显著法向耦合。

probe source FAIL:
  先修复工具跟踪、接触维持或恢复；禁止通过调整 7g 增益掩盖物理 probe 失败。
```

## 禁止事项

本阶段及后续禁止：

```text
penetration -> force
World/UIPC_RuntimeMounted/Membrane 作为计算膜
native contact gradient 作为 force source
visual/membrane_camera_surface 参与力计算
abs(Ft) 替代带符号 Fx/Fy
为通过剪切验收而修改冻结 7g 模型或降低附件阈值
```
