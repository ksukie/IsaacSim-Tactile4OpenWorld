# OpenWorldTactile_v2: 力估计核心模块化

## Goal

把 V1 中的三维力估计逻辑从 demo 脚本中抽离，形成一个可复用、可单独测试的核心模块。

V2 要解决的问题：

```text
V1 目前把场景创建、UIPC 运行、力估计、preview、保存全部写在一个脚本里。
后续插孔任务也需要同一套 fxyz 算法。
因此必须先把力估计核心稳定成独立模块。
```

V2 不解决的问题：

```text
不接 Piper。
不写 HDF5。
不做插孔。
不做真实牛顿标定。
不改变 V1 的物理意义和通道定义。
```

## Current State

V1 参考文件：

```text
experiments/tactile-bench/OpenWorldTactile_v1.py
```

V1 已经实现：

```text
UIPC 软膜形变读取
MembraneForceEstimator
局部顶点面积估计
法向/切向本构力计算
摩擦锥裁剪
conservative splat
fxyz.npy / metadata.json / preview 输出
```

V2 已生成独立脚本和 API 接口：

```text
experiments/tactile-bench/OpenWorldTactile_v2.py
experiments/tactile-bench/api/openworldtactile_uipc_force.py
experiments/tactile-bench/api/__init__.py
experiments/tactile-bench/OpenWorldTactile_v2_force_test.py
```

V1 暂时保留为参考版本；V2.py 使用 `api` 中的通用力估计接口。

## Key Changes

新增核心模块：

```text
experiments/tactile-bench/api/openworldtactile_uipc_force.py
```

该模块负责：

```text
1. 保存 rest_surface。
2. 接收 current_surface。
3. 自动识别前表面和背面顶点。
4. 用背面顶点估计 global drift。
5. 计算 compression / shear displacement / velocity。
6. 估计每个前表面顶点局部面积。
7. 计算顶点法向力和切向力。
8. 进行摩擦锥裁剪。
9. 将顶点力 conservative splat 到 H x W x 3。
10. 输出 fxyz、disp_grid、stats。
```

V2 脚本只保留：

```text
场景创建
压头轨迹
UIPC step
读取 surf_nodal_pos_w
调用 api/openworldtactile_uipc_force.py
保存输出
```

V2 不在 demo 脚本内部重复实现 `MembraneForceEstimator`。

## Public Interface

核心类建议仍命名为：

```text
MembraneForceEstimator
```

初始化输入：

```text
rest_surface
width
length
tactile_height
tactile_width
front_eps
normal_stiffness
normal_damping
shear_stiffness
shear_damping
friction_mu
splat_sigma_px
splat_radius_sigmas
dt
```

每帧输入：

```text
current_surface
```

每帧输出：

```text
fxyz: H x W x 3
disp_grid: H x W x 3
stats: dict
```

固定通道顺序：

```text
fxyz[..., 0] = fx_local_y
fxyz[..., 1] = fy_local_z
fxyz[..., 2] = fz_local_x_normal_pressure
```

固定力单位：

```text
sim_constitutive_force
```

## Synthetic Tests

新增不依赖 Isaac 的合成测试脚本：

```text
experiments/tactile-bench/OpenWorldTactile_v2_force_test.py
```

测试输入：

```text
规则矩形前表面顶点
对应背面顶点
人为构造 current_surface
```

测试 case：

```text
1. zero_displacement
   所有点不动，fxyz 应接近零。

2. normal_indent
   前表面 X 方向被压入，fz 应为正。

3. shear_y
   前表面 Y 方向发生切向位移，fx 应非零。

4. shear_z
   前表面 Z 方向发生切向位移，fy 应非零。

5. drift_only
   前后表面整体刚体平移，fxyz 应接近零。

6. conservation
   sum(pixel_fxyz) 与 sum(vertex_force) 相对误差 < 1%。
```

## Run Commands

V2 模块测试：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v2_force_test.py
```

V2 回归验证：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v2.py \
  --shape sphere \
  --approach_steps 1 \
  --indent_steps 1 \
  --rub_steps 0 \
  --release_steps 1 \
  --front_segments_y 8 \
  --front_segments_z 8 \
  --thickness_segments 2 \
  --tet_edge_length_r 0.2 \
  --tactile_width 64 \
  --tactile_height 64 \
  --output_dir /tmp/openworldtactile_uipc_v2_regression
```

## Outputs

模块测试输出：

```text
/tmp/openworldtactile_uipc_v2_force_core_test/summary.json
```

V2 回归输出：

```text
/tmp/openworldtactile_uipc_v2_regression/fxyz.npy
/tmp/openworldtactile_uipc_v2_regression/metadata.json
/tmp/openworldtactile_uipc_v2_regression/preview_force.png
```

## Acceptance Criteria

V2 通过条件：

```text
V2 脚本已完全复用 api/openworldtactile_uipc_force.py。
V2 输出 shape 与 V1 约定一致。
合成 normal_indent 能产生正 fz。
合成 shear_y 能产生 fx。
合成 shear_z 能产生 fy。
drift_only 不产生明显假力。
pixel force 与 vertex force 守恒误差 < 1%。
所有输出无 NaN。
```

## Failure Criteria

V2 失败条件：

```text
V1 输出明显变化且无法解释。
通道顺序混乱。
splat 不守恒。
整体刚体漂移产生大假力。
模块必须启动 Isaac 才能测试。
```

## Next Version Gate

进入 V2.1 / V2.2 前必须满足：

```text
api/openworldtactile_uipc_force.py 接口稳定。
V2.py 已完全复用该模块。
合成测试全部通过。
V2 最小 UIPC 回归能输出 fxyz.npy 和 metadata.json。
```
