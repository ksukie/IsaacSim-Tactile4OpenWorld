# OpenWorldTactile_v1 三维力软膜验证流程

## 目标

V1 只验证一条最小、可解释的三维力触觉链路：

```text
UIPC 软膜形变 -> 膜表面位移/速度 -> 本构模型反推顶点力 -> 守恒 splat 到 fxyz 图
```

输出不是 GelSight 风格 RGB，而是：

```text
fxyz: T x H x W x 3
默认 H = W = 300

fx = 膜局部 Y 方向切向力
fy = 膜局部 Z 方向切向力
fz = 膜局部 X 方向法向压力
```

V1 不接 Piper，不做插孔任务，不写 HDF5，只做固定传感器压膜验证。

## 为什么只保留一张膜

早期版本里有两张膜：

```text
/World/Bench/Membrane/mesh
/World/Bench/MembraneTextureVisual/mesh
```

其中第一张是 UIPC 物理膜，第二张只是贴 GelSight 纹理的可视化皮肤。第二张不参与接触、不参与形变、不参与 fxyz 计算，因此对当前三维力目标没有帮助，还容易误导观察结果。

当前 V1 已删除可视化贴图膜。场景里只保留：

```text
/World/Bench/Membrane/mesh
```

这张膜同时承担：

```text
1. UIPC 软体形变来源
2. fxyz 力场反推来源
```

纹理触觉细节不依赖 RGB 贴图，而来自真实接触几何，例如 `dots / cross_lines / wave1 / random` 压头表面的高度起伏。

## UIPC 在本方案里的角色

UIPC 是软体/有限元物理仿真模块。它负责让膜在被压、被摩擦时产生真实几何形变。

当前 Python 层稳定可读的是膜表面顶点位置，例如：

```text
surf_nodal_pos_w
```

没有稳定使用“每个接触点真实力”的接口。因此 V1 不假装直接读 UIPC 接触力，而是从膜形变反推三维力。

```text
UIPC = 形变引擎
fxyz = 从形变反推出的三维力场
```

力单位在 V1 中标记为：

```text
sim_constitutive_force
```

它是仿真本构力，不是已经标定到真实传感器的牛顿值。

## 场景组成

脚本：

```text
experiments/tactile-bench/OpenWorldTactile_v1.py
```

核心物体：

```text
/World/Bench/Membrane        UIPC 软膜
/World/Bench/Tool            运动压头
/World/Bench/MembraneAnchor  背面锚定块
```

默认膜尺寸参考 GelSight Mini：

```text
width     = 20.75 mm   对应局部 Y
length    = 25.25 mm   对应局部 Z
thickness = 4.5 mm     对应局部 X
```

膜前表面在局部 `X = 0` 附近，压头沿 `-X` 方向压入。

背面锚定块只用于固定膜背面顶点，防止整片膜刚体漂移。它不是触觉输出来源。

## 压头类型

支持两类压头：

```text
光滑几何:
  sphere
  cylinder

纹理几何:
  dots
  cross_lines
  wave1
  random
```

纹理几何是压头表面的真实高度场，不是图像贴图。它们会通过接触让 UIPC 膜产生对应形变，因此有机会在 `fz/fx/fy` 中留下空间纹理。

## 运动轨迹

V1 使用固定传感器、移动压头的四阶段轨迹：

```text
1. approach
   压头从初始间隙靠近膜

2. indent
   压头沿 -X 方向压入指定深度

3. rub
   在固定压入深度下沿 Y 方向横向摩擦

4. release
   压头退回初始间隙
```

主要 CLI 参数：

```text
--indent_depth_mm  法向压入深度
--rub_distance_mm  横向摩擦距离
--warmup_steps     无接触预热步数，预热后重新记录 rest surface
--approach_steps   靠近步数
--indent_steps     压入步数
--rub_steps        摩擦步数
--release_steps    释放步数
--render_viewport  每步刷新 Isaac viewport，便于观察压头运动
--render_every     开启 viewport 时每 N 步渲染一次，降低卡顿
--render_sleep_sec viewport 刷新后的可选延时，便于看慢动作
--no_save          只运行/观察，不写 fxyz、metadata、preview 文件
--loop_forever     无限循环压入轨迹，直到关闭窗口或 Ctrl+C
```

## 力场反推流程

每一帧读取 UIPC 膜表面顶点：

```text
current_surface = membrane.data.surf_nodal_pos_w
```

初始化时先让压头保持无接触间隙，运行 `warmup_steps` 步让 UIPC 膜和 attachment 稳定。预热完成后再保存：

```text
rest_surface = membrane.data.surf_nodal_pos_w
```

这样可以减少第 0 帧就出现明显假压缩的问题。

每帧计算：

```text
front vertices       膜前表面顶点
back vertices        膜背面顶点，用于估计整体漂移
global_drift         背面平均漂移
corrected_front      去掉整体漂移后的前表面
compression          X 方向法向压缩量
shear_disp           Y/Z 方向切向位移
compression_velocity X 方向压缩速度
shear_velocity       Y/Z 方向切向速度
vertex_area          每个前表面顶点的局部面积
```

局部顶点面积用前表面顶点的近邻尺度估计，并归一化到整片膜面积：

```text
sum(vertex_area) = width * length
```

## 本构力模型

法向力：

```text
normal_pressure = kn * compression + cn * positive_compression_velocity
Fn = vertex_area * max(normal_pressure, 0)
```

切向力：

```text
Ft_raw = vertex_area * (kt * shear_disp + ct * shear_velocity)
```

摩擦锥裁剪：

```text
|Ft| <= mu * Fn
```

最终顶点力通道：

```text
vertex_force = [Ft_y, Ft_z, Fn_x]
```

对应输出：

```text
fxyz[..., 0] = fx_local_y
fxyz[..., 1] = fy_local_z
fxyz[..., 2] = fz_local_x_normal_pressure
```

## 像素投影

前表面顶点力通过 Gaussian splat 投影到 `H x W` 像素网格。

每个顶点的 splat 权重会归一化：

```text
sum(weights_per_vertex) = 1
```

因此理论上满足：

```text
sum(pixel_fxyz) ~= sum(vertex_force)
```

metadata 中记录：

```text
conservation_error
max_conservation_error
```

目标误差：

```text
< 1%
```

低分辨率烟测中误差通常可以到 `1e-7` 量级。

## 输出文件

默认输出目录：

```text
/tmp/openworldtactile_newbench_validation
```

输出内容：

```text
fxyz.npy
metadata.json
preview_force.png
preview_sequence.mp4
preview_frames/
```

`metadata.json` 记录：

```text
force_units
force_definition
sdf_used_for_force
shape
fxyz_shape
channel_order
membrane parameters
uipc parameters
force_model parameters
trajectory parameters
per-step stats
```

V1 明确标记：

```text
sdf_used_for_force = false
texture_visual_skin_enabled = false
```

## 推荐运行命令

当前机器上已验证可用的环境：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v1.py \
  --shape dots \
  --indent_depth_mm 0.8 \
  --rub_distance_mm 3.0 \
  --output_dir /tmp/openworldtactile_newbench_validation
```

只看 Isaac viewport、不保存文件时：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v1.py \
  --shape sphere \
  --indent_depth_mm 0.8 \
  --rub_distance_mm 3.0 \
  --approach_steps 30 \
  --indent_steps 60 \
  --rub_steps 80 \
  --release_steps 30 \
  --render_viewport \
  --render_sleep_sec 0.02 \
  --no_save
```

低负载球压膜无限循环时：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v1.py \
  --shape sphere \
  --tool_radius_mm 3.0 \
  --indent_depth_mm 0.6 \
  --rub_distance_mm 0.0 \
  --front_segments_y 24 \
  --front_segments_z 30 \
  --thickness_segments 3 \
  --tet_edge_length_r 0.05 \
  --warmup_steps 5 \
  --approach_steps 20 \
  --indent_steps 40 \
  --rub_steps 0 \
  --release_steps 20 \
  --render_viewport \
  --render_every 5 \
  --render_sleep_sec 0.0 \
  --loop_forever \
  --no_save
```

`--loop_forever` 会强制关闭保存，避免无限运行时持续占用内存和磁盘。

如果 `isaaclab` 命令已正确配置，也可以使用：

```bash
isaaclab -p experiments/tactile-bench/OpenWorldTactile_v1.py \
  --shape dots \
  --indent_depth_mm 0.8 \
  --rub_distance_mm 3.0 \
  --output_dir /tmp/openworldtactile_newbench_validation
```

## V1 测试计划

空载测试：

```text
无接触运行，fxyz 应接近全零，无 NaN。
```

法向压入测试：

```text
压入 0.2 / 0.5 / 1.0 mm
sum(fz) 应随压入深度单调增加。
```

横向摩擦测试：

```text
同一压入深度下沿 +Y / -Y rub
sum(fx) 符号应反转。
```

纹理压入测试：

```text
dots / cross_lines / wave1 / random
fz 图应出现对应接触几何纹理。
```

空洞检查：

```text
力图不应出现由前表面顶点过稀导致的大面积空洞或棋盘断裂。
必要时提高 front_segments_y/front_segments_z 或调大 splat_sigma_px。
```

守恒检查：

```text
sum(pixel_fxyz) 与 sum(vertex_force) 的误差目标 < 1%。
```

## 后续阶段

阶段 1：单膜固定验证

```text
确认 UIPC 膜可压、可摩擦。
确认 fxyz 无 NaN、守恒、趋势正确。
确认纹理压头可以在力图中留下结构。
```

阶段 2：参数调优

```text
调 kn/cn/kt/ct/mu。
调膜网格密度和 splat 半径。
调压头纹理尺度与高度。
目标是得到稳定、连续、可解释的 fxyz。
```

阶段 3：批量验证脚本

```text
批量跑不同 shape、depth、rub direction。
自动生成 metadata 汇总。
检查法向单调性、摩擦符号翻转、守恒误差。
```

阶段 4：接回完整采集链路

```text
把固定 bench 中验证过的 UIPC 膜和 fxyz 推断模块接回插孔任务。
保持输出接口为 T x 300 x 300 x 3。
再决定是否写 HDF5，以及如何和原采集格式对齐。
```

阶段 5：真实标定

```text
用真实载荷或标定数据把 sim_constitutive_force 映射到真实单位。
在此之前，不把 V1 输出宣称为已标定牛顿值。
```
