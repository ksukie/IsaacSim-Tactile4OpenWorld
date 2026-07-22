# OpenWorldTactile_v2.2: OpenWorldTactile Camera-Observed Membrane 方案

## 目标

V2.2 的目标是把 OpenWorldTactile 触觉膜升级为“相机观测式软膜”：

```text
压头 / 物体从外侧接触软膜
相机从膜背后 / 传感器内部观察软膜表面形变
相机像素形成 dense tactile observation
再从 dense observation 推出 deformation / contact / fxyz
```

当前 V2.2 实现里需要区分两层：

```text
UIPC 物理膜
  参与接触、有限元形变、摩擦和 surface-reference force 估计。

camera-observable surface
  只参与渲染给内部相机看。
  不参与 UIPC。
  不参与碰撞。
  不直接生成力。
  每帧从 UIPC 前表面复制最近邻位移。

marker dots
  贴在 camera-observable surface 的相机侧。
  只参与 RGB / motion 观测。
  不参与 UIPC。
  不参与碰撞。
  不直接生成力。
  用于让内部相机画面出现可追踪纹理。
```

这样做的原因是：Isaac 内部相机需要一个稳定、规则、可渲染的表面来产生 dense depth / motion；原始 UIPC tet/surface mesh 对相机来说不一定是理想的可观测层。

这一路线更接近真实成像式触觉传感器的观测方式，但仍然保持本项目目标：

```text
输出 fxyz 三维力场，而不是 RGB 触觉图
```

## 命名原则

后续命名统一使用 OpenWorldTactile：

```text
OpenWorldTactile camera-observed membrane
OpenWorldTactile internal tactile camera
OpenWorldTactile marker layer
OpenWorldTactile observed deformation map
OpenWorldTactile fxyz
```

GelSight 只作为结构参考，不作为模块名。

## 相机应该放在哪里

相机应放在膜背后或传感器内部，而不是压头那一侧。

推荐结构：

```text
外侧接触物体 / 压头
        ↓
OpenWorldTactile 软膜前表面
UIPC 物理膜
camera-observable surface / marker layer
marker dots
透明支撑或内部空间
内部相机 + 光源
```

原因：

```text
如果相机在压头同侧，压头一接触就会遮挡膜。
相机在内部时，外侧负责接触，内侧负责观测膜形变。
```

## 与 V2.1 的区别

V2.1：

```text
直接使用 UIPC 前表面三角面片
手动 rasterize / interpolate
得到 dense deformation map
```

V2.2：

```text
使用 Isaac / renderer / camera
由虚拟相机像素观测 camera-observable surface
得到 RGB / depth / normal / marker-like motion
再重建 dense deformation map
```

两者都不直接输出真实力。区别在于观测方式：

```text
V2.1 = mesh-based dense deformation
V2.2 = camera-observed dense deformation
```

## 已新增文件

核心 API：

```text
experiments/tactile-bench/api/openworldtactile_camera_membrane.py
```

V2.2 demo：

```text
experiments/tactile-bench/OpenWorldTactile_v2_2.py
```

测试脚本：

```text
experiments/tactile-bench/OpenWorldTactile_v2_2_camera_membrane_test.py
```

## 运行命令

API 合成测试：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v2_2_camera_membrane_test.py
```

固定膜相机观测 demo，默认循环且不保存：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v2_2.py \
  --shape texture_stamp \
  --indent_depth_mm 1.3 \
  --rub_distance_mm 2.0 \
  --texture_bump_height_mm 0.7 \
  --front_segments_y 48 \
  --front_segments_z 60 \
  --thickness_segments 4 \
  --tet_edge_length_r 0.04 \
  --warmup_steps 10 \
  --approach_steps 30 \
  --indent_steps 60 \
  --rub_steps 60 \
  --release_steps 50 \
  --force_source camera
```

需要保存输出时使用有限运行：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v2_2.py \
  --single_run \
  --save_output \
  --shape texture_stamp \
  --indent_depth_mm 1.3 \
  --rub_distance_mm 2.0 \
  --texture_bump_height_mm 0.7 \
  --front_segments_y 48 \
  --front_segments_z 60 \
  --thickness_segments 4 \
  --tet_edge_length_r 0.04 \
  --warmup_steps 10 \
  --approach_steps 30 \
  --indent_steps 60 \
  --rub_steps 60 \
  --release_steps 50 \
  --force_source camera \
  --save_observed_camera \
  --output_dir /tmp/openworldtactile_uipc_v2_2_validation
```

## 核心数据流

```text
UIPC soft membrane deformation
-> update render mesh
-> update camera-observable surface from UIPC front-surface displacement
-> update marker dots from UIPC front-surface displacement
-> internal tactile camera render
-> RGB / depth / normal / segmentation / marker observation
-> reconstruct observed deformation map
-> compression / shear / contact_mask
-> fxyz
```

注意：

```text
camera-observable surface 不是第二张物理膜。
它只是为了让内部相机稳定观察到 UIPC 软膜前表面形变。
marker dots 只是相机可见纹理，不是新的接触物体。
最终 fxyz 仍来自 depth / motion 反推的 deformation-based constitutive force。
```

当前相机 RGB 预期画面：

```text
绿色可观测膜背景
黑色 marker dots
压入区域随 UIPC 膜形变产生局部运动
横向摩擦时 marker dots 在局部出现可见剪切位移
```

输出建议：

```text
fxyz: H x W x 3
observed_rgb: H x W x 3
observed_depth: H x W
observed_normal: H x W x 3
marker_flow: H x W x 2 or N x 2
compression_map: H x W
shear_map: H x W x 2
contact_mask: H x W
valid_mask: H x W
camera_metadata: dict
stats: dict
```

其中 `observed_rgb` 只是观测中间量，不作为最终触觉目标。

## fxyz 通道定义

保持 V1/V2/V2.1 不变：

```text
fxyz[..., 0] = fx_local_y
fxyz[..., 1] = fy_local_z
fxyz[..., 2] = fz_local_x_normal_pressure
```

力单位仍然是：

```text
sim_constitutive_force
```

## 相机可观测内容

V2.2 可分阶段实现。

第一阶段：深度 / 法线观测

```text
internal camera depth
internal camera normal
membrane segmentation mask
```

用于得到：

```text
observed surface geometry
normal direction
contact shape
compression map
```

第二阶段：marker-like 观测

```text
在膜可观测层放置虚拟 marker dots
相机观察 marker 运动
通过 marker displacement 估计切向位移
```

用于增强：

```text
shear_y
shear_z
fx
fy
slip tendency
```

第三阶段：光照 / RGB 观测

```text
多方向光照
反光或漫反射膜表面
RGB-to-normal / RGB-to-depth model
```

此阶段只在需要更像成像式触觉时加入，不影响最终目标仍为 `fxyz`。

## 关键限制

相机观测不会自动解决真实力问题。

相机直接看到的是：

```text
图像
深度
法线
marker 位移
接触轮廓
```

不是直接看到：

```text
fx / fy / fz
```

因此仍需要：

```text
材料模型
切向位移模型
摩擦模型
标定
```

另外，虚拟相机看到的膜表面仍然来自 UIPC / USD mesh 渲染：

```text
UIPC mesh deformation
-> renderer rasterization
-> camera pixels
```

所以它仍受膜 mesh 分辨率、渲染质量、相机分辨率和 marker 设计限制。

## 验证压头

V2.2 应至少验证：

```text
sphere
cylinder
texture_stamp
ring
perforated_plate
hollow_grid
dots
cross_lines
```

验证原则：

```text
实心球 / 圆柱:
  observed_depth / compression / fz 应形成连续接触区。

ring / perforated_plate:
  真实孔洞区域应保持低 compression / 低 fz。

dots / cross_lines:
  观测图和 fz 应出现对应几何纹理。

横向摩擦:
  marker_flow 或 shear_map 应出现方向性。
```

## 验收标准

V2.2 通过条件：

```text
1. OpenWorldTactile_v2_2.py 能运行固定膜压入 demo。
2. 内部相机能稳定看到 OpenWorldTactile 膜可观测层。
3. 压头不遮挡内部相机视线。
4. 能输出 observed_depth / valid_mask / contact_mask。
5. 能从相机观测重建 compression_map。
6. fxyz shape 为 H x W x 3。
7. 实心压头接触区连续。
8. 有孔压头真实孔洞区域保持低 fz。
9. 横向摩擦时 marker_flow 或 shear_map 有方向性。
10. 无 NaN。
```

## 失败标准

V2.2 失败条件：

```text
1. 相机放在压头同侧导致接触区被遮挡。
2. 相机看不到膜可观测层。
3. observed_depth 与膜形变无关。
4. contact_mask 大面积误判。
5. 有孔压头孔洞区域被错误填满。
6. fxyz 通道顺序与 V1/V2/V2.1 不一致。
7. 把 RGB 图像误当成最终触觉力数据。
```

## 与 V2.3 / V3 的关系

当前路线已经选择 V2.2 作为主线，V2.1 保留为 mesh dense deformation 参考路线：

```text
V2.1:
  OpenWorldTactile dense mesh deformation reference

V2.2:
  OpenWorldTactile camera-observed membrane

V2.3:
  RGB marker tracking shear

V3:
  自动化验证套件，对 V2.3 完整 camera depth + RGB marker tracking fxyz 链路做系统验证
```

进入 V2.3 前，V2.2 必须满足：

```text
内部相机能稳定看到 camera-observable surface。
observed_depth 与 UIPC 膜形变一致。
marker dots 在 RGB 中清晰可见。
texture_stamp 能产生可见局部形变。
```
