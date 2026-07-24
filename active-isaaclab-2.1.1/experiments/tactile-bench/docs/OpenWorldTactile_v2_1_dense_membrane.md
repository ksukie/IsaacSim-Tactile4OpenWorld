# OpenWorldTactile_v2.5: OpenWorldTactile Dense Membrane 方案

## 目标

V2.1 的目标是在 V2 已完成 API 化之后，升级膜表面观测方式：

```text
当前 V2:
  UIPC 表面顶点 -> 顶点力估计 -> conservative splat -> fxyz

V2.1:
  UIPC 前表面三角面片 -> dense deformation map -> fxyz
```

V2.1 仍然不做 RGB，不做 GelSight 风格图像，不接 Piper，不写 HDF5。它只解决一个问题：

```text
如何更像真实触觉膜一样，密集观察整张 OpenWorldTactile 软膜表面形变。
```

## 命名原则

后续代码和文档统一使用 OpenWorldTactile 命名：

```text
OpenWorldTactile dense membrane
OpenWorldTactile dense deformation map
OpenWorldTactile marker-like shear sampling
OpenWorldTactile fxyz
```

GelSight 只作为设计参考，不作为模块名、脚本名或输出数据名。

## 为什么需要 V2.1

V2 的力估计核心已经可复用，但它仍然是：

```text
顶点采样
顶点力估计
高斯 splat 到像素
```

这能跑通流程，但对以下目标还不够自然：

```text
整片膜形变区域的连续表达
有孔物体压膜时保留真实孔洞
凸起 / 边缘 / 纹理几何的细节保真
更稳定的 contact_mask
后续切向位移场估计
```

V2.1 要把观测方式升级为：

```text
front surface triangles
-> rasterize to 300 x 300
-> interpolate rest/current surface at each pixel
-> dense compression / shear / mask
-> fxyz
```

## 建议新增文件

核心 API：

```text
experiments/tactile-bench/api/openworldtactile_dense_membrane.py
```

V2.1 demo：

```text
experiments/tactile-bench/OpenWorldTactile_v2_1.py
```

测试脚本：

```text
experiments/tactile-bench/OpenWorldTactile_v2_1_dense_membrane_test.py
```

## 核心数据流

```text
UIPC surf_nodal_pos_w
-> identify front surface vertices
-> recover / build front surface triangles
-> project rest y/z to tactile image plane
-> rasterize triangles to H x W
-> barycentric interpolate rest/current xyz
-> dense displacement map
-> dense compression/shear map
-> constitutive fxyz
```

输出：

```text
fxyz: H x W x 3
compression_map: H x W
shear_map: H x W x 2
disp_map: H x W x 3
contact_mask: H x W
valid_mask: H x W
stats: dict
```

## fxyz 通道定义

保持 V1/V2 不变：

```text
fxyz[..., 0] = fx_local_y
fxyz[..., 1] = fy_local_z
fxyz[..., 2] = fz_local_x_normal_pressure
```

力单位仍然是：

```text
sim_constitutive_force
```

## Dense Deformation Map

V2.1 不再只把顶点力 splat 到图上，而是先构建密集形变图：

```text
rest_xyz_grid:    H x W x 3
current_xyz_grid: H x W x 3
disp_grid = current_xyz_grid - rest_xyz_grid
```

法向压缩：

```text
compression = max(rest_x - current_x, 0)
```

切向位移：

```text
shear_y = current_y - rest_y
shear_z = current_z - rest_z
```

这样得到的是“整张 OpenWorldTactile 软膜前表面”的像素级形变，而不是仅顶点位置的形变。

## Contact Mask

V2.1 应输出至少两个 mask：

```text
valid_mask:
  该像素是否被前表面三角面片覆盖，是否有可靠插值结果。

contact_mask:
  该像素是否被认为存在接触。
```

初版 contact_mask 可用：

```text
compression > compression_threshold
```

后续可加入：

```text
fz threshold
surface normal consistency
local curvature / edge response
temporal stability
```

## OpenWorldTactile Marker-like Shear Sampling

V2.1 可以先做 dense deformation，不一定第一版就做 marker。

如果加入 marker-like shear，建议不要渲染 RGB marker，而是在计算层定义虚拟采样点：

```text
marker rest position on front surface
marker triangle id
marker barycentric coordinate
current marker position from interpolated triangle motion
marker displacement -> shear field
```

这个模块用于增强切向位移估计：

```text
fx / fy 趋势
滑动方向
局部剪切模式
```

但 marker-like shear 仍然不是已标定真实力。

## 膜参数建议

V2.1 需要比 V2 更重视膜质量。

轻量调试：

```text
front_segments_y = 32
front_segments_z = 40
thickness_segments = 3
tet_edge_length_r = 0.05
```

中等验证：

```text
front_segments_y = 64
front_segments_z = 80
thickness_segments = 4
tet_edge_length_r = 0.03
```

高质量离线：

```text
front_segments_y = 96~160
front_segments_z = 120~192
thickness_segments = 6+
tet_edge_length_r = 0.01~0.02
```

高质量版本不建议实时 viewport 观察，适合离线输出 `fxyz.npy` 和 deformation maps。

## 几何压头

V2.1 应保留：

```text
sphere
cylinder
dots
cross_lines
wave1
random
```

并新增用于结构保真验证的压头：

```text
ring
perforated_plate
hollow_grid
```

验证原则：

```text
实心球 / 圆柱:
  fz 应该连续，不应出现由采样造成的破碎空白。

ring / perforated_plate / hollow_grid:
  实体接触区域 fz 高，真实孔洞区域 fz 低。

dots / cross_lines:
  fz 应该能看到对应几何纹理。
```

## 验收标准

V2.1 通过条件：

```text
1. OpenWorldTactile_v2_1.py 能运行固定膜压入 demo。
2. api/openworldtactile_dense_membrane.py 不依赖 Isaac app 启动即可测试。
3. 输出 fxyz shape 为 H x W x 3。
4. 输出 compression_map / shear_map / contact_mask / valid_mask。
5. 实心 sphere/cylinder 接触区连续。
6. ring/perforated_plate 的真实孔洞区域保持低 fz。
7. drift_only 不产生明显假力。
8. 无 NaN。
9. fxyz 通道顺序与 V1/V2 一致。
```

## 失败标准

V2.1 失败条件：

```text
1. dense map 仍然依赖顶点 splat 作为主要形变来源。
2. 有孔压头的孔洞被错误填满为高 fz。
3. 实心压头接触区域破碎。
4. valid_mask 大面积缺失。
5. 刚体漂移产生明显假力。
6. 通道顺序与 V1/V2 不一致。
```

## 当前路线定位

V2.1 是 mesh-based dense deformation 参考路线，不是当前主线的必经 gate。
当前主线已经转向 V2.2 / V2.3：

```text
V2:
  API 化当前 force estimator

V2.1:
  OpenWorldTactile dense membrane / dense deformation map reference

V2.2:
  OpenWorldTactile camera-observed membrane

V2.3:
  RGB marker tracking shear

V3:
  自动化验证套件，对 V2.3 完整 camera depth + RGB marker tracking fxyz 链路做系统验证
```

如果未来继续实现 V2.1，至少应证明：

```text
实心物体连续接触
有孔物体保留真实孔洞
dense deformation map 无明显无效区域
fxyz 无 NaN 且通道正确
```
