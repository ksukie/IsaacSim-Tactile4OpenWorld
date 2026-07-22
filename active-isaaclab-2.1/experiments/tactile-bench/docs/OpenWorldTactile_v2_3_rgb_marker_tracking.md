# OpenWorldTactile_v2.3: Surface Force + Visual-Only Texture Camera

## Current Decision

V2.3 的主线已经从旧的 RGB marker tracking / dense optical flow 路线切换为：

```text
UIPC membrane deformation
-> MembraneForceEstimator
-> fxyz grid
```

相机、RGB、纹理、marker、optical flow 都不参与 `fxyz` 计算。

下一步允许把“纹理”和“内部相机画面”加回来，但它们只用于观察和保存视频：

```text
UIPC membrane deformation
        |
        |-> MembraneForceEstimator -> fxyz
        |
        |-> visual texture skin -> internal camera RGB
```

也就是说：

```text
fxyz = surface deformation based force
camera RGB = visual/debug only
```

## What V2.3 Is

V2.3 是一个固定 bench，用来验证：

1. UIPC 软膜被压头按压后能产生稳定表面形变。
2. 形变能通过显式 spring-damper 模型转成 `fxyz`。
3. `fxyz` 可以保存成矩阵和热力图视频。
4. 可选地，内部相机能看到膜表面的纹理随形变变化。

V2.3 不是 GelSight 图像反演力，也不是 RGB/marker 追踪力。

## Non Goals

当前 V2.3 明确不做：

```text
RGB -> optical flow -> shear
marker detection -> marker matching -> shear
camera depth -> compression
UIPC internal contact force tensor -> fxyz
photometric stereo -> surface normal
real GelSight optical rendering calibration
```

这些能力可以以后作为单独实验，但不能混进当前 `fxyz` 主链路。

## Force Data Flow

每次运行时，先得到无接触参考膜面：

```text
rest_surface = membrane.data.surf_nodal_pos_w
```

每个仿真 step 读取当前膜面：

```text
current_surface = membrane.data.surf_nodal_pos_w
```

然后调用：

```text
surface_fxyz, surface_disp_grid, surface_stats =
    MembraneForceEstimator.compute(current_surface)
```

最终：

```text
selected_fxyz = surface_fxyz
```

## Force Formula

`MembraneForceEstimator` 不读取 UIPC 内部接触力。它只读取膜表面顶点位置，并用显式模型估计力。

先用背面点消除整体漂移：

```text
global_drift = mean(current_back - rest_back)
corrected_front = current_front - global_drift
```

法向压缩：

```text
compression = max(rest_front.x - corrected_front.x, 0)
```

切向位移：

```text
shear_disp = corrected_front.yz - rest_front.yz
```

法向力：

```text
normal_pressure = kn * compression + cn * positive_compression_velocity
fz_vertex = area_vertex * max(normal_pressure, 0)
```

切向力：

```text
shear_force = area_vertex * (ks * shear_disp + cs * shear_velocity)
```

摩擦限制：

```text
|shear_force| <= mu * fz_vertex
```

顶点力通道：

```text
vertex_force = [
    shear_force_y,
    shear_force_z,
    normal_force_x,
]
```

最后把前表面顶点力按静止膜面 `(y, z)` 位置 splat 到 tactile grid：

```text
fxyz: H x W x 3
```

Gaussian splat 只负责把稀疏顶点量映射成图像矩阵，不是用图像估计力。

## Channel Definition

默认输出大小为 `300 x 300 x 3`，但可以由 `--tactile_height` 和 `--tactile_width` 修改。

```text
fxyz[..., 0] = fx_local_y
fxyz[..., 1] = fy_local_z
fxyz[..., 2] = fz_local_x_normal_pressure
```

含义：

```text
fx = local Y 方向切向力
fy = local Z 方向切向力
fz = local X 法向压力，正值表示压入
```

注意：如果要表达“压头受到的反作用力”，方向可能需要取反。

## Units

当前单位是：

```text
sim_constitutive_force
```

它不是实验标定后的牛顿值。

要变成可信的 N，需要用真实实验标定：

```text
kn, ks, cn, cs, mu
```

以及膜材料和压头接触参数。

## Current Code State

当前 `OpenWorldTactile_v2_3.py` 已经是 surface-force-only：

```text
enable_cameras = False
force_source = surface_force
selected_fxyz = surface_fxyz
```

当前保留的主要输出：

```text
fxyz.npy
compression_map.npy
shear_map.npy
shear_confidence.npy
preview_sequence.mp4
fxyz_channels.mp4
preview_force.png
preview_fxyz_channels.png
metadata.json
```

当前不再输出：

```text
marker_tracks.npy
marker_tracks.json
observed_rgb.npy
observed_depth.npy
marker_tracking_overlay.mp4
observed_rgb_sequence.mp4
```

## Visual-Only Texture Camera Plan

下一步可以加回相机和纹理，但只用于显示。

目标：

```text
显示膜表面纹理随形变变化
保存内部相机 RGB 视频
不改变 fxyz 计算
不调用 optical flow
不调用 marker tracking
```

推荐新增视觉模块：

```text
visual texture skin
internal visual camera
visual_rgb_sequence.mp4
preview_visual_rgb.png
```

视觉纹理层要求：

1. 不参与 UIPC 接触。
2. 不参与 PhysX 碰撞。
3. 不参与 `MembraneForceEstimator`。
4. 每帧跟随 UIPC front surface 形变更新。
5. 可以使用随机 speckles、条纹、网格、彩色纹理或 marker-like dots。
6. 只作为相机可见材质或 mesh skin。

建议实现数据流：

```text
rest_front_surface
current_front_surface
-> update visual texture skin vertices
-> render internal camera RGB
-> save/display visual_rgb
```

视觉相机得到的 RGB 不进入：

```text
compression
shear_disp
fxyz
metadata force_source
```

## Metadata Requirements

`metadata.json` 必须清楚区分 force 和 visual：

```text
force_source = surface_force
force_definition = uipc_membrane_surface_deformation_to_constitutive_fxyz
force_api_module = scripts.demos.OpenWorldTactileBench.api.openworldtactile_uipc_force
normal_source = uipc_front_surface_normal_deformation_x
shear_source = uipc_front_surface_tangential_deformation_yz
visual_camera_enabled = true/false
visual_texture_enabled = true/false
visual_rgb_used_for_force = false
marker_tracking_used_for_force = false
optical_flow_used_for_force = false
force_units = sim_constitutive_force
```

如果未来保存 RGB 视频，应写入：

```text
output_files.visual_rgb_sequence
output_files.preview_visual_rgb
```

## Run Command: Current Surface Force

当前可运行的 surface-force-only 命令：

```bash
cd "${OWT_ROOT}"
conda activate isaaclab211

WORK=/tmp/openworldtactile_uipc_v23_$(date +%s)

python experiments/tactile-bench/OpenWorldTactile_v2_3.py \
  --headless \
  --single_run \
  --cycles 1 \
  --save_output \
  --output_dir /tmp/openworldtactile_v23_surface_force \
  --workspace_dir "$WORK" \
  --shape edged_box \
  --edged_box_width_mm 4 \
  --edged_box_length_mm 4 \
  --indent_depth_mm 0.5 \
  --rub_distance_mm 0.0 \
  --warmup_steps 0 \
  --sim_hz 120 \
  --attachment_strength_ratio 500 \
  --uipc_contact_d_hat_mm 0.2 \
  --uipc_contact_resistance_gpa 1.0 \
  --tool_m_kappa_mpa 20 \
  --preview_every 1 \
  --log_every 10
```

查看三通道力热力图：

```bash
mplayer /tmp/openworldtactile_v23_surface_force/fxyz_channels.mp4
```

## Future Run Command: With Visual Texture

视觉相机实现后，建议命令形态为：

```bash
python experiments/tactile-bench/OpenWorldTactile_v2_3.py \
  --single_run \
  --cycles 1 \
  --save_output \
  --output_dir /tmp/openworldtactile_v23_surface_force_visual \
  --shape edged_box \
  --edged_box_width_mm 4 \
  --edged_box_length_mm 4 \
  --indent_depth_mm 0.5 \
  --rub_distance_mm 0.0 \
  --enable_visual_camera \
  --visual_texture_mode random_stripes \
  --save_visual_rgb \
  --preview_every 1 \
  --log_every 10
```

这组 visual 参数是下一步实现目标，当前代码还没有全部提供。

## Validation Rules

V2.3 验证标准：

1. `fxyz.npy` 存在，shape 为 `T x H x W x 3`。
2. `fxyz` 无 NaN、无 Inf。
3. `metadata.force_source = surface_force`。
4. `sum(pixel_fxyz)` 与 `sum(vertex_force)` 的 conservation error 接近 0。
5. 无接触时 `fxyz` 接近 0。
6. 按压时 `fz` 在接触区域非零。
7. 横向摩擦时 `fx/fy` 出现方向性变化。
8. 加不加 visual camera，`fxyz` 结果应保持一致。
9. visual RGB 可以显示纹理随膜表面形变变化。
10. visual RGB 不应出现在 force source 中。

## Known Limitations

1. 当前 `fxyz` 是形变推力，不是 UIPC 内部接触力直接输出。
2. 当前 `fxyz` 未做真实牛顿标定。
3. Gaussian splat 会平滑局部力图，像素级细节不是严格局部物理测量。
4. visual texture 只用于观察，不代表真实 GelSight 光学模型。
5. 如果压头太大或压入太深，力图可能出现大面积接触，不一定是计算错误。
6. 如果 UIPC 在 warmup 或接触阶段出现 NaN，应优先调软 `attachment/contact/tool/dt` 参数。

## Summary

V2.3 的准确定位是：

```text
surface deformation truth -> scripted force estimator -> fxyz
visual texture camera -> human inspection only
```

不要再把 V2.3 描述成：

```text
RGB marker tracking force
dense optical flow force
camera depth force
GelSight image inversion
```

这个文档之后的代码修改都应遵守这条边界。
