# UIPC_Pad force contract

本文档冻结后续抓取版本的 fxyz 来源定义。后续实现必须遵守这个 contract，不能再混用 v5 中出现过的 contact penetration、PhysX force、native contact force、pressure gradient 等来源。

## 唯一数据源

```text
UIPC_Pad.usda
  simulation/membrane_sim_mesh
    -> UIPC solver
    -> membrane surface deformation
    -> constitutive force estimator
    -> pad-local fxyz
```

fxyz 不来自：

- `camera_surface`
- `visual_back_mesh`
- contact geometry penetration
- PhysX contact force
- native UIPC contact force
- pressure gradient

fxyz 只来自：

- `simulation/membrane_sim_mesh` 的表面形变

## 原始 UIPC 输出

每个仿真 timestep 读取：

```python
current_surface_world = membrane.data.surf_nodal_pos_w
```

初始化阶段保存未接触状态：

```python
rest_surface_world = membrane.data.surf_nodal_pos_w.clone()
```

这里的 surface vertices 对应：

```text
/World/envs/env_0/Robot/link8/UIPC_Pad/simulation/membrane_sim_mesh
```

## 坐标转换

pad 会跟随 `link8` 运动，不能直接比较 world 坐标。每一步必须把当前 world surface 转到 pad local frame：

```python
current_surface_local = world_to_local(current_surface_world, pad_pose)
rest_surface_local = world_to_local(rest_surface_world, rest_pad_pose)
```

如果 rest surface 已经在初始化时转换并缓存为 pad local，则后续直接复用缓存的 `rest_surface_local`。

## 计算面选择

使用 `simulation/membrane_sim_mesh` 的 pad-local `+X` 前表面，也就是 max-x face：

```python
front_indices = select_max_x_surface_indices(rest_surface_local)
back_indices = select_min_x_or_back_surface_indices(rest_surface_local)

rest_front_l = rest_surface_local[front_indices]
current_front_l = current_surface_local[front_indices]
```

红色膜对应计算膜；visual 和 camera surface 不能作为 fxyz 数据源。

## 去除整体漂移

继承 v2.7 的做法，用 back face 估计 attachment 或整体刚性漂移：

```python
global_drift = mean(current_back_l - rest_back_l)
current_front_corrected = current_front_l - global_drift
```

这一项只用于去除整体漂移，不能替代真实前表面形变。

## Fz 来源

法向压缩沿 pad local `+X`：

```python
compression = rest_front_l[:, 0] - current_front_corrected[:, 0]
compression = max(compression, 0)
```

弹簧-阻尼模型：

```python
local_fz = area * (
    normal_stiffness * compression
    + normal_damping * normal_velocity
)
```

`local_fz` 的语义是 pad local X normal force。

## Fx/Fy 来源

切向形变来自 pad local `Y/Z`：

```python
shear_disp = current_front_corrected[:, 1:3] - rest_front_l[:, 1:3]
```

弹簧-阻尼模型：

```python
shear_force = area * (
    shear_stiffness * shear_disp
    + shear_damping * shear_velocity
)
```

通道语义：

```text
local_fx = pad local Y tangential shear
local_fy = pad local Z tangential shear
local_fz = pad local X normal compression
```

切向力必须经过库仑摩擦限制：

```python
sqrt(local_fx**2 + local_fy**2) <= mu * local_fz
```

## 最终 fxyz 输出

保存：

```text
local_fxyz_vertices.npy
```

shape：

```text
(T, N, 3)
```

通道固定为：

```text
channel 0: local_fx = pad local Y tangential shear
channel 1: local_fy = pad local Z tangential shear
channel 2: local_fz = pad local X normal compression
```

注意这里的 `fx/fy/fz` 是触觉输出通道命名，不是 world XYZ force。

## Pressure field 来源

压力场不能直接从 camera、contact penetration 或 gradient 生成。正式链路为：

```text
local_fxyz_vertices.npy
  -> local_fz_vertices
  -> front_face_vertex_yz
  -> force-conserving Gaussian splat
  -> pressure_fz_grid_reconstructed_proxy.npy
```

即：

```text
simulation membrane deformation
  -> fxyz
  -> Fz vertex field
  -> pressure field
```

## metadata 必填字段

后续抓取版本输出 metadata 时必须固定写入：

```json
{
  "force_source": "uipc_membrane_surface_deformation_constitutive_fxyz",
  "fz_source": "uipc_front_surface_normal_deformation",
  "fx_fy_source": "uipc_front_surface_tangential_deformation",
  "native_uipc_contact_force_used": false,
  "contact_geometry_role": "diagnostic_only"
}
```

## 与旧版本区别

`v5_new_5c` 的问题是来源混合：

```text
Fz: contact geometry penetration
Fx/Fy: membrane deformation
```

最终抓取版本必须统一为：

```text
UIPC membrane_sim_mesh
  -> surface deformation
  -> constitutive force model
  -> Fx/Fy/Fz
```

最终定义：

```text
UIPC_Pad.usda 中 simulation/membrane_sim_mesh 是唯一触觉计算源。
UIPC solver 负责产生膜表面形变。
基于 v2.7 / v5_new_6 MembraneForceEstimator 的弹簧-阻尼-摩擦模型，
将法向压缩和切向形变转换为 pad-local fxyz，
再通过 Fz 重建连续压力场。
```
