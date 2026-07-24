# V5 new 7b membrane contact probe

## 目标

`OpenWorldTactile_v5_new_7b_membrane_contact_probe.py` 只验证下面这条链路：

```text
kinematic indenter
  -> UIPC contact
  -> simulation/membrane_sim_mesh
  -> non-penetrating deformable contact response
  -> deformation/gap/inversion diagnostics
```

本阶段不做：

- Piper 抓取运动
- attachment
- force estimator
- fxyz
- pressure reconstruction
- contact geometry penetration force proxy

`diagnostic_penetration_proxy_mm` 只用于判断 contact barrier 是否失败，不能作为触觉力来源。

## 启动

有界面查看：

```bash
cd "${OWT_ROOT}"

./run.sh -p experiments/tactile-bench/OpenWorldTactile_v5_new_7b_membrane_contact_probe.py \
  --render_viewport \
  --render_every 1 \
  --render_sleep_sec 0.01 \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7b_membrane_contact_probe
```

默认流程：

```text
static Piper link8
  -> adjusted UIPC_Pad.usda
  -> UIPC membrane object from simulation/membrane_sim_mesh
  -> kinematic cylinder starts outside front face
  -> cylinder slowly moves along pad local -X
  -> hold at final commanded overlap
```

## 输出

```text
membrane_rest_vertices_local.npy
membrane_current_vertices_local.npy
membrane_deformation_local.npy
membrane_rest_vertices.npy
membrane_current_vertices.npy
membrane_deformation.npy
membrane_surface_triangles.npy
commanded_overlap_mm.npy
contact_distance_min_mm.npy
diagnostic_penetration_proxy_mm.npy
diagnostic_min_gap_mm.npy
max_normal_compression_mm.npy
max_deformation_mm.npy
mean_deformation_mm.npy
tool_motion_depth_mm.npy
flipped_triangle_ratio.npy
triangle_normal_dot_min.npy
triangle_normal_dot_mean.npy
verdict.json
metadata.json
status.json
```

metadata 关键字段：

```json
{
  "uipc_solver_used": true,
  "uipc_contact_enabled": true,
  "deformation_source": "simulation/membrane_sim_mesh",
  "attachment_used": false,
  "force_source": "none",
  "pressure_source": "none",
  "contact_geometry_role": "diagnostic_only",
  "surface_triangle_source": "uipc_sim.sio.simplicial_surface(2)"
}
```

## 验收

跑完先看：

```bash
cat /tmp/openworldtactile_uipc_v5_new_7b_membrane_contact_probe/verdict.json
```

`verdict.json` 中：

```json
{
  "contact_barrier_passed": true
}
```

表示 7b 主线验证通过。这个 verdict 只判断 contact barrier，不判断 force/fxyz。

通过条件：

```text
commanded_overlap_mm >= 0.50
max_deformation_mm 随压入增加
max_normal_compression_mm >= 0.02
contact_distance_min_mm >= 0，或只在数值容差内略小于 0
diagnostic_penetration_proxy_mm <= 0.15
flipped_triangle_ratio ~= 0
triangle_normal_dot_min > 0
```

默认阈值可通过下面参数改：

```text
--accept_min_commanded_overlap_mm
--accept_min_normal_compression_mm
--accept_min_deformation_increase_mm
--accept_max_penetration_proxy_mm
--accept_max_flipped_triangle_ratio
--accept_min_triangle_normal_dot
```

失败判据：

```text
commanded_overlap_mm > 0
但 max_normal_compression_mm 接近 0
且 contact_distance_min_mm 明显为负
且 diagnostic_penetration_proxy_mm 持续增大
```

这表示圆柱正在穿过膜，而不是压迫膜产生接触形变。

如果 `flipped_triangle_ratio > 0` 或 `triangle_normal_dot_min < 0`，说明膜 surface 出现局部翻转，需要先解决接触/网格/材料稳定性，不能进入 fxyz。

`membrane_surface_triangles.npy` 默认来自 UIPC 实际 render surface topology，而不是 USD 原始 8 点盒子拓扑；这样 inversion 检查对应 `membrane.data.surf_nodal_pos_w` 的 surface vertices。

## 后续

7b 通过后再进入：

```text
v5_new_7c attachment validation
v5_new_7d deformation -> Fz
v5_new_7e Fx/Fy shear
v5_new_7f Piper grasp
```

force contract 仍固定为：

```text
simulation/membrane_sim_mesh deformation
  -> constitutive force estimator
  -> pad-local Fx/Fy/Fz
```
