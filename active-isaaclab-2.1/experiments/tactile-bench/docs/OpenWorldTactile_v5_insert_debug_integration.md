# OpenWorldTactile_v5: 插孔任务 Debug 接入

## Goal

参考 `OpenWorldTactile_USD_insert_random_UIPC.py` 接入完整 Piper 插孔流程，但暂不保存 HDF5，只 debug fxyz 是否随插孔接触合理变化。

V5 要解决的问题：

```text
V1-V4 只证明了单膜和挂载膜可以输出 fxyz。
最终目标需要在真实插孔动作中输出触觉。
V5 先接入插孔任务，但不引入 HDF5 保存复杂度。
```

V5 不解决的问题：

```text
不写 HDF5。
不批量采集数据集。
不做真实力标定。
不优化数据质量。
```

## Current State

V4 之后应具备：

```text
挂载膜坐标转换正确
刚体运动不产生假力
接触方向和 V1 一致
```

V5 从此处开始接入 Piper 插孔任务。

## Key Changes

新增脚本：

```text
experiments/tactile-bench/OpenWorldTactile_insert_v1.py
```

参考源文件：

```text
experiments/manipulation/pick-place/OpenWorldTactile_USD_insert_random_UIPC.py
```

保留原插孔链路：

```text
Piper
cylinder
socket
pick
grasp
preinsert
insert
recovery
scene_camera
wrist_camera
```

替换触觉来源：

```text
旧: SDF/hybrid tactile force
新: UIPC membrane deformation -> internal camera depth + RGB marker tracking -> tactile_force_field
```

## Integration Rules

必须遵守：

```text
不直接修改原始 OpenWorldTactile_USD_insert_random_UIPC.py。
新脚本放在 OpenWorldTactileBench。
不调用 SDF 生成最终 fxyz。
不创建旧 RGB 贴图膜。
RGB 只用于 marker tracking 切向形变，不作为最终 RGB 触觉图。
最终每帧 tactile_force_field 默认来自 V2.3 camera marker fxyz。
surface-reference fxyz 只作为 debug 对照。
```

每步流程：

```text
sim.step()
env.uipc_sim.update_render_meshes()
env._sync_runtime_openworldtactile_pose()
surface_w = env._uipc_gelpad.data.surf_nodal_pos_w
surface_local = world_to_tactile(surface_w)
surface_reference_fxyz = MembraneForceEstimator.compute(surface_local)
internal_camera.update()
fxyz = camera_depth_rgb_marker_estimator.compute(camera_output)
env.openworldtactile_left._data.output["tactile_force_field"] = fxyz
debug preview/log
```

## Debug Outputs

V5 不写 HDF5，但必须输出 debug artifacts：

```text
/tmp/openworldtactile_uipc_v5_insert_debug/
  episode_debug_summary.json
  tactile_preview.mp4
tactile_stats.json
keyframe_fxyz.npy
marker_tracking_overlay.mp4
shear_map_keyframes.npy
```

`tactile_stats.json` 至少记录：

```text
step
phase
sum_fx
sum_fy
sum_fz
max_fz
max_compression
marker_count
mean_marker_displacement
surface_reference_sum_fz
camera_sum_fz
```

## Run Command

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_insert_v1.py \
  --output_dir /tmp/openworldtactile_uipc_v5_insert_debug \
  --disable_demo_insert_assist
```

具体 CLI 可以继承原插孔脚本，但 V5 必须能在无 `--dataset_dir` 情况下运行。

## Acceptance Criteria

V5 通过条件：

```text
能跑完一个完整 episode。
grasp 接触时 fxyz 非零。
insert 接触时 fxyz 有响应。
非接触阶段 fxyz 回落。
fxyz 无 NaN。
marker tracking 在接触阶段可用。
rub 或偏心接触阶段 fx/fy 有方向性。
UIPC 膜不爆炸、不明显漂移。
不调用 SDF 生成最终 fxyz。
debug summary 能标出各阶段触觉统计。
```

## Failure Criteria

V5 失败条件：

```text
插孔流程无法跑完。
UIPC 膜爆炸或漂移。
接触阶段 fxyz 始终为零。
非接触阶段持续大 fxyz。
fxyz 依赖 SDF 生成。
坐标转换导致方向错误。
marker tracking 大量丢失导致 fx/fy 不可信。
```

## Next Version Gate

进入 V6 前必须满足：

```text
至少 1 个插孔 episode debug 运行完整。
grasp/insert 阶段触觉响应可信。
非接触回零可信。
camera/marker artifacts 可复查。
debug artifacts 可复查。
```
