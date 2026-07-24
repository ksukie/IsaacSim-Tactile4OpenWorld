# OpenWorldTactile_v4: 传感器挂载接触验证

## Goal

把固定世界坐标膜升级为挂载在 OpenWorldTactile/Piper link 上的相机观测式触觉传感器，
验证坐标转换、刚体运动补偿、内部相机观测和 marker tracking。

V4 要解决的问题：

```text
V1-V3 的膜和内部相机固定在世界坐标中。
最终插孔任务中，UIPC 膜、camera-observable surface、marker layer 和内部相机都会跟随 Piper/OpenWorldTactile 运动。
如果坐标转换错误，刚体运动会被误判为膜形变，导致 fxyz 假力。
```

V4 不解决的问题：

```text
不跑完整插孔。
不保存 HDF5。
不做数据集采集。
不做真实力标定。
```

## Current State

V3 之后应具备：

```text
openworldtactile_uipc_force.py
openworldtactile_camera_membrane.py
openworldtactile_marker_tracking.py
自动验证 suite
固定膜 camera/marker bench 已验证
```

V4 在这个基础上增加挂载坐标系。

## Key Changes

新增脚本：

```text
experiments/tactile-bench/OpenWorldTactile_v4_mounted_contact.py
```

脚本职责：

```text
1. 创建一个可移动 OpenWorldTactile mount frame。
2. 将 UIPC 膜、observable surface、marker layer、internal camera 绑定到该 frame。
3. 在无接触情况下移动/旋转传感器。
4. 将 UIPC surface world positions 转换到 tactile local frame，作为 surface-reference 对照。
5. 用内部相机输出 depth/RGB，调用 V2.3 camera marker 链路计算 fxyz。
6. 验证刚体运动不会产生假力，也不会产生假 marker displacement。
```

## Data Flow

每帧流程：

```text
sim.step()
uipc_sim.update_render_meshes()
surface_w = membrane.data.surf_nodal_pos_w
surface_local = world_to_tactile(surface_w)
surface_reference_fxyz = MembraneForceEstimator.compute(surface_local)
update observable surface / marker layer in tactile local frame
internal_camera.update()
camera_marker_fxyz = OpenWorldTactileCameraMarkerForceEstimator.compute(camera_output)
```

关键点：

```text
rest_surface 和 current_surface 必须处在同一个 tactile local frame。
不能把 world frame 直接传给 force estimator。
不能把传感器刚体移动当作膜压缩。
internal camera 必须跟随 tactile frame 运动。
无接触刚体运动不应产生 marker displacement。
```

## Motion Cases

V4 固定测试动作：

```text
1. no_contact_translate_y
   传感器沿 Y 平移，无接触。

2. no_contact_translate_z
   传感器沿 Z 平移，无接触。

3. no_contact_rotate_small
   传感器小角度旋转，无接触。

4. mounted_normal_indent
   移动压头压向挂载膜。

5. mounted_release
   压头离开，fxyz 回落。

6. no_contact_camera_motion
   传感器移动，内部相机看到的 rest marker 不应产生假 shear。

7. mounted_marker_rub
   接触并横向摩擦，RGB marker tracking 应产生正确方向的 fx/fy。
```

## Outputs

建议输出目录：

```text
/tmp/openworldtactile_uipc_v4_mounted_contact
```

输出文件：

```text
mounted_contact_fxyz.npy
mounted_contact_metadata.json
mounted_contact_preview.mp4
mounted_contact_summary.json
mounted_marker_overlay.mp4
mounted_shear_map.npy
```

metadata 必须记录：

```text
frame = tactile local frame
world_to_tactile_transform_used = true
rigid_motion_compensation = tactile_local_surface_positions
force_source = camera_depth_rgb_marker_tracking
normal_source = camera_depth
shear_source = rgb_marker_tracking
sdf_used_for_force = false
```

## Run Command

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v4_mounted_contact.py \
  --output_dir /tmp/openworldtactile_uipc_v4_mounted_contact
```

## Acceptance Criteria

V4 通过条件：

```text
无接触平移时 fxyz 接近零。
无接触旋转时 fxyz 接近零。
无接触移动时 marker displacement 接近零。
接触时 fz 非零。
离开接触后 fxyz 回落。
法向接触仍主要进入 fz。
Y/Z 切向运动仍进入 fx/fy。
mounted_marker_rub 中 fx/fy 方向与运动方向一致。
坐标方向和 V1 一致。
```

## Failure Criteria

V4 失败条件：

```text
传感器刚体移动导致大 fxyz。
传感器刚体移动导致大 marker displacement。
坐标轴方向反了。
接触后 fxyz 不回零。
world/local 混用导致 rest_surface 与 current_surface 不在同一坐标系。
internal camera 没有跟随 tactile frame。
```

## Next Version Gate

进入 V5 前必须满足：

```text
挂载膜坐标转换通过。
刚体运动补偿通过。
V4 输出的 camera-marker fxyz 通道方向和 V1 一致。
无接触移动不产生假 marker shear。
```
