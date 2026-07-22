# OpenWorldTactileBench: UIPC 三维力触觉迭代路线

## 总目标

本文件是 `experiments/tactile-bench` 的主 README，用来统一管理 OpenWorldTactile UIPC 三维力触觉方案的版本路线。

最终目标不是生成 GelSight 风格 RGB，而是得到可用于插孔采集链路的三维力场：

```text
UIPC 软膜真实形变
-> deformation-based 三维力 fxyz
-> 300 x 300 x 3
-> 接入 Piper 插孔任务
-> 写入 HDF5
-> 后续真实力标定
```

最终触觉数据约定：

```text
fxyz: T x 300 x 300 x 3

fx = OpenWorldTactile local Y 方向切向力
fy = OpenWorldTactile local Z 方向切向力
fz = OpenWorldTactile local X 方向法向压力
```

核心原则：

```text
UIPC 负责软膜形变。
fxyz 从膜表面形变反推。
SDF 不生成最终力。
RGB 贴图膜不参与力计算。
未标定前 force_units = sim_constitutive_force。
```

## 当前状态

当前已有 V1/V2/V2.2，V2.3 已形成阶段文档：

```text
OpenWorldTactile_v1.py
docs/OpenWorldTactile_v1_workflow.md
OpenWorldTactile_v2.py
api/openworldtactile_uipc_force.py
OpenWorldTactile_v2_force_test.py
docs/OpenWorldTactile_v2_force_module.md
OpenWorldTactile_v2_2.py
api/openworldtactile_camera_membrane.py
OpenWorldTactile_v2_2_camera_membrane_test.py
docs/OpenWorldTactile_v2_2_camera_observed_membrane.md
docs/OpenWorldTactile_v2_3_rgb_marker_tracking.md
```

V1 已完成：

```text
固定单膜压入/摩擦验证
只保留一张 UIPC 物理膜
去掉可视化 RGB 贴图膜
支持 sphere / cylinder / dots / cross_lines / wave1 / random 压头
从 surf_nodal_pos_w 反推 fxyz
支持 warmup 后重新记录 rest surface
支持 --render_viewport 实时观察
支持 --render_every 降低 viewport 渲染负载
支持 --loop_forever 无限循环观察
支持 --no_save 不写输出文件
输出 fxyz.npy / metadata.json / preview
```

V1 不包含：

```text
Piper 插孔任务
HDF5 写入
真实牛顿标定
批量验证套件
```

V2 已完成：

```text
力估计核心移动到 api/openworldtactile_uipc_force.py
OpenWorldTactile_v2.py 复用通用 API
新增不依赖 Isaac 的合成测试
zero / indent / shear_y / shear_z / drift / conservation case 已通过
```

V2.2 已完成：

```text
新增 api/openworldtactile_camera_membrane.py
新增 OpenWorldTactile_v2_2.py
内部相机从膜背后 / 传感器内部观察软膜
新增 camera-observable surface：不参与 UIPC/碰撞，只跟随 UIPC 前表面形变供内部相机观测
默认在 camera-observable surface 上启用 marker dots，用于 RGB/motion 可视观测
默认 fxyz 来自 camera-observed depth / motion
默认不保存输出，默认循环播放
保留 surface force estimator 作为参考统计
新增不依赖 Isaac 的 camera membrane 合成测试
zero / depth_indent / motion_shear_y / motion_shear_z / invalid_depth case 已通过
```

V2.3 规划中：

```text
新增 RGB marker tracking 切向形变估计
从 observed_rgb 中检测 marker dots
追踪 marker displacement
生成 dense shear_map
用 RGB marker tracking 生成 fx/fy
继续用 camera depth 生成 fz
输出 marker_tracking_overlay / shear_map / fxyz 对照 artifacts
```

当前推荐运行环境：

```bash
"${PYTHON_BIN:-python}"
```

V1 推荐运行命令：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v1.py \
  --shape dots \
  --indent_depth_mm 0.8 \
  --rub_distance_mm 3.0 \
  --output_dir /tmp/openworldtactile_newbench_validation
```

V1 只观察、不保存文件的命令：

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

V1 低负载球压膜无限循环命令：

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

V2 API 合成测试命令：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v2_force_test.py
```

V2.2 camera membrane API 合成测试命令：

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v2_2_camera_membrane_test.py
```

V2.2 固定膜相机观测 demo 命令，默认循环且不保存：

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

V2.2 需要保存输出时使用有限运行：

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

V2 轻量回归命令：

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

## 版本路线总览

```text
V1 固定单膜压入验证
-> V2 力估计核心模块化
-> V2.1 OpenWorldTactile dense mesh deformation reference
-> V2.2 OpenWorldTactile camera-observed membrane
-> V2.3 RGB marker tracking shear
-> V3 自动化验证套件
-> V4 传感器挂载接触验证
-> V5 插孔任务 debug 接入
-> V6 HDF5 数据采集链路
-> V7 数据质量迭代
-> V8 真实力标定
```

每个版本必须满足本版本验收标准后，才能进入下一版本。

## V1: 固定单膜压入验证

文档：

```text
docs/OpenWorldTactile_v1_workflow.md
```

代码：

```text
OpenWorldTactile_v1.py
```

目标：

```text
证明一张 UIPC 物理膜可以被压、被摩擦，并输出 deformation-based fxyz。
```

核心链路：

```text
UIPC membrane surf_nodal_pos_w
-> rest/current/previous surface
-> compression / shear / velocity / vertex_area
-> constitutive vertex force
-> conservative splat
-> fxyz
```

V1 验收标准：

```text
只有一张 UIPC 物理膜。
无 RGB 贴图膜。
无 SDF 生成力。
空载 fxyz 接近零。
压入越深 sum(fz) 越大。
+Y/-Y rub 时 sum(fx) 符号反转。
max_conservation_error < 1%。
```

## V2: 力估计核心模块化

阶段文档：

```text
docs/OpenWorldTactile_v2_force_module.md
```

目标：

```text
把 V1 中的 MembraneForceEstimator、局部面积估计、本构力计算、
摩擦裁剪、conservative splat 从 demo 脚本中抽离，形成可复用核心模块。
```

任务：

```text
1. 新建 api/openworldtactile_uipc_force.py。
2. 将 V1 中的力估计逻辑迁移到该模块。
3. 新建 OpenWorldTactile_v2.py 调用该模块。
4. V1 保留为参考版本，V2 脚本负责建场景、移动压头、保存结果。
5. 新增不依赖 Isaac 的合成输入测试。
```

模块输入接口：

```text
rest_surface
current_surface
dt
width
length
tactile_height
tactile_width
kn / cn / kt / ct / mu
splat_sigma_px
splat_radius_sigmas
```

模块输出接口：

```text
fxyz: H x W x 3
disp_grid
stats
```

验收标准：

```text
V2 调用新模块后输出 shape 与 V1 约定一致。
合成压入输入能产生正 fz。
合成 Y 方向剪切能产生 fx。
合成 Z 方向剪切能产生 fy。
整体刚体漂移不产生明显假力。
pixel force 与 vertex force 守恒误差 < 1%。
无 NaN。
模块测试不需要启动 Isaac。
```

失败标准：

```text
V1 输出明显变化且无法解释。
通道顺序混乱。
splat 不守恒。
整体刚体漂移产生大假力。
模块必须启动 Isaac 才能测试。
```

进入 V2.1 / V2.2 条件：

```text
V2 模块接口稳定，OpenWorldTactile_v2.py 已完全复用该模块。
```

## V2.1: OpenWorldTactile Dense Membrane

阶段文档：

```text
docs/OpenWorldTactile_v2_1_dense_membrane.md
```

目标：

```text
借鉴 dense surface observation 思想，但统一使用 OpenWorldTactile 命名。
从 UIPC 前表面三角面片构建 dense deformation map，
再输出 compression / shear / contact_mask / fxyz。
```

V2.1 不做：

```text
GelSight RGB
Piper 插孔
HDF5
真实牛顿标定
```

V2.1 当前定位：

```text
V2.1 是 mesh-based dense deformation 参考路线，不是当前主线的必经 gate。
当前主线选择 V2.2 camera-observed membrane，并在 V2.3 做 RGB marker tracking。

V2.1 若继续实现，至少应证明：
实心压头接触区连续。
有孔压头真实孔洞区域保持低 fz。
dense deformation map 无明显无效区域。
fxyz 通道顺序与 V1/V2 一致。
```

## V2.2: OpenWorldTactile Camera-Observed Membrane

阶段文档：

```text
docs/OpenWorldTactile_v2_2_camera_observed_membrane.md
```

目标：

```text
相机放在 OpenWorldTactile 膜背后 / 传感器内部，
压头从外侧接触软膜，
由内部相机像素观测 camera-observable surface 的形变，
再重建 compression / shear / contact_mask / fxyz。
```

V2.2 当前实现说明：

```text
UIPC 物理膜仍然是唯一参与接触和形变求解的软膜。
camera-observable surface 是一张渲染层，不参与 UIPC、碰撞或直接力计算。
它每帧从 UIPC 前表面最近邻顶点复制位移，让内部相机能稳定看到 dense depth/motion。
marker dots 是 camera-observable surface 前方的一层视觉纹理，同样只参与渲染观测。
```

V2.2 不做：

```text
直接把 RGB 当成最终触觉力
真实牛顿标定
Piper 插孔
HDF5
```

进入 V2.3 条件：

```text
内部相机能稳定看到膜可观测层。
压头不遮挡内部相机视线。
observed_depth / contact_mask 与膜形变一致。
fxyz 通道顺序与 V1/V2/V2.1 一致。
```

## V2.3: RGB Marker Tracking 切向形变

阶段文档：

```text
docs/OpenWorldTactile_v2_3_rgb_marker_tracking.md
```

目标：

```text
把 V2.2 中可见的 marker dots 变成可计算的切向形变来源。
从 observed_rgb 检测 marker，跟踪 marker displacement，
插值成 dense shear_map，再与 camera depth compression 一起生成 fxyz。
```

V2.3 主数据流：

```text
rest_rgb / current_rgb
-> marker detection
-> marker matching / tracking
-> sparse marker displacement
-> dense shear_y / shear_z

rest_depth / current_depth
-> compression

compression + shear
-> constitutive fxyz
```

V2.3 不做：

```text
真实牛顿标定
Piper 插孔
HDF5
RGB-only 法向恢复
```

进入 V3 条件：

```text
无接触 marker displacement 接近零。
+Y / -Y rub 下 RGB marker tracking 得到的 fx 符号反转。
法向压入 fz 与 V2.2 depth/surface-reference 趋势一致。
marker overlay、shear_map、fxyz preview 可复查。
metadata 明确记录 shear_source = rgb_marker_tracking。
```

## V3: 自动化验证套件

阶段文档：

```text
docs/OpenWorldTactile_v3_validation_suite.md
```

目标：

```text
把 V2.3 的 camera-observed depth + RGB marker tracking fxyz 链路变成自动检查，
不再只靠肉眼看 preview。
```

任务：

```text
1. 新建 run_openworldtactile_uipc_validation_suite.py。
2. 批量运行固定测试 case。
3. 默认调用 OpenWorldTactile_v2_3.py。
4. 每个 case 输出 fxyz、metadata、preview、marker overlay、shear_map。
5. 汇总 validation_summary.json。
6. 自动标记 pass/fail。
```

固定测试 case：

```text
empty_no_contact
indent_0p2mm
indent_0p5mm
indent_1p0mm
rub_positive_y
rub_negative_y
shape_sphere
shape_cylinder
shape_dots
shape_cross_lines
shape_wave1
shape_random
shape_texture_stamp
camera_depth_indent
marker_no_contact
marker_rub_positive_y
marker_rub_negative_y
marker_texture_stamp_rub
```

核心检查：

```text
空载 fxyz 接近零。
0.2 / 0.5 / 1.0 mm 下 sum(fz) 单调增加。
+Y / -Y rub 下 sum(fx) 符号反转。
无接触 marker displacement 接近零。
RGB marker tracking 的 +Y / -Y rub 位移方向反转。
camera depth fz 与 surface-reference fz 趋势一致。
所有 case 无 NaN。
纹理 case 的 fz preview 非空且无大片空洞。
texture_stamp 能显示物体纹理导致的局部形变结构。
```

输出：

```text
validation_summary.json
validation_summary.md
case_outputs/<case_name>/fxyz.npy
case_outputs/<case_name>/metadata.json
case_outputs/<case_name>/preview_force.png
case_outputs/<case_name>/marker_tracking_overlay.mp4
case_outputs/<case_name>/shear_map.npy
```

验收标准：

```text
所有 smoke case 完成。
所有 case 无 NaN。
空载接近零。
fz 随压入深度单调增加。
+Y / -Y rub 下 sum(fx) 符号反转。
marker detection 数量稳定。
无接触 marker displacement 接近零。
camera depth fz 与 surface-reference fz 趋势一致。
纹理 case preview 非空且无大片空洞。
validation_summary.json overall_status = pass。
```

失败标准：

```text
任一核心物理趋势失败。
任一 case 出现 NaN。
纹理 case 完全看不到结构。
marker tracking 大量丢失或误匹配。
任一 case 运行崩溃且无法自动记录 blocked 原因。
```

进入 V4 条件：

```text
V3 smoke suite 通过。
至少一个 full resolution case 通过。
marker overlay / shear_map artifacts 可复查。
validation_summary.json 可复现。
```

## V4: 传感器挂载接触验证

阶段文档：

```text
docs/OpenWorldTactile_v4_sensor_mounted_contact.md
```

目标：

```text
把固定世界坐标膜升级为挂载在 OpenWorldTactile/Piper link 上的相机观测式触觉传感器，
验证坐标转换、刚体运动补偿、内部相机观测和 marker tracking。
```

任务：

```text
1. 新建 OpenWorldTactile_v4_mounted_contact.py。
2. 复用 V2.3 camera marker fxyz 链路。
3. 将 UIPC 膜、observable surface、marker layer、internal camera 绑定到 OpenWorldTactile local frame。
4. 读取 surf_nodal_pos_w 后转换到 tactile local frame，作为 surface-reference 对照。
5. 验证传感器移动但无接触时 fxyz 和 marker displacement 仍接近零。
```

每帧数据流：

```text
sim.step()
uipc_sim.update_render_meshes()
surface_w = membrane.data.surf_nodal_pos_w
surface_local = world_to_tactile(surface_w)
surface_reference_fxyz = MembraneForceEstimator.compute(surface_local)
internal_camera.update()
camera_marker_fxyz = V2.3 marker/depth estimator
```

重点验证：

```text
world frame -> tactile local frame 转换正确。
膜随传感器移动时不产生假力。
内部相机随传感器移动时不产生假 marker displacement。
接触法向仍对应 fz。
切向 Y/Z 仍对应 fx/fy。
```

输出：

```text
mounted_contact_fxyz.npy
mounted_contact_metadata.json
mounted_contact_preview.mp4
mounted_contact_summary.json
mounted_marker_overlay.mp4
mounted_shear_map.npy
```

验收标准：

```text
无接触移动时 fxyz 接近零。
无接触旋转时 fxyz 接近零。
无接触移动时 marker displacement 接近零。
接触时 fz 非零。
离开接触后 fxyz 回落。
法向接触仍主要进入 fz。
Y/Z 切向运动仍进入 fx/fy。
坐标方向和 V1 一致。
```

失败标准：

```text
传感器刚体移动导致大 fxyz。
传感器刚体移动导致大 marker displacement。
坐标轴方向反了。
接触后 fxyz 不回零。
world/local 混用导致 rest_surface 与 current_surface 不在同一坐标系。
```

进入 V5 条件：

```text
挂载膜坐标转换通过。
刚体运动补偿通过。
V4 输出的 camera-marker fxyz 通道方向和 V1 一致。
```

## V5: 插孔任务 Debug 接入

阶段文档：

```text
docs/OpenWorldTactile_v5_insert_debug_integration.md
```

目标：

```text
参考 OpenWorldTactile_USD_insert_random_UIPC.py 接入完整 Piper 插孔流程，
但暂不保存 HDF5，只 debug fxyz 是否随插孔接触合理变化。
```

任务：

```text
1. 新建 OpenWorldTactile_insert_v1.py。
2. 复制/参考原插孔任务主链路。
3. 保留 Piper、cylinder、socket、camera、recovery 流程。
4. 替换触觉来源为 V2.3 camera depth + RGB marker tracking fxyz。
5. 禁用 SDF 作为最终力来源。
6. 每步把 fxyz 写入 tactile_force_field。
```

保留原插孔链路：

```text
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

每步集成流程：

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

输出：

```text
/tmp/openworldtactile_uipc_v5_insert_debug/
  episode_debug_summary.json
  tactile_preview.mp4
  tactile_stats.json
  keyframe_fxyz.npy
  marker_tracking_overlay.mp4
  shear_map_keyframes.npy
```

验收标准：

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

失败标准：

```text
插孔流程无法跑完。
UIPC 膜爆炸或漂移。
接触阶段 fxyz 始终为零。
非接触阶段持续大 fxyz。
fxyz 依赖 SDF 生成。
坐标转换导致方向错误。
marker tracking 大量丢失导致 fx/fy 不可信。
```

进入 V6 条件：

```text
至少 1 个插孔 episode debug 运行完整。
grasp/insert 阶段触觉响应可信。
非接触回零可信。
camera/marker artifacts 可复查。
debug artifacts 可复查。
```

## V6: HDF5 数据采集链路

阶段文档：

```text
docs/OpenWorldTactile_v6_hdf5_dataset_pipeline.md
```

目标：

```text
把 V5 的 fxyz 接入 HDF5 recorder，生成可用 episode 数据。
```

任务：

```text
1. 新建 OpenWorldTactile_insert_v2_hdf5.py。
2. 复用 V5 插孔流程。
3. 接入 HDF5EpisodeRecorder。
4. 保存 observations/tactile/fxyz。
5. 更新全部 tactile metadata。
```

HDF5 数据契约：

```text
observations/tactile/fxyz: T x 300 x 300 x 3
```

metadata 必须写：

```text
tactile_force_model = deformation_based_constitutive_uipc_camera_marker
tactile_contact_source = uipc_membrane_deformation_camera_observed_depth_rgb_marker
normal_source = camera_depth
shear_source = rgb_marker_tracking
tactile_sdf_generates_force = false
tactile_rgb_texture_used_for_force = true
tactile_rgb_used_for_normal_force = false
tactile_rgb_used_for_shear_force = true
tactile_visual_skin_enabled = false
tactile_uipc_conservative_splat_used = false
surface_reference_force_saved = optional
force_units = sim_constitutive_force
normalized = false
frame = tactile sensor local frame
```

建议同时保存或引用中间量：

```text
observations/tactile/compression_map
observations/tactile/shear_map
observations/tactile/shear_confidence
observations/tactile/marker_tracks
observations/tactile/marker_valid_mask
```

必须保留相机/触觉标定信息：

```text
T_world_tactile
T_camera_top_tactile
T_camera_wrist_tactile
T_link_tactile
T_tactile_link
corners_tactile
active_size_m
resolution
```

验收标准：

```text
能保存 1 个成功 episode。
HDF5 文件可打开。
observations/tactile/fxyz 存在。
fxyz shape = T x 300 x 300 x 3。
fxyz 无 NaN。
相机、关节、触觉帧数一致。
metadata 不再包含旧 SDF 生成力描述。
metadata 明确记录 normal_source = camera_depth。
metadata 明确记录 shear_source = rgb_marker_tracking。
force_units = sim_constitutive_force。
```

失败标准：

```text
HDF5 写入失败。
帧数不同步。
metadata 仍宣称 SDF 生成力。
metadata 仍宣称 RGB 完全不参与力估计。
fxyz 全程为零。
fxyz 出现 NaN。
触觉数据路径或 shape 不符合约定。
```

进入 V7 条件：

```text
至少 1 个 HDF5 episode 完整可用。
HDF5 checker 通过。
metadata 与新的 UIPC camera-marker deformation force 模型一致。
```

## V7: 数据质量迭代

阶段文档：

```text
docs/OpenWorldTactile_v7_dataset_quality_iteration.md
```

目标：

```text
让 HDF5 数据不只是能保存，而是稳定、连续、有接触结构，适合后续训练或分析。
```

任务：

```text
1. 批量采集多个 episode。
2. 统计 fxyz 分布。
3. 对比 grasp / preinsert / insert / recovery 阶段触觉差异。
4. 调整膜参数、splat 参数、接触速度。
5. 调整 marker 参数、camera 参数和 tracking 参数。
6. 形成 recommended_params.json。
7. 形成 quality_report.md。
```

建议最小批量：

```text
smooth_insert: 5 episodes
xy_offset: 5 episodes
edge_search: 5 episodes
partial_jam: 5 episodes
```

重点调参：

```text
normal_stiffness
normal_damping
shear_stiffness
shear_damping
friction_mu
front_segments_y
front_segments_z
tet_edge_length_r
splat_sigma_px
camera_width
camera_height
marker_spacing_mm
marker_radius_mm
marker_detection_threshold
marker_tracking_max_disp_px
shear_interpolation_radius_px
insert speed
grasp threshold
recovery motion speed
```

质量指标：

```text
fxyz_nan_count
fxyz_max_abs
fz_contact_mean
fz_free_mean
fz_contact_to_free_ratio
fx_direction_consistency
fy_direction_consistency
max_conservation_error
empty_area_ratio
spike_frame_count
episode_success_rate
uipc_failure_count
marker_detection_count_mean
marker_detection_count_std
marker_tracking_loss_rate
marker_displacement_free_mean
marker_displacement_contact_mean
shear_confidence_mean
camera_surface_fz_consistency
```

输出：

```text
quality_report.md
quality_summary.json
recommended_params.json
sample_previews/
case_comparison_plots/
```

验收标准：

```text
至少 20 个 episode 可写入并可打开。
所有 episode fxyz 无 NaN。
连续 episode 无 UIPC 崩溃。
接触阶段 fz 明显高于空载阶段。
fx/fy 在偏心或摩擦阶段有方向性。
marker tracking 在接触阶段稳定。
无接触 marker displacement 接近零。
force map 无大面积空洞。
force map 无明显随机尖峰。
recommended_params.json 固定。
quality_report.md 明确记录参数和结论。
```

失败标准：

```text
多 episode 后 UIPC 不稳定。
触觉响应不可重复。
force map 大面积空洞。
接触和非接触无法区分。
fx/fy 方向完全不稳定。
marker tracking 大量丢失。
shear_confidence 长期过低。
recommended_params 无法收敛。
```

进入 V8 条件：

```text
数据质量稳定。
推荐参数固定。
HDF5 数据可用于后续训练或分析。
仍明确 force_units = sim_constitutive_force。
```

## V8: 真实力标定

阶段文档：

```text
docs/OpenWorldTactile_v8_force_calibration.md
```

目标：

```text
把 sim_constitutive_force 映射到真实力单位。
```

标定输入：

```text
known_normal_load
known_shear_load
真实力计数据
真实 OpenWorldTactile/GelSight 对照数据
标准砝码或已知压入力实验
```

最低要求：

```text
至少 3 个不同法向载荷水平。
至少 2 个不同切向方向。
每个载荷水平重复多次。
记录真实力和对应 fxyz。
```

任务：

```text
1. 设计法向标定实验。
2. 设计切向标定实验。
3. 收集真实力和仿真 fxyz 对应数据。
4. 拟合 sim_constitutive_force 到 Newton 的映射。
5. 评估误差。
6. 输出 calibration_params.json。
7. 更新 HDF5 metadata 标定字段。
```

建议先做简单线性标定：

```text
F_newton = scale * F_sim + bias
```

如果线性误差过大，再考虑：

```text
分通道 scale
分段线性
深度相关 scale
区域相关校正
```

输出：

```text
calibration_params.json
calibration_report.md
calibrated_fxyz_samples/
calibration_error_plots/
```

metadata 更新：

```text
force_units = calibrated_newton
calibration_version = v1
calibration_source = known_load_or_force_gauge
calibration_params_file = calibration_params.json
```

未完成标定的数据必须继续写：

```text
force_units = sim_constitutive_force
```

验收标准：

```text
法向力误差在项目可接受范围内。
切向力方向正确。
不同压入深度尺度一致。
重复实验误差可接受。
calibration_params.json 可复用。
metadata 明确记录标定版本。
未标定数据和已标定数据能明确区分。
```

失败标准：

```text
sim force 与真实力无稳定关系。
切向力方向不可靠。
不同深度下尺度严重漂移。
重复实验误差过大。
metadata 无法区分标定/未标定数据。
```

## 文档索引

阶段文档集中放在 `docs/`：

```text
README.md
docs/OpenWorldTactile_v1_workflow.md
docs/OpenWorldTactile_v2_force_module.md
docs/OpenWorldTactile_v2_1_dense_membrane.md
docs/OpenWorldTactile_v2_2_camera_observed_membrane.md
docs/OpenWorldTactile_v2_3_rgb_marker_tracking.md
docs/OpenWorldTactile_v3_validation_suite.md
docs/OpenWorldTactile_v4_sensor_mounted_contact.md
docs/OpenWorldTactile_v5_insert_debug_integration.md
docs/OpenWorldTactile_v6_hdf5_dataset_pipeline.md
docs/OpenWorldTactile_v7_dataset_quality_iteration.md
docs/OpenWorldTactile_v8_force_calibration.md
docs/OpenWorldTactile_v1_v6_flowchart.png
docs/为什么不能直接使用GelSight膜.md
```

`README.md` 是总入口；`docs/` 中保留分阶段细节记录和流程图。
