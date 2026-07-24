# OpenWorldTactile UIPC V6.2 单侧软膜闭环抓取

V6.2 的正式结构是：PhysX 独占 Piper 与自由圆柱状态；link7 只保留普通 PhysX
刚性碰撞；link8 只安装一张计算膜：

```text
/World/envs/env_0/Robot/link8/UIPC_Pad/simulation/membrane_sim_mesh
```

默认机器人使用 `piper_openworldtactile.usda`。link7 上与 OpenWorldTactile 膜尺寸匹配的
`openworldtactile_case_left/openworldtactile_pad_visual` 在运行时启用为 PhysX 刚性 collider，与 link8
单张 UIPC 膜形成对称夹持；link7 仍没有第二张 UIPC 膜。
link8 膜后 0.5 mm 的资产刚性底座也启用为 PhysX convex-hull collider，作为膜的
物理背板，但正常抓取时不应接触圆柱。

UIPC 中还注册同一 PhysX 圆柱的运动学外部碰撞边界。它没有独立动力学状态，
不受 UIPC 重力推进。UIPC 求解域固定在 link8 Pad-local：膜和背面约束目标不随
机械臂做 world 平移；每帧只把 PhysX 圆柱转换成相对 Pad 的位姿：

```text
T_pad_object = inverse(T_world_pad) * T_world_object
```

UIPC 原生接触增量势梯度先除以 `uipc_dt²` 恢复为物理力，再在 Pad-local 边界
顶点上归约为 N 和 N·m、旋转回 world。原始反力先投影到单边 Coulomb 接触锥：
Pad-X 法向不得为负，切向模长不得超过 `mu * normal`。随后进行显式松弛并把力模
限制为 0.25 N、力矩模限制为 `0.25 * object_radius`。处理后的 wrench 在下一个
1/480 s 耦合子步施加给圆柱，同时将等大反向 wrench 施加给 link8。因此膜反力
既推动圆柱，也真正进入夹爪关节动力学；原始反力仍完整保存供诊断。

冻结 7g 的 TU 输出与这条动力学反力链完全分离，TU 不会施加给 PhysX。

## 每帧顺序

```text
1. 每个 60 Hz 记录间隔分为 8 个 1/480 s 耦合子步
2. 每个子步写入 Piper IK/夹爪目标，并将上一子步的松弛 UIPC 物体
   wrench 及等大反向 link8 wrench 写入 PhysX
3. PhysX 推进机器人和唯一自由圆柱
4. 读取 link8 Pad 与圆柱世界位姿，计算 `inverse(T_world_pad) * T_world_object`
5. 更新 Pad-local UIPC 外部圆柱边界及其上一/当前相对位姿历史
6. UIPC 在同一 1/480 s 步长下求解接触、摩擦和膜形变
7. 从 `ContactSystemFeature.contact_gradient` 读取原始反力/矩，除以 `uipc_dt²`
8. 将原始反力投影到单边 Coulomb 接触锥，再执行
   `feedback += alpha * (admissible_reaction - feedback)` 和模长限制
9. 8 个子步完成后保存原始、接触锥后、实际施加 wrench 序列及其时间平均，
   并计算冻结 7g TU
10. 仅为 viewport 显示把 Pad-local 膜面恢复到当前 Pad world 位姿
```

初始化时会同时用旧 world 表示和新 Pad-local 表示重建膜与圆柱。两者最大 world
坐标误差必须不超过 `1e-8 m`，因此坐标结构调整不会改变软膜、link8 和圆柱之间的
初始相对位置。这个检查失败时脚本会在第一次 UIPC 求解前终止。

正式运动循环没有任何圆柱 pose write，也没有 gap/形变驱动的夹爪停止、回退或
吸附逻辑。`minimum_signed_gap_mm.npy` 仅用于诊断和离线不可穿透验收。

## 夹爪物理驱动

`close` 阶段按固定时长把目标从打开插值到关闭。实际开度由有限隐式关节驱动与
接触反力共同决定：

```text
--gripper_closed_mm 17.5
--object_friction 2.0
--close_frames 90
--gripper_drive_stiffness 200
--gripper_drive_damping 8
--gripper_effort_limit_n 6
--gripper_closing_velocity_m_s 0.03
```

这些参数是普通关节驱动参数，不读取 gap、接触状态或 7g 输出。

## 运行

```bash
cd "${OWT_ROOT}"

"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v6_2_grasp.py \
  --render_viewport \
  --render_every 1 \
  --sim_hz 60 \
  --uipc_substeps_per_record 8 \
  --uipc_feedback_relaxation 1.0 \
  --uipc_feedback_force_limit_n 0.25 \
  --slow_frame_threshold_sec 0.5 \
  --loop_forever \
  --contract_dir /tmp/openworldtactile_uipc_v5_new_7f_contract_verified \
  --workspace_dir /tmp/openworldtactile_uipc_v6_2_workspace \
  --output_dir /tmp/openworldtactile_uipc_v6_2_run \
  --log_every 1
```

保留 `--uipc_substeps_per_record 8`（脚本拒绝小于 8 的设置，以免 60 Hz 运动边界
跨过 0.1 mm 接触层）。不添加 `--headless` 即可观察 viewport；在线不
启动 Camera sensor 或视频编码。

终端 stdout/stderr 的每一行（包括 Isaac Sim 原生日志）都带本地时区绝对时间戳。
初始 PhysX warmup、UIPC warmup 或正式仿真中任意一帧 wall time 超过 `0.5 s`，
脚本输出 `[V62_SLOW_FRAME]` 诊断：整帧耗时、UIPC 总耗时、最慢子步编号/耗时、
非 UIPC 开销、接触状态和 gap。慢帧本身不判定方案失败，仿真继续并保存子步诊断数据；
最终结论仍由穿入、物体抬升、释放回零等物理验收决定。

## 核心输出

```text
surface_displacement_pad_local.npy  # 冻结 7g 输入
force_pad_local.npy                 # TU，Pad-local
tactile_force_channels.npy         # [Fx,Fy,Fz]，TU
contact_active.npy                  # 原生 UIPC reaction 是否超过数值阈值

uipc_reaction_force_w.npy           # 8 个 UIPC 原始反力的时间平均，N
uipc_reaction_torque_w.npy          # 8 个 UIPC 原始反力矩的时间平均，N·m
applied_uipc_force_w.npy            # 8 个 PhysX 实际施加力的时间平均
applied_uipc_torque_w.npy
uipc_reaction_force_substeps_w.npy  # [T,8,3]，未松弛的原始反力
uipc_reaction_torque_substeps_w.npy
uipc_admissible_force_substeps_w.npy # [T,8,3]，接触锥投影后的反力
uipc_admissible_torque_substeps_w.npy
uipc_contact_cone_scale_substeps.npy
uipc_feedback_force_scale_substeps.npy
uipc_feedback_torque_scale_substeps.npy
applied_uipc_force_substeps_w.npy   # [T,8,3]，PhysX 实际施加反力
applied_uipc_torque_substeps_w.npy
opposing_contact_force_substeps_w.npy
backing_contact_force_substeps_w.npy
uipc_boundary_surface_sync_error_mm.npy
uipc_reaction_vertex_count.npy
uipc_step_time_sec.npy
frame_wall_time_sec.npy             # 整帧 wall time
uipc_substep_time_sec.npy           # [T,8]，用于定位 >0.5 s 慢帧的子步

object_pose_w.npy
object_velocity_w.npy
pad_pose_w.npy
gripper_opening_mm.npy
commanded_gripper_opening_mm.npy
minimum_signed_gap_mm.npy           # 只诊断，不参与控制
```

将 `[T,8,3]` 数组展平为连续子步序列后，对于 `n >= 1` 必须满足：

```text
candidate = applied[n-1] + alpha * (
    admissible_reaction[n-1] - applied[n-1]
)
applied[n] = norm_limit(candidate)
```

## 离线三轴场和视频

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/render_tactile_field_offline.py \
  --input_dir /tmp/openworldtactile_uipc_v6_2_run \
  --video_fps 15
```

## 一次性验收

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/validate_v6_2_once.py \
  --input_dir /tmp/openworldtactile_uipc_v6_2_run \
  --penetration_tolerance_mm 0.15 \
  --precontact_deformation_tolerance_mm 0.25 \
  --minimum_object_lift_mm 20 \
  --fail_on_failure
```

验收器检查：单个 link8 求解膜声明、link7 无 UIPC 表示、正式物体 pose write 为零、
固定 Pad-local 膜、姿态对齐、物理背板、初始相对位置不变、外部边界无独立动力学与
边界表面同步、8 次 PhysX/UIPC 等步长子循环、单边接触锥和限幅反馈递推、记录平均、
接触前机械臂运动不造成膜形变、接触反力阈值、穿透容差、冻结 7g 重算、圆柱至少
抬升 20 mm、释放回零，以及可选的离线力场守恒和视频完整性。

2026-07-15 的完整 545 帧默认参数验证中，圆柱最大抬升 39.75 mm，提升末端相对
滑移约 0.057 mm；最慢帧 0.365 s，没有超过 0.5 s。最小 signed gap 为
-0.0319 mm，且只有两帧低于 -0.02 mm，满足 0.15 mm 离线验收容差。
