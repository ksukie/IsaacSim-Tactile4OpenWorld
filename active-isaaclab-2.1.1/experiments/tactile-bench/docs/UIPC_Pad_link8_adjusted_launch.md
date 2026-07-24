# UIPC_Pad link8 adjusted launch

## 推荐启动方式

这个 preset 使用已经手动调好的 UIPC_Pad 挂载位姿：

```bash
cd "${OWT_ROOT}"

./run.sh -p experiments/tactile-bench/OpenWorldTactile_pad_usd_viewer.py \
  --use_adjusted_link8_preset
```

默认输出：

```text
/tmp/openworldtactile_uipc_pad_loaded_adjusted/metadata.json
/tmp/openworldtactile_uipc_pad_loaded_adjusted/camera_rgb/
```

## preset 等价参数

`--use_adjusted_link8_preset` 等价于使用下面这套核心参数：

```bash
--add_robot \
--mount_link_path /World/envs/env_0/Robot/link8 \
--pad_rotation_frame local \
--pad_x_mm -0.712491 \
--pad_y_mm -10.564254 \
--pad_z_mm -1.977508 \
--pad_roll_deg 145.758588 \
--pad_pitch_deg 89.999263 \
--pad_yaw_deg 150.755001 \
--render_viewport \
--save_camera_rgb \
--camera_width 640 \
--camera_height 480 \
--camera_save_every 30 \
--camera_save_final
```

如果需要换输出目录：

```bash
./run.sh -p experiments/tactile-bench/OpenWorldTactile_pad_usd_viewer.py \
  --use_adjusted_link8_preset \
  --output_dir /tmp/my_uipc_pad_run
```

如果要从这套已调好位置继续微调：

```bash
./run.sh -p experiments/tactile-bench/OpenWorldTactile_pad_usd_viewer.py \
  --use_adjusted_link8_preset \
  --manual_adjust \
  --output_dir /tmp/my_uipc_pad_refine
```

## 挂载规则

- USD 直接 reference 到 `/World/envs/env_0/Robot/link8/UIPC_Pad`。
- 不使用 `UIPC_Pad_MotionFrame` 或其他中间父节点。
- 不修改 `UIPC_Pad.usda` 内部的相机、红色膜、visual、simulation 子节点相对位姿。
- preset 中的姿态是 `link8` 下的 local pose，后续机械臂运动时整个 USD 会刚性跟随 `link8`。

## 后续抓取版本的力来源

后续抓取/UIPC 版本必须遵守单独的 force contract：

```text
experiments/tactile-bench/docs/UIPC_Pad_force_contract.md
```

核心定义：

- `simulation/membrane_sim_mesh` 是唯一触觉计算源。
- fxyz 来自 UIPC 膜表面形变，再经过 v2.7 / v5_new_6 的弹簧-阻尼-摩擦本构估计器。
- `camera_surface`、`visual_back_mesh`、contact penetration、PhysX force、native UIPC contact force、pressure gradient 都不能作为 fxyz 来源。
- 最终输出必须是 pad-local fxyz：`fx=local Y shear`，`fy=local Z shear`，`fz=local X compression`。

## 红色膜是否可以被穿透

在这个 `OpenWorldTactile_pad_usd_viewer.py` 里，红色膜只是被加载出来的 UIPC_Pad USD 内部仿真膜网格；脚本不创建抓取物体，也不启动 UIPC 软体求解器。因此这个 viewer 不能把红色膜当成真正会阻挡物体的软体接触面。

实际含义：

- 只看这个 viewer：红色膜主要用于检查相机看到的内容和挂载姿态，物体如果被放到同一位置，可能会和膜发生几何重叠。
- 若没有给膜和物体配置有效 PhysX collision，物体可以直接穿过它。
- 即使有普通 PhysX collision，它也只是刚体/碰撞近似，不等于 UIPC 软膜形变。
- 只有在后续接入 UIPC deformable/contact solve 时，红色膜才是参与软体接触求解的膜；那时仍可能因为时间步、接触厚度、材料刚度、网格分辨率或高速运动出现数值上的轻微穿透。

所以当前这套启动方法用于确认“挂载位置、相机姿态、相机拍到红膜”的结果，不用于验证红膜物理不可穿透。
