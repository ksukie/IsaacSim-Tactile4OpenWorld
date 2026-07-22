# Franka 触觉/抓取实验索引

最后整理日期：2026-06-16

范围：本文档只索引近期 Franka 触觉 RGB / 抓取实验过程中创建的相关脚本。
本文档有意不整理 `experiments/sensors/_archive/`。

## 最终确定方案

最终确定的 Franka 触觉 RGB 工作流保留两个脚本。

### 主线：SDF Taxel 路线

脚本：

```text
experiments/franka/current/sdf_taxel_rgb_sdk.py
```

状态：当前主线，用于在 Panda 左右手指上挂虚拟 taxel pad，并对带 SDF collision mesh 的物体计算触觉力场。

用途：
- 用虚拟 SDF taxel pad 替代 COP 重建。
- 默认物体是 `{ENV_REGEX_NS}/Object` 下的 `Factory/factory_nut_m16.usd`。
- 必须找到 SDF collision mesh；如果没有 SDF mesh，脚本直接报错，不自动回退到其他方案。
- normal force 来自 SDF penetration depth。
- shear 来自 taxel 与物体之间的相对切向速度，并加 Coulomb friction cap。
- 渲染左/右 OpenWorldTactile RGB，并把 RGB 送入 OpenWorldTactile SDK 得到 `fx/fy/fz`。
- 保留 Isaac UI 四画面预览和保存输出。

推荐预览运行：

```bash
TERM=xterm ./isaaclab.sh -p experiments/franka/current/sdf_taxel_rgb_sdk.py \
    --max_steps 500
```

推荐保存运行：

```bash
TERM=xterm ./isaaclab.sh -p experiments/franka/current/sdf_taxel_rgb_sdk.py \
    --headless \
    --hide_rgb_maps \
    --max_steps 500 \
    --max_saved_frames 3
```

### 已确认保留：COP + Friction 基线

脚本：

```text
experiments/franka/current/cop_rgb_sdk_friction.py
```

状态：最终保留的 COP/friction 基线脚本，用于对比和调试。

用途：
- 使用官方 `Isaac-Lift-Cube-Franka-IK-Abs-v0` pick/lift 状态机。
- 在 Panda 左右手指上添加真实 PhysX `ContactSensor`。
- 用 COP 从稀疏 contact points 重建 dense normal force map。
- 启用 `track_friction_forces=True`。
- 使用 `contact_sensor.data.friction_forces_w` 作为 OpenWorldTactile texture displacement 的 shear 来源。
- 将 world friction 转到 finger local frame：
  - local `x` 驱动 `displacement_x`
  - local `z` 经过符号翻转后驱动 image-y displacement
- 保留左/右 OpenWorldTactile RGB + 左/右 SDK FXYZ 四画面预览。
- 保留 `.npy` force map、RGB PNG、gray PNG 保存逻辑。

重要区别：
- 这是已经确认保留的 COP/friction 基线，不是 SDF taxel 路线。
- 它适合在不依赖 SDF mesh 的情况下检查 RGB renderer、SDK bridge、预览窗口和保存流程。
- 它不应被理解为 OpenWorldTactile 风格的 SDF force-field 实现；上面的 SDF taxel 路线在思想上更接近 OpenWorldTactile 的 SDF force-field。

推荐预览运行：

```bash
TERM=xterm ./isaaclab.sh -p experiments/franka/current/cop_rgb_sdk_friction.py \
    --max_steps 400
```

推荐保存运行：

```bash
TERM=xterm ./isaaclab.sh -p experiments/franka/current/cop_rgb_sdk_friction.py \
    --headless \
    --hide_rgb_maps \
    --max_steps 400 \
    --max_saved_frames 3
```

输出：
- `outputs/franka_lift_object_contact_cop_rgb_friction/left_force_map/*.npy`
- `outputs/franka_lift_object_contact_cop_rgb_friction/right_force_map/*.npy`
- `outputs/franka_lift_object_contact_cop_rgb_friction/left_rgb/*.png`
- `outputs/franka_lift_object_contact_cop_rgb_friction/right_rgb/*.png`
- `outputs/franka_lift_object_contact_cop_rgb_friction/left_gray/*.png`
- `outputs/franka_lift_object_contact_cop_rgb_friction/right_gray/*.png`

## 早期 Camera-Based 主线

### `experiments/franka/current/lift_cube_tactile_rgb_sm.py`

状态：早期稳定的 camera-only 主线；现在已被上面的最终确定方案取代。

用途：
- 使用官方 `Isaac-Lift-Cube-Franka-IK-Abs-v0` 状态机。
- 默认能保持 Franka 抓取稳定。
- 在 `panda_leftfinger` 和 `panda_rightfinger` 下添加两个轻量级 camera-only tactile sensor。
- 保存左/右 RGB 图像，以及可选 debug depth 图像。
- 默认保存的 RGB 映射方式是 wavelength-style：
  `depth_diff -> 600nm..400nm -> RGB`。

重要说明：
- 这不是真正的 GelSight contact-surface imaging。
- 除非显式传入 `--use_mounted_gelsight_usd`，否则不会添加完整 GelSight USD assets。
- 不要把 `--use_mounted_gelsight_usd` 当作默认路径；该模式在抓取测试中不稳定。

推荐运行：

```bash
TERM=xterm ./isaaclab.sh -p experiments/franka/current/lift_cube_tactile_rgb_sm.py \
    --num_envs 1 \
    --max_steps 800 \
    --save_every 5 \
    --save_debug_depth
```

输出：
- `outputs/franka_lift_tactile_rgb/left/*.png`
- `outputs/franka_lift_tactile_rgb/right/*.png`
- `outputs/franka_lift_tactile_rgb/debug_depth_left/*.png`
- `outputs/franka_lift_tactile_rgb/debug_depth_right/*.png`

## 有用的构建模块

### `experiments/franka/stable_refs/franka_panda_gripper_basic.py`

状态：保留为最小 sanity check。

用途：
- 加载 Franka Panda。
- 移动 arm joints，并控制 gripper 打开/关闭。
- 当 Panda asset 或 gripper control 表现异常时，用它先检查机器人本身是否正常。

推荐用途：
- 如果机器人本体行为异常，先运行这个脚本，再继续调试触觉功能。

### `experiments/franka/stable_refs/franka_panda_auto_control.py`

状态：保留为 IK / control 参考。

用途：
- 演示对 `panda_hand` 的 Differential IK 控制。
- 单独控制 `panda_finger_joint.*`。
- 可作为 scripted end-effector motion + gripper open/close 的参考。

### `experiments/franka/stable_refs/franka_panda_grasp_release_loop.py`

状态：保留为稳定 grasp-loop 参考。

用途：
- 重复执行 scripted grasp/release loop。
- 使用 IK 控制机械臂运动，并控制 gripper 打开/关闭。
- 打印左/右手指接触力。

为什么重要：
- 这是 manager-based lift task 之外，一个稳定 Franka 抓取循环的好参考。

### `experiments/franka/stable_refs/franka_panda_contact_sensor_dynamic.py`

状态：保留为首选 contact-force diagnostic。

用途：
- 使用 dynamic cube。
- 使用更柔和的 gripper 参数。
- 输出经过 cube 过滤后的左/右 `ContactSensor` 数据。

优先使用它，而不是：
- `franka_panda_contact_sensor.py`，后者只是更早期、更简单的版本。

### `experiments/franka/stable_refs/franka_panda_tactile_pad_depth.py`

状态：保留为稳定 depth-image 参考。

用途：
- 避免使用外部 GelSight follower articulations。
- 在 Panda finger links 下添加 runtime tactile-pad depth cameras。
- 保存 no-contact baseline 与 current depth 之间的 difference images。

为什么重要：
- 这是早期 `lift_cube_tactile_rgb_sm.py` 方案最接近的稳定前身。
- 适合作为 camera-only depth-difference 实验的参考。

### `experiments/franka/stable_refs/franka_panda_grasp_force_rgb.py`

状态：保留为 synthetic force-to-RGB 参考。

用途：
- 仍然用 Panda hand 本体作为夹爪。
- 读取左/右 contact forces。
- 把 force vectors 编码成 synthetic RGB images。
- 解码/记录 force values，用于检查编码是否正确。

重要说明：
- 这里的 RGB 是确定性的 force encoding，不是 optical GelSight rendering。

## 次要 / 已被取代

### `experiments/franka/superseded/franka_panda_contact_sensor.py`

状态：保留，但只当作早期简单版本。

用途：
- 基础的左/右 Panda finger `ContactSensor` demo。
- 适合第一次检查 contact sensor 是否正常。

已被以下脚本取代：
- `franka_panda_contact_sensor_dynamic.py`，后者提供更真实的 dynamic-cube contact。

## 高风险 / 非主线

### `experiments/franka/high_risk/franka_panda_gelsight_mount_minimal.py`

状态：只保留为参考；不要作为默认路径。

用途：
- 添加完整 `LeftGelSight` 和 `RightGelSight` USD assets。
- 让它们 pose-follow 到 Panda fingers。
- 可启用 GelSight elastomer contact 与 OpenWorldTactile RGB / force-field sensing。

已观察到的风险：
- 完整 GelSight follower articulations 会干扰抓取物理。
- 在之前测试中，这类方案和抓取耦合后容易导致机械臂/物体不稳定。

更安全的用途：
- 只作为 geometry/contact debugging 参考。
- 如果重新测试它，优先使用 fixed-arm diagnostic mode。
- 除非明确是在测试 mounted GelSight behavior，否则不要把它混入当前稳定 lift 脚本。

## 非 Franka 参考

### `experiments/sensors/openworldtactile_finger_sensor.py`

状态：保留为 OpenWorldTactile RGB 参考，但它不是 Franka 抓取脚本。

用途：
- Standalone OpenWorldTactile GelSight finger demo。
- 使用 USD 自带的 `elastomer_tip/cam`。
- 通过 `get_initial_render()` 捕获 no-contact baseline。
- 计算 `nominal_depth - current_depth`。
- 通过 `VisuoTactileSensor` 渲染 OpenWorldTactile RGB。

为什么重要：
- 它是目标 RGB mapping 逻辑的参考。
- 它本身不能解决 Franka-mounted setup 的物理安装/接触稳定性问题。

## 输出目录

### `outputs/franka_lift_tactile_rgb/`

状态：早期 camera-only 输出。

内容：
- 80 张 left RGB frames。
- 80 张 right RGB frames。
- 80 张 left debug-depth frames。
- 80 张 right debug-depth frames。

含义：
- 早期稳定 Franka lift run，使用轻量级 finger cameras。

### `tactile_record_franka_gelsight/`

状态：mounted-GelSight 实验输出。

含义：
- 来自完整 GelSight follower 方案的输出。
- 可作为视觉参考，但不是当前稳定路径。

### `tactile_record_openworldtactile/`、`tactile_record_openworldtactile_cube/`、`tactile_record_openworldtactile_demo/`

状态：非 Franka OpenWorldTactile 参考输出。

含义：
- 适合用来对比 standalone OpenWorldTactile 行为。
- 不能证明 Franka-mounted setup 的物理实现正确。

## 推荐下一步

后续 Franka 工作从最终 SDF taxel 路线开始：

```text
experiments/franka/current/sdf_taxel_rgb_sdk.py
```

如果要做 COP/friction 对比，或调试 renderer / SDK 流程，使用：

```text
experiments/franka/current/cop_rgb_sdk_friction.py
```

稳定的底层参考脚本：

```text
experiments/franka/stable_refs/franka_panda_tactile_pad_depth.py
experiments/franka/stable_refs/franka_panda_grasp_release_loop.py
experiments/franka/stable_refs/franka_panda_contact_sensor_dynamic.py
```

在完整 mounted-GelSight 物理问题被隔离清楚之前，不要把 `experiments/franka/high_risk/franka_panda_gelsight_mount_minimal.py` 作为主线。
