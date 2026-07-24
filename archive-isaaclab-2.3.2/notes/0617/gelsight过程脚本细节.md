# GelSight 过程脚本细节

## openworldtactile_rgb_fxyz_ui.py

### 脚本目标

该脚本是最基础的单传感器版本，用一个 OpenWorldTactile/GelSight 触觉传感器和一个 M16 螺母构成接触场景。它主要验证两条触觉数据能否稳定跑通：

- 相机触觉 RGB 图像。
- SDF/force-field 计算得到的 FXYZ 力分量图。

### 数据细节

RGB 图来自 OpenWorldTactile/GelSight 的相机触觉路径。脚本先采集无接触 baseline，然后在接触过程中显示当前触觉 RGB。

FXYZ 来自传感器输出的：

- `tactile_normal_force`
- `tactile_shear_force`

其中 `tactile_normal_force` 对应 Fz，`tactile_shear_force[..., 0]` 和 `tactile_shear_force[..., 1]` 对应 Fx/Fy。

### 过程细节

脚本启动后先进行无接触 warmup，并保存 tactile baseline。之后根据 `contact_start_step` 开始给螺母施加向下力，同时对不同环境施加正负 Z 向扭矩。这样可以观察螺母压入软胶后 RGB 图和 FXYZ 图的变化。

该脚本的作用是建立后续所有脚本的基础验证：先确认单个 GelSight/OpenWorldTactile 传感器能在 UI 中同时显示 RGB 和力分量。

## dual_openworldtactile_clamp_rgb_fxyz_ui.py

### 脚本目标

该脚本从单传感器版本扩展到左右两个竖直相对的 OpenWorldTactile/GelSight 触觉传感器，用两个触觉面夹合 M16 螺母。它主要验证双传感器夹合场景下，左右触觉 RGB 图像能否同步显示，并进一步验证 RGB 输入 SDK 后是否能得到压力灰度图和 Fx/Fy 箭头图。

### 数据细节

左右 RGB 图分别来自左右触觉传感器的 `tactile_rgb_image`。这两张图表示两个触觉面在夹合螺母时各自看到的触觉 RGB 变化。

SDK 箭头图来自 RGB 后处理链路：

```text
tactile RGB -> IsaacLabOpenWorldTactileBridge.update(rgb) -> hue/flow -> gray + arrows
```

其中灰度图主要表示 SDK 解算出的 Fz/pressure，箭头表示 SDK optical flow 对应的 Fx/Fy 趋势。该脚本虽然启用了 force field 传感器，但 UI 中主要展示 RGB 和 SDK arrows，不展示 SDF FXYZ 图。

### 过程细节

脚本启动后先让左右传感器保持打开状态，并分别采集左右无接触 tactile baseline。之后进入循环夹合过程：先从 `open-baseline` 进入 `closing`，再进入 `holding`，最后回到 `opening`。

每轮循环结束后会重新采集 baseline，避免 RGB 和 SDK 解算受到上一轮残留状态影响。该脚本的作用是把单触觉面扩展成双触觉面，并验证左右 RGB 与 RGB-to-SDK 箭头图是否能在同一 UI 中稳定对比。

## dual_openworldtactile_physical_lift_rgb_fxyz_ui.py

### 脚本目标

该脚本是当前最完整的独立双传感器物理夹持版本。它演示两个竖直相对的 OpenWorldTactile/GelSight 触觉传感器先夹住竖起的 M16 螺母，再只移动传感器本体向上提。螺母孔轴对准左右传感器按压方向，中间孔洞应在触觉图中保留为空。螺母不会被代码绑定跟随，是否能被带起完全由真实接触、摩擦和闭合挤压力决定。

它主要验证四类触觉结果能否在同一过程中对比：

- GelSight RGB。
- OpenWorldTactile RGB。
- GelSight/OpenWorldTactile force field。
- OpenWorldTactile RGB 输入 SDK 解算出的 FXYZ。

### 数据细节

GelSight RGB 和 OpenWorldTactile RGB 都来自各自触觉传感器的 `tactile_rgb_image`。其中 GelSight RGB 对应上分支：

```text
Depth Rendering -> Depth Image -> Depth-to-RGB Mapping -> Tactile Image (RGB)
```

OpenWorldTactile FXYZ 图来自 OpenWorldTactile RGB 后处理链路：

```text
OpenWorldTactile RGB -> SDK hue/flow -> FXYZ
```

GelSight/OpenWorldTactile force field 对应下面分支：

```text
Signed Distance Field & Gradient
-> Interpenetration Depth
-> Penalty-based Model
-> Tactile Force Field
```

force field 图使用 SDF/penetration/penalty 模型算出的 normal force 和 shear force，不使用 RGB，也不使用 SDK 输出。图中颜色表示法向压力 Fz，箭头表示切向力 Fx/Fy。

### 过程细节

脚本启动后先让左右触觉传感器保持打开状态，并采集左右无接触 tactile baseline。之后进入完整的物理夹持循环：

- `open-baseline`
- `closing`
- `holding`
- `lifting-physical`
- `lifted-hold`
- `lowering-physical`
- `opening`

关键约束是：上提阶段只写左右 OpenWorldTactile 传感器的 root pose，不写螺母的 root pose。因此螺母不会被代码强制带起，能否跟随上移取决于 PhysX 接触、摩擦和夹持压力。

该脚本的作用是把 GelSight RGB、OpenWorldTactile RGB、OpenWorldTactile SDK FXYZ 和 GelSight/OpenWorldTactile force field 放在同一个物理夹持上提过程中对比，是当前最完整的 GelSight/OpenWorldTactile 双指过程脚本。

## franka_openworldtactile_grasp_rgb_fxyz_ui.py

### 脚本目标

该脚本尝试把 OpenWorldTactile/GelSight 触觉显示接入 Franka Panda 机械臂夹取 M16 螺母的流程中。文件顶端中文注释已标注“失败”，因此它目前是一个机械臂集成尝试记录，而不是稳定可用版本。

它主要尝试验证：

- Franka 夹爪运动能否带动左右 OpenWorldTactile/GelSight tactile pad。
- 左右触觉 RGB 能否跟随机械臂夹取过程实时显示。
- 左右 FXYZ 力分量图能否在机械臂夹取过程中同步显示。

### 数据细节

左右 RGB 图来自左右 OpenWorldTactile/GelSight 传感器的 `tactile_rgb_image`。

左右 FXYZ 图来自各自传感器输出的：

- `tactile_normal_force`
- `tactile_shear_force`

其中 `tactile_normal_force` 对应 Fz，`tactile_shear_force[..., 0]` 和 `tactile_shear_force[..., 1]` 对应 Fx/Fy。FXYZ 的颜色编码与前面脚本一致：

- R：`|Fx|`
- G：`|Fy|`
- B：`max(Fz, 0)`

该脚本中的 OpenWorldTactile/GelSight tactile pad 不是 Franka 原生手指模型的一部分，而是根据 Franka 左右指尖位置进行运动学跟随。

### 过程细节

脚本启动后先解析 Franka 的 arm joint、finger joint、hand body 和左右 finger body。随后把螺母放到左右指尖中心附近，并让左右 OpenWorldTactile pad 根据 Franka 指尖位置计算中心、轴向和间距，从而跟随指尖移动。

抓取过程包括：

- `open-baseline`
- `closing`
- `lifting`
- `holding`

其中机械臂末端上提由 differential IK 控制，夹爪闭合由 finger joint target 控制。由于该脚本目前标注为失败，说明 Franka、螺母、触觉 pad、RGB/FXYZ UI 和 IK 抓取流程虽然已经接到一起，但物理夹取稳定性或最终效果还没有达到可用状态。
