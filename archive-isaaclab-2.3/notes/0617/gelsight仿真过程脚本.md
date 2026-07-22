# GelSight 仿真过程脚本

本文档基于 `experiments/rgb-pipeline/` 目录下各 Python 脚本顶端的中文说明整理，只记录各脚本相对前一阶段的主要改进。

## openworldtactile_rgb_fxyz_ui.py

该脚本演示螺母接触 OpenWorldTactile/GelSight 触觉传感器，并在 Isaac Sim UI 中实时展示触觉 RGB 和 FXYZ。

## dual_openworldtactile_clamp_rgb_fxyz_ui.py

该脚本演示两个竖直相对的 OpenWorldTactile/GelSight 触觉传感器夹合螺母，并在 Isaac Sim UI 中只实时展示左右触觉 RGB 和 RGB 输入 SDK 解算出的箭头图。

## dual_openworldtactile_physical_lift_rgb_fxyz_ui.py

该脚本演示两个竖直相对的 OpenWorldTactile/GelSight 触觉传感器先夹住竖起的螺母，再只移动传感器本体上提；螺母孔轴对准左右传感器按压方向，中间孔洞应在触觉图中保留为空。螺母不会被代码绑定跟随，是否被带起完全由接触和摩擦决定。UI 实时展示 GelSight RGB、OpenWorldTactile RGB、GelSight/OpenWorldTactile force field，以及 OpenWorldTactile RGB 输入 SDK 解算出的 FXYZ。

## franka_openworldtactile_grasp_rgb_fxyz_ui.py

该脚本演示 Franka 机械臂夹取螺母，并在 Isaac Sim UI 中实时展示左右触觉 RGB 和 FXYZ。

当前标注：

- 失败。
