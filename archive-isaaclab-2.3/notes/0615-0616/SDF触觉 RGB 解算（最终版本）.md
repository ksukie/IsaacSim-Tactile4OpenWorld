# OpenWorldTactile 对齐版 SDF 触觉 RGB 简要实现

## 文件

- `experiments/franka/current/sdf_taxel_rgb_sdk_openworldtactile_aligned.py`
- 基于 `sdf_taxel_rgb_sdk.py` 新建实验版，原脚本不替换。

## 目标

验证 SDF taxel 生成的触觉 RGB 输入 SDK 后，`fx/fy/fz` 的显示和解算效果。

## 主要实现

- RGB 生成链路改为：`SDF penetration height_map + SDF shear displacement -> OpenWorldTactile RGB`。
- RGB 输出尺寸保持原虚拟 pad 尺寸：`pad_image_shape()`，不再强制 resize 到 `320x240`。
- 无接触基底保持红色：`openworldtactile_saturation=255`。
- SDK 仍使用 `IsaacLabOpenWorldTactileBridge.update(rgb)`，`fx/fy` 继续由 RGB 灰度光流计算。
- 光流窗口支持按当前 pad 图像尺寸自动缩放：`--openworldtactile_flow_winsize 0`。
- FXYZ 预览箭头支持按当前 pad 图像尺寸自动缩放：`--openworldtactile_fxyz_arrow_scale 0`、`--openworldtactile_fxyz_arrow_step 0`。
- 原始 `.npy` force map 保存逻辑保留，用于检查 SDF normal/shear。
- 控制流程、抓取状态机、物体/机器人配置不作为本实验改动目标。

## 运行

```bash
TERM=xterm ./isaaclab.sh -p experiments/franka/current/sdf_taxel_rgb_sdk_openworldtactile_aligned.py --max_steps 500
```

保存少量帧：

```bash
TERM=xterm ./isaaclab.sh -p experiments/franka/current/sdf_taxel_rgb_sdk_openworldtactile_aligned.py --max_steps 500 --hide_rgb_maps --max_saved_frames 3
```

## 当前注意点

- `fx/fy` 当前仍是 SDK 光流结果，不是直接 SDF shear 求和。
- RGB 转灰度后，颜色变化仍可能影响光流，后续如需更稳可继续调整 RGB 生成或光流 mask。
