# 最小启动文档

## 进入环境

```bash
cd ~/IsaacLab-v2.3.2
conda activate isaaclab
```

## 跑主线 SDF Taxel

```bash
env TERM=xterm ./isaaclab.sh -p experiments/franka/current/sdf_taxel_rgb_sdk.py --max_steps 500
```

## 跑基线 COP + Friction

```bash
env TERM=xterm ./isaaclab.sh -p experiments/franka/current/cop_rgb_sdk_friction.py --max_steps 400
```

## 保存 3 帧

在上面任意命令后加：

```bash
--headless --hide_rgb_maps --max_saved_frames 3
```

