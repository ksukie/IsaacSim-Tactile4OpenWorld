# Depth 到 SDK 三轴力链路

## 目标

把 OpenWorldTactile 内部数据渲染成 OpenWorldTactile-style RGB，再交给 OpenWorldTactile SDK 从 RGB 图像反算 `fx, fy, fz`。

核心链路：

```text
OpenWorldTactile depth diff + tactile_shear_force
  -> OpenWorldTactileRGBRenderer
  -> tactile_rgb_image
  -> IsaacLabOpenWorldTactileBridge
  -> VirtualTouchSensor
  -> OpenWorldTactile SDK
  -> fx, fy, fz
```

## RGB 生成

OpenWorldTactile 相机侧先计算深度差：

```python
diff = nominal_depth - current_depth
```

`diff.squeeze(-1)` 表示每个像素的压入量，主要用于生成 RGB 的 Hue 变化。

OpenWorldTactile force field 同时提供：

```python
tactile_shear_force
```

它表示每个触觉点的 x/y 切向力，只用于生成当前 RGB 帧里的纹理位移，不直接交给 SDK 算 `fx/fy`。

```text
depth diff
  -> Hue 变化

tactile_shear_force
  -> 纹理位移

Hue 变化 + 纹理位移
  -> 当前帧 tactile_rgb_image
```

## SDK 计算

SDK 只接收 RGB，不直接接收 depth 或 shear。

```text
baseline RGB + 当前 RGB
  -> Hue 做差
  -> fz

baseline RGB + 当前 RGB
  -> Farneback optical flow
  -> fx, fy
```

也就是说：

```text
depth 通过 Hue 影响 fz
shear 通过纹理位移影响 fx/fy
SDK 最终根据两帧 RGB 的差异计算三轴力
```

## 关键参数

```python
render_cfg=GELSIGHT_R15_CFG.replace(
    openworldtactile_max_pressure=6e-4,
    openworldtactile_base_value=220,
    openworldtactile_pressure_blur=5,
    openworldtactile_displacement_scale=12000.0,
)
```

```text
openworldtactile_max_pressure       控制 depth 到 Hue 的尺度
openworldtactile_displacement_scale 控制 shear 到纹理位移的尺度
```

## 运行

使用 demo：

```text
experiments/sensors/openworldtactile_finger_sensor.py
```

开窗口运行：

```bash
cd "${ISAACLAB_ROOT}"

rm -rf tactile_record_openworldtactile

HEADLESS=0 LIVESTREAM=0 ENABLE_CAMERAS=1 TERM=xterm ./isaaclab.sh -p experiments/sensors/openworldtactile_finger_sensor.py \
  --use_tactile_rgb \
  --use_tactile_ff \
  --contact_object_type nut \
  --num_envs 1 \
  --enable_cameras \
  --save_viz \
  --save_viz_dir tactile_record_openworldtactile
```

输出：

```text
tactile_record_openworldtactile/openworldtactile_virtual_forces.csv
tactile_record_openworldtactile/tactile_rgb_image/
tactile_record_openworldtactile/tactile_force_field/
```

图片按 200 帧循环覆盖。
