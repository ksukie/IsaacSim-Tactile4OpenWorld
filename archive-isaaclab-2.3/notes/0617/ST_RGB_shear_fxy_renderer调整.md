# ST_RGB_shear fxy 与 RGB renderer 调整记录

日期：2026-06-17

## 文件

- `experiments/franka/current/ST_RGB_shear.py`
- 参考对比：`experiments/franka/current/sdf_taxel_rgb_sdk_openworldtactile_aligned.py`

## 本次修改

修改 `ST_RGB_shear.py`：

1. `--openworldtactile_max_pressure` 默认值从 `1.5e-3` 改为 `6.0e-4`。
2. `make_openworldtactile_renderer()` 改为返回 SDK 原生 `OpenWorldTactileRGBRenderer`。
3. 保留 SDF taxel、shear、SDK bridge、baseline、FXYZ 预览窗口、保存逻辑不变。
4. 自定义 `SdfContactWavelengthRgbRenderer` 暂时保留在文件中，方便后续 A/B 或回退。

当前 renderer 链路：

```text
SDF height_map + SDF shear displacement
-> OpenWorldTactileRGBRenderer
-> RGB frame
-> IsaacLabOpenWorldTactileBridge.update(rgb)
-> SDK hue + optical flow
-> fxyz preview
```

## 运行命令

```bash
./isaaclab.sh -p experiments/franka/current/ST_RGB_shear.py   --openworldtactile_fxyz_arrow_scale 2

```

## 后续注意

- 当前 `fx/fy` 仍是 SDK 光流结果，不是直接 SDF shear 求和。
- `--openworldtactile_fxyz_arrow_scale` 只是显示倍率，不改变 SDK 解算数值。
- 如果后续继续使用自定义 wavelength renderer，需要重新处理颜色、灰度纹理强度、pressure scale 和 displacement scale，否则容易再次导致 `fxy` 偏弱。
- 现在 `--wavelength_high_nm`、`--wavelength_low_nm`、`--openworldtactile_iso_saturation`、`--openworldtactile_displacement_clip_px` 对主 renderer 基本不再起作用；后续可清理或改成仅在实验 renderer 下生效。
