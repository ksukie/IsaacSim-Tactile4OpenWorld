# dual_openworldtactile_physical_lift 结果展示说明

## 脚本结果

`dual_openworldtactile_physical_lift_rgb_fxyz_ui.py` 运行后，会在 Isaac Sim 中显示两个竖直相对的 OpenWorldTactile/GelSight 触觉传感器夹住一个竖起的 M16 螺母。螺母孔轴对准左右传感器的按压方向，因此触觉图中间应保留孔洞区域。

夹住后，脚本只移动左右传感器本体向上提，螺母不会被代码绑定跟随。螺母能否被带起，取决于左右传感器和螺母之间的真实接触、摩擦和夹持力。

运行时 UI 窗口标题为：

```text
Dual OpenWorldTactile Physical Lift GelSight/OpenWorldTactile RGB + FXYZ
```

窗口中一共展示 8 张图，分为上下两行。

## 展示图片含义

| 图片名称 | 内容说明 |
| --- | --- |
| GELSIGHT LEFT RGB | 左侧 GelSight 触觉 RGB 图，显示左侧传感器看到的螺母接触形状。 |
| GELSIGHT RIGHT RGB | 右侧 GelSight 触觉 RGB 图，显示右侧传感器看到的螺母接触形状。 |
| OWT LEFT RGB | 左侧 OpenWorldTactile 触觉 RGB 图，用于后续 SDK 解算。 |
| OWT RIGHT RGB | 右侧 OpenWorldTactile 触觉 RGB 图，用于后续 SDK 解算。 |
| GELSIGHT LEFT FORCE FIELD | 左侧 GelSight/OpenWorldTactile 触觉力场图，显示左侧接触力分布。 |
| GELSIGHT RIGHT FORCE FIELD | 右侧 GelSight/OpenWorldTactile 触觉力场图，显示右侧接触力分布。 |
| OWT LEFT FXYZ | 左侧 OpenWorldTactile RGB 经 SDK 解算后的 FXYZ 图。 |
| OWT RIGHT FXYZ | 右侧 OpenWorldTactile RGB 经 SDK 解算后的 FXYZ 图。 |

## 运行指令

```bash
./isaaclab.sh -p experiments/rgb-pipeline/dual_openworldtactile_physical_lift_rgb_fxyz_ui.py
```
