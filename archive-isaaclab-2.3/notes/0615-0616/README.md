# 0616 Franka 触觉仿真归档

日期：2026-06-16  
主题：Franka 夹爪接触物体后，生成左右触觉力图、OpenWorldTactile RGB 和 SDK FXYZ。

## 文档

- [当前最终实现.md](当前最终实现.md)：当前主线方案的实现流程和运行指令。
- [0615-0616所有测试脚本.md](0615-0616所有测试脚本.md)：`experiments/franka/` 下每个 `.py` 的采用方法和运行指令。

## 当前主脚本

```bash
experiments/franka/current/cop_rgb_sdk.py
```

## 当前实现

```text
ContactSensor
-> COP + 最大熵 force_map
-> OpenWorldTactile pressure_map
-> 局部切向力生成 texture displacement
-> OpenWorldTactileRGBRenderer 生成带纹理 RGB
-> OpenWorldTactile SDK 解算 FXYZ
-> Isaac UI 显示 LEFT/RIGHT RGB + LEFT/RIGHT FXYZ
```

## 归档逻辑

```text
experiments/franka/current/
-> 当前保留脚本

experiments/franka/stable_refs/
experiments/franka/high_risk/
experiments/franka/superseded/
-> 本次归档中统一视为失败/历史案例
```
