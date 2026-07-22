# Depth 到 SDK 三轴力方案说明

本次任务是把 OpenWorldTactile 仿真触觉数据接入 OpenWorldTactile SDK，让 SDK 按 OpenWorldTactile 传感器的 RGB 图像流程输出三轴力 fx、fy、fz。

实现方式是：在 OpenWorldTactile 内部用 depth 和 shear 生成 OpenWorldTactile-style RGB，并继续写入 tactile_rgb_image。bridge 仍然只读取 tactile_rgb_image，不直接读取 depth 或 shear。

SDK 会对比无接触 baseline RGB 和当前 RGB：通过 Hue 变化计算 fz，通过两帧图像光流计算 fx 和 fy。

最终链路为：OpenWorldTactile depth 和 shear 生成 OpenWorldTactile-style RGB，bridge 读取 RGB，SDK 根据两帧 RGB 差异输出三轴力。
