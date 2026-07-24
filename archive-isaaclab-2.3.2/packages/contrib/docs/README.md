# Archived Isaac Lab contrib package / 归档的 Isaac Lab contrib 包

[English](#english) · [简体中文](#简体中文)

<a id="english"></a>

## English

This directory belongs to the archived Isaac Lab 2.3.2 route. Its `isaaclab_contrib` snapshot includes experimental community components, including the historical OpenWorldTactile visual-tactile sensor integration used by the archived demos.

The OpenWorldTactile integration converts sensor geometry and contact/depth information into tactile images through the archived rendering pipeline. Some paths also expect the separately obtained external camera SDK described under `archive-isaaclab-2.3.2/hardware-sdk/`.

This package is not installed by `active-isaaclab-2.1.1/run.sh`, is not compatible with the current task package by default, and is retained for source-level reproducibility. Use the maintained 2.1.1 packages for new work. See [Legacy and archive](../../../../docs/en/guides/legacy.md) for isolation rules and limitations.

<a id="简体中文"></a>

## 简体中文

本目录属于归档的 Isaac Lab 2.3.2 路线。其中的 `isaaclab_contrib` 快照包含实验性社区组件，也包含归档演示所使用的历史 OpenWorldTactile 视觉触觉传感器集成。

OpenWorldTactile 集成通过归档渲染流程，把传感器几何和接触/深度信息转换为触觉图像。部分路径还要求使用 `archive-isaaclab-2.3.2/hardware-sdk/` 中说明的、需单独取得的外部相机 SDK。

该包不会由 `active-isaaclab-2.1.1/run.sh` 安装，默认也不与当前任务包兼容，仅用于源码级历史复现。新工作请使用持续维护的 2.1.1 包。隔离规则和限制见[旧版与归档](../../../../docs/zh-CN/guides/legacy.md)。
