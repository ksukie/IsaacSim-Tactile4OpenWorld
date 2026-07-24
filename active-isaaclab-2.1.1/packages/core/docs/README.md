# Core tactile package / 触觉核心包

[English](#english) · [简体中文](#简体中文)

<a id="english"></a>

## English

`openworldtactile` is the runtime package for tactile simulation. It contains reusable sensor abstractions and implementations for image-, marker-, and deformation-based tactile signals.

Install it through the repository wrapper rather than running this directory directly:

```bash
cd active-isaaclab-2.1.1
./run.sh --install
./run.sh --python -c "import openworldtactile; print('openworldtactile: OK')"
```

The main implementation families are under `openworldtactile/simulation_approaches/`:

- `taxim`: example-based tactile RGB rendering;
- `fots`: marker-motion simulation;
- `uipc`: deformable membrane integration backed by the separately installed UIPC package;
- supporting geometry, rendering, and sensor utilities used by the task and experiment packages.

Sensor-specific dimensions, USD paths, and robot assemblies belong to `openworldtactile_assets`; task registrations belong to `openworldtactile_tasks`. For public APIs, lifecycle expectations, and a minimal integration pattern, see the [custom integration guide](../../../../docs/en/guides/custom-integration.md).

<a id="简体中文"></a>

## 简体中文

`openworldtactile` 是触觉仿真的运行时包，提供可复用的传感器抽象，以及图像、标记点和形变触觉信号实现。

请通过仓库包装脚本安装，不要把本目录当作独立程序直接运行：

```bash
cd active-isaaclab-2.1.1
./run.sh --install
./run.sh --python -c "import openworldtactile; print('openworldtactile: OK')"
```

主要实现位于 `openworldtactile/simulation_approaches/`：

- `taxim`：基于样例的触觉 RGB 渲染；
- `fots`：标记点运动仿真；
- `uipc`：由单独安装的 UIPC 包支持的可形变膜集成；
- 供任务和实验包复用的几何、渲染与传感器辅助模块。

传感器尺寸、USD 路径和机器人装配配置属于 `openworldtactile_assets`；任务注册属于 `openworldtactile_tasks`。公共 API、生命周期约定和最小集成方式见[自定义集成指南](../../../../docs/zh-CN/guides/custom-integration.md)。
