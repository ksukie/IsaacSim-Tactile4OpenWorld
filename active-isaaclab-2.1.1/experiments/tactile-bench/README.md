# OpenWorldTactile tactile bench / OpenWorldTactile 触觉实验台

[English](#english) · [简体中文](#简体中文)

<a id="english"></a>

## English

This directory contains the UIPC membrane experiments and offline utilities used to produce dense three-axis tactile fields. Start with the maintained [experiment guide](../../../docs/en/guides/experiments.md); the files under [`docs/`](docs/) are chronological research records and are not the public setup guide.

### Choose an entry point

| Goal | Entry point |
|---|---|
| Verify a new installation | `OpenWorldTactile_v1.py` |
| Run the current closed-loop Piper grasp | `OpenWorldTactile_v6_2_grasp.py` |
| Render an existing V6.2 result | `render_tactile_field_offline.py` |
| Validate an existing V6.2 result | `validate_v6_2_once.py` |
| Integrate the estimator in another program | modules under `api/` plus `OpenWorldTactile_v5_new_7g_deformation_force_estimator.py` |

Run scripts through the mainline wrapper from `active-isaaclab-2.1.1/`:

```bash
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v1.py --help
```

V1 is self-contained after installation. V6.2 additionally requires a valid deformation contract produced by the V5.7d → V5.7e → V5.7f preparation sequence. Do not point V6.2 at an arbitrary directory. The exact commands, expected files, validation steps, force-axis convention, and output units are documented in the [experiment guide](../../../docs/en/guides/experiments.md).

### Data contract

The main tactile result is an array shaped `[T, 300, 300, 3]`. Channels are local-Y shear, local-Z shear, and local-X normal force. Unless a separate calibration has been applied, values use the label `sim_constitutive_force` and must not be reported as Newtons.

### About the local research notes

The versioned files under [`docs/`](docs/) preserve implementation decisions and experiment provenance. They may use the language and assumptions of the original research process. User-facing instructions are maintained in matched English and Chinese pages under the repository-level `docs/en/` and `docs/zh-CN/` trees.

<a id="简体中文"></a>

## 简体中文

本目录包含基于 UIPC 膜形变生成稠密三轴触觉力场的实验脚本和离线工具。请从持续维护的[实验指南](../../../docs/zh-CN/guides/experiments.md)开始；[`docs/`](docs/) 下的文件是按时间保留的研究记录，不是公开用户的安装指南。

### 选择入口

| 目标 | 入口 |
|---|---|
| 验证新安装 | `OpenWorldTactile_v1.py` |
| 运行当前 Piper 闭环抓取 | `OpenWorldTactile_v6_2_grasp.py` |
| 渲染已有 V6.2 结果 | `render_tactile_field_offline.py` |
| 校验已有 V6.2 结果 | `validate_v6_2_once.py` |
| 集成到其他程序 | `api/` 下的模块与 `OpenWorldTactile_v5_new_7g_deformation_force_estimator.py` |

请在 `active-isaaclab-2.1.1/` 中通过主线包装脚本运行：

```bash
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v1.py --help
```

完成安装后，V1 可独立运行。V6.2 还要求使用 V5.7d → V5.7e → V5.7f 准备流程生成有效的 deformation contract，不能随意指定一个目录代替。完整命令、预期文件、校验步骤、力轴约定和输出单位见[实验指南](../../../docs/zh-CN/guides/experiments.md)。

### 数据约定

主要触觉结果数组形状为 `[T, 300, 300, 3]`，三个通道依次表示局部 Y 切向力、局部 Z 切向力和局部 X 法向力。未另行完成标定时，数值单位标签为 `sim_constitutive_force`，不得直接作为牛顿值报告。

### 关于本目录的研究记录

[`docs/`](docs/) 下的版本化文件用于保留实现决策和实验来源，可能沿用原始研究过程的语言与上下文。面向用户的说明统一在仓库级 `docs/en/` 与 `docs/zh-CN/` 中以中英镜像形式维护。
