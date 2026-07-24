<p align="right">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <img src="docs/media/openworldtactile-logo-lockup-v6.png" alt="OpenWorldTactile 标志">
</p>

<h1 align="center">IsaacSim-Tactile4OpenWorld</h1>

<p align="center">
  面向 Isaac Sim 与 Isaac Lab 的接触、形变、力场和视触觉仿真项目。
</p>

<p align="center">
  <a href="docs/zh-CN/getting-started/installation.md">安装</a> ·
  <a href="docs/zh-CN/getting-started/quick-start.md">快速开始</a> ·
  <a href="docs/README.zh-CN.md">文档中心</a> ·
  <a href="CITATIONS.zh-CN.md">引用</a>
</p>

> [!IMPORTANT]
> 这是研究代码库，不是独立的 Isaac Lab 发行版。当前路线依赖外部 Isaac Lab 2.1.1 / Isaac Sim 4.5 环境；UIPC 实验还会编译仓库内的 libuipc 源码。安装前请先阅读[兼容性与验证状态](docs/zh-CN/reference/compatibility.md)。

## 项目提供什么

OpenWorldTactile 将触觉感知与接触研究整合到统一的 Isaac Lab 工作流中：

- 基于 UIPC/libuipc 的柔性膜以及刚体/可形变接触仿真；
- GelSight Mini 资产，以及 Taxim、FOTS、FEM、RGB、标记点运动与力场方法；
- AgileX Piper、Franka 和 Allegro 集成；
- 抓取、抬升、插孔、滚动与强化学习环境；
- NumPy、图像、视频、JSON 和 HDF5 检查工具。

本项目中的“开放世界”指实验可以扩展到不同机器人、传感器、物体和接触条件；不表示项目提供通用世界模型或任意场景下的零样本控制。

<p align="center">
  <img src="docs/media/openworldtactile-hero.png" alt="OpenWorldTactile 接触与触觉信号流程">
</p>

## 按目标选择入口

| 目标 | 从这里开始 |
|---|---|
| 安装当前主线 | [安装指南](docs/zh-CN/getting-started/installation.md) |
| 验证环境并运行第一个触觉仿真 | [快速开始](docs/zh-CN/getting-started/quick-start.md) |
| 运行 V1 基准或进阶 V6.2 抓取流程 | [实验指南](docs/zh-CN/guides/experiments.md) |
| 训练或检查已注册的 Isaac Lab 任务 | [任务与训练](docs/zh-CN/guides/tasks-and-training.md) |
| 读取和导出运行结果 | [数据与输出](docs/zh-CN/guides/data-and-outputs.md) |
| 将项目包接入自定义场景 | [自定义集成](docs/zh-CN/guides/custom-integration.md) |
| 诊断安装或运行错误 | [故障排查](docs/zh-CN/help/troubleshooting.md) |

## 首次运行

安装外部 Isaac Lab 环境和所需构建工具后，执行：

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab
cd active-isaaclab-2.1.1

./run.sh --install all
./run.sh --python -m unittest discover \
  -s experiments/tactile-bench \
  -p "test_membrane_local_frame.py" -v

./run.sh --python experiments/tactile-bench/OpenWorldTactile_v1.py \
  --headless \
  --shape sphere \
  --indent_depth_mm 0.6 \
  --rub_distance_mm 0.0 \
  --front_segments_y 24 \
  --front_segments_z 30 \
  --thickness_segments 3 \
  --tet_edge_length_r 0.05 \
  --warmup_steps 5 \
  --approach_steps 20 \
  --indent_steps 40 \
  --rub_steps 0 \
  --release_steps 20 \
  --output_dir "$PWD/outputs/v1-smoke" \
  --workspace_dir "$PWD/outputs/v1-smoke-workspace"
```

V1 成功运行后，会在指定目录写入 `fxyz.npy`、`metadata.json` 和可视化预览。该冒烟测试需要 UIPC，但不需要 V6.2 使用的冻结形变契约。完整验证步骤和预期结果见[快速开始](docs/zh-CN/getting-started/quick-start.md)。

## 仓库结构

```text
active-isaaclab-2.1.1/   当前包、任务、资产与实验
archive-isaaclab-2.3.2/  历史 GelSight/SDK 集成；不是默认路线
docs/                    英文与简体中文镜像用户文档
tools/repository/        仅供维护者使用的静态审计与清单工具
```

OpenWorldTactile 的实验标签（`V1` 至 `V6.2`）是研究里程碑，不是 Isaac Lab 版本。选择历史脚本前请阅读[实验版本谱系](docs/zh-CN/reference/experiment-lineage.md)。

## 项目状态与问题反馈

源码树已通过仓库级静态检查，但公开版本不宣称所有 GPU、驱动、UIPC 构建、任务、归档脚本或硬件集成都已重新运行验证。反馈问题时，请提供操作系统、GPU 与驱动、Isaac Sim 与 Isaac Lab 版本、安装命令、完整报错和最小复现命令。建议先查阅[常见问题](docs/zh-CN/help/faq.md)与[故障排查](docs/zh-CN/help/troubleshooting.md)。

## 引用、许可证与贡献

项目引用元数据见 [`CITATION.cff`](CITATION.cff)，方法对应的论文引用见 [`CITATIONS.zh-CN.md`](CITATIONS.zh-CN.md)。

项目原创贡献采用 [BSD-3-Clause](LICENSE)，但本仓库包含多种许可证。libuipc Python 绑定、TetGen、GelSight 衍生资产及其他组件适用额外条款；再分发前请阅读 [`THIRD_PARTY_NOTICES.zh-CN.md`](THIRD_PARTY_NOTICES.zh-CN.md)。

欢迎贡献，详见 [`CONTRIBUTING.zh-CN.md`](CONTRIBUTING.zh-CN.md)。安全问题请按 [`SECURITY.zh-CN.md`](SECURITY.zh-CN.md) 报告。
