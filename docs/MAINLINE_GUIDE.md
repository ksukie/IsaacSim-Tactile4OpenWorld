# Isaac Lab 2.1.1 当前主线指南

## 定位

[`active-isaaclab-2.1/`](../active-isaaclab-2.1/) 是当前开发主线，保存 OpenWorldTactile、UIPC 与 AgileX/Piper 相关项目内容。它需要外部准备匹配的 Isaac Lab 2.1.1 与 Isaac Sim，不是一套完整的官方 Isaac Lab 仓库。

## 目录导航

| 位置 | 内容 |
|---|---|
| [`packages/core/`](../active-isaaclab-2.1/packages/core/) | OpenWorldTactile 核心扩展 |
| [`packages/assets/`](../active-isaaclab-2.1/packages/assets/) | 触觉传感器、机器人和场景资产 |
| [`packages/tasks/`](../active-isaaclab-2.1/packages/tasks/) | OpenWorldTactile 任务与环境 |
| [`packages/uipc/`](../active-isaaclab-2.1/packages/uipc/) | libuipc 源码及 Isaac Lab 集成 |
| [`experiments/benchmarks/`](../active-isaaclab-2.1/experiments/benchmarks/) | 性能对比与基准实验 |
| [`experiments/manipulation/`](../active-isaaclab-2.1/experiments/manipulation/) | 抓取、插孔与其他操作实验 |
| [`experiments/tactile-bench/`](../active-isaaclab-2.1/experiments/tactile-bench/) | OpenWorldTactile UIPC V1–V6.2 实验演进 |
| [`tools/data/`](../active-isaaclab-2.1/tools/data/) | HDF5 查看与导出工具 |
| [`vendor/agilex-piper/`](../active-isaaclab-2.1/vendor/agilex-piper/) | AgileX/Piper 外部集成内容 |
| [`run.sh`](../active-isaaclab-2.1/run.sh) | Bash 环境、安装、Python 与仿真包装入口 |

## 推荐阅读顺序

1. 先看 [OpenWorldTactileBench 原 README](../active-isaaclab-2.1/experiments/tactile-bench/README.md)，理解数据目标和历史路线。
2. 再看 [OpenWorldTactile 版本索引](OWTBENCH_VERSION_INDEX.md)，区分 V1–V6.2 的阶段。
3. 当前默认参考入口是 [`OpenWorldTactile_v6_2_grasp.py`](../active-isaaclab-2.1/experiments/tactile-bench/OpenWorldTactile_v6_2_grasp.py)。
4. 需要查找其他演示、测试或工具时，使用 [脚本入口矩阵](ENTRYPOINT_MATRIX.md)，不要只按文件名猜用途。

V6.2 会沿用 V5.9、V5.7g 和其他同目录辅助模块，因此不能只复制一个脚本作为独立程序。

## 环境边界

现有项目说明给出的参考组合是 Ubuntu 24.04、Python 3.10、Isaac Sim 4.5、Isaac Lab 2.1.1 与 libuipc 0.9.0。该组合是静态参考，不是本次重构重新验证的兼容性结论。

运行前至少需要确认：

- 已准备完整 Isaac Lab 2.1.1，并设置 `ISAACLAB_PATH`。
- Isaac Sim、Python、CUDA、GPU 驱动和 PyTorch ABI 相互匹配。
- `openworldtactile_uipc`/libuipc 在目标环境重新构建，不复用来源机的缺失构建产物。
- 入口脚本引用的 USD、机器人和传感器资产可解析。

完整缺口见 [依赖缺口与迁移边界](DEPENDENCY_GAPS.md)。

## 命令模板

以下命令来自现有 `run.sh` 的静态接口，只作为目标环境准备完成后的导航模板；本次没有实际运行验证。

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab-2.1.1
cd active-isaaclab-2.1

./run.sh --help
./run.sh --install
./run.sh --install all
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v6_2_grasp.py --help
```

其中普通 `--install` 安装 `core`、`assets` 和 `tasks` 中的 Python 包；`--install all` 还会处理 `uipc`。是否能够完成仍取决于完整 Isaac Lab、编译工具链和 GPU 环境。

## 维护规则

- 新实验优先复用现有 API 和辅助模块，不再复制一整套版本目录。
- 默认入口变更时，只更新导航说明；不要删除历史实验脚本。
- 新增依赖或资产时，在文档中明确“仓库内提供”还是“外部环境提供”。
- 主线不应直接吸收历史归档的全部 `packages/`；确需使用 OpenWorldTactile SDK 时保留清晰、可覆盖的边界。
