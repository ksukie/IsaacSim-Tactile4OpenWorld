<p align="right">
  <a href="../../en/getting-started/installation.md">English</a> · <strong>简体中文</strong>
</p>

# 安装指南

本指南用于在 Linux 上安装当前 `active-isaaclab-2.1.1/` 路线。本仓库是在已有 Isaac Lab 环境上扩展功能，不会代替用户安装 Isaac Sim 或 Isaac Lab。

## 1. 检查环境

首次安装建议采用以下组合：

| 组件 | 项目目标 | 说明 |
|---|---|---|
| 操作系统 | 带 Bash 的 Linux | 包装脚本和当前 UIPC 构建流程以 Linux 为主。 |
| Isaac Sim | 4.5.0 | 先独立安装并验证。 |
| Isaac Lab | 2.1.1 | 放在本仓库以外的独立目录。 |
| Python | 3.10 | 使用 Isaac Sim 提供的解释器或已激活的 Isaac Lab 环境。 |
| GPU 与驱动 | 与 Isaac Sim 兼容的 NVIDIA RTX GPU 和驱动 | Python 能导入不代表 GPU 兼容。 |
| UIPC 构建工具 | CMake 3.26+、CUDA Toolkit、C++20 编译器、vcpkg | 编译仓库内 libuipc 时需要。 |

Isaac Lab 2.1.1 上游指南记录了在 Ubuntu 22.04 上搭配 Isaac Sim 4.5 的测试组合。仓库中也保留了 Ubuntu 24.04 的历史迁移记录，但它不是公开兼容性基线。选择驱动和系统依赖前，请阅读官方 [Isaac Sim 4.5 系统要求](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html)和 [Isaac Lab 2.1.1 二进制安装指南](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/setup/installation/binaries_installation.html)。

> [!NOTE]
> Isaac Sim 4.5 上游支持原生 Windows，但本仓库尚未提供经过验证的 PowerShell 安装流程，当前包装脚本和 libuipc 构建也以 Bash/Linux 为主。请使用 Linux 完成本指南中的流程。

## 2. 安装 Isaac Sim 与 Isaac Lab

按 Isaac Lab 上游指南完成：

1. 安装 Isaac Sim 4.5，并确认能够独立启动；
2. 克隆 Isaac Lab 并切换到 `v2.1.1`；
3. 按上游方法将 Isaac Lab 连接到 Isaac Sim；
4. 创建或激活 Isaac Lab Python 环境。

完成后，设置 Isaac Lab 根目录的绝对路径并检查文件。二进制安装通常应包含 `_isaac_sim/python.sh`：

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab

test -f "$ISAACLAB_PATH/isaaclab.sh"
test -f "$ISAACLAB_PATH/_isaac_sim/python.sh"
```

如果通过 Python 包安装 Isaac Sim，`_isaac_sim/python.sh` 可能不存在；使用 `run.sh` 前应先激活对应环境。包装脚本能够识别已激活的 Conda 环境或已安装的 `isaacsim-rl` 包。

## 3. 准备 UIPC 工具链

主要触觉基准会编译仓库内的 libuipc 源码。本地构建指南声明需要 CMake 3.26+、CUDA 12.4+、vcpkg 和合适的编译器；具体上游要求见仓库内的 [libuipc Linux 构建说明](../../../active-isaaclab-2.1.1/packages/uipc/libuipc/docs/build_install/linux.md)。

让 CMake 能够找到 vcpkg：

```bash
export VCPKG_ROOT=/absolute/path/to/vcpkg
export CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

cmake --version
nvcc --version
test -f "$CMAKE_TOOLCHAIN_FILE"
```

`CMAKE_CUDA_ARCHITECTURES` 是可选项。只有明确目标 GPU 对应的 CMake 架构编号时才设置：

```bash
export CMAKE_CUDA_ARCHITECTURES=<your-cmake-architecture-number>
```

UIPC 构建可能通过 vcpkg 下载 C++ 依赖，耗时会明显长于纯 Python 包安装。

## 4. 获取项目

```bash
git clone <repository-url> IsaacSim-Tactile4OpenWorld
cd IsaacSim-Tactile4OpenWorld/active-isaaclab-2.1.1
```

请将 `<repository-url>` 替换为发布后的 Git 地址。项目仓库与 Isaac Lab 应放在不同目录。

## 5. 安装项目包

安装包含 UIPC 的完整当前路线：

```bash
./run.sh --install all
```

该命令会向 Isaac Lab/Isaac Sim Python 环境中安装四个可编辑包：

| 包 | 用途 |
|---|---|
| `openworldtactile` | 传感器接口与触觉仿真方法 |
| `openworldtactile_assets` | 机器人、传感器和 USD 资产配置 |
| `openworldtactile_tasks` | 已注册的 Isaac Lab 环境与智能体配置 |
| `openworldtactile_uipc` 和 `uipc` | Isaac Lab 集成与编译后的 libuipc 绑定 |

不带 `all` 的 `./run.sh --install` 会跳过 UIPC 构建，只适用于不导入 UIPC 的代码；它无法满足触觉基准快速开始和 UIPC 任务变体的要求。

## 6. 验证安装

```bash
./run.sh --python -c "import openworldtactile; import openworldtactile_assets; import openworldtactile_uipc; import uipc; print('OpenWorldTactile imports: OK')"
```

然后运行一项依赖较少的单元测试：

```bash
./run.sh --python -m unittest discover \
  -s experiments/tactile-bench \
  -p "test_membrane_local_frame.py" -v
```

导入成功只说明 Python 能找到各个包和编译绑定，并未验证 GPU 接触仿真。请继续按[快速开始](quick-start.md)完成仿真级检查。

## 安装结果判断

| 结果 | 含义 | 下一步 |
|---|---|---|
| 所有导入成功 | 可编辑包和 UIPC 绑定可用 | 继续快速开始 |
| 无法导入 `uipc` | 原生构建未安装到当前 Python 环境 | 检查 vcpkg/CMake/CUDA 和当前解释器 |
| 无法导入 `isaaclab` | `ISAACLAB_PATH` 或激活环境错误 | 重新检查 Isaac Lab 上游安装 |
| 解析 vcpkg 包时停止 | 工具链或联网依赖步骤失败 | 见[故障排查](../help/troubleshooting.md#uipc-构建失败) |

除非目标机的操作系统、Python ABI、CUDA 工具链、GPU 目标和依赖库完全匹配，否则不要从另一台机器复制编译后的 UIPC 模块。

## 下一步

继续阅读[快速开始](quick-start.md)。版本限制与验证范围见[兼容性](../reference/compatibility.md)。
