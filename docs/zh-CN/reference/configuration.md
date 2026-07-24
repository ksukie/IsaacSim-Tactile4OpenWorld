<p align="right">
  <a href="../../en/reference/configuration.md">English</a> · <strong>简体中文</strong>
</p>

# 配置参考

项目通过环境变量指定外部位置，通过命令行参数控制单次运行设置，目前没有统一的全局项目配置文件。

## 环境变量

| 变量 | 路线 | 用途 | 是否必需 |
|---|---|---|---|
| `ISAACLAB_PATH` | 主线 | 外部 Isaac Lab 2.1.1 绝对根目录 | `run.sh` 必需 |
| `VCPKG_ROOT` | UIPC 构建 | vcpkg 目录，内置 CMake 逻辑也会读取 | 通常需要 |
| `CMAKE_TOOLCHAIN_FILE` | UIPC 构建 | `vcpkg.cmake` 路径 | 除非能发现受支持默认值，否则需要 |
| `CMAKE_CUDA_ARCHITECTURES` | UIPC 构建 | 可选 CMake CUDA 架构列表 | 非必需；明确数值后再设置 |
| `OWT_ASSET_ROOT` | 归档 | 外部历史触觉资产根目录 | 仅受影响脚本需要 |
| `OWT_SDK_ROOT` | 归档/部分原型 | 外部取得的相机 SDK 根目录 | 仅依赖 SDK 的脚本需要 |
| `PYTHON_BIN` | 部分历史脚本 | 覆盖子进程 Python | 可选 |

`run.sh` 会在内部把 `OWT_PATH` 设置为主线目录，用户通常不应自行设置。

Shell 会话示例：

```bash
export ISAACLAB_PATH=/opt/IsaacLab-2.1.1
export VCPKG_ROOT=/opt/vcpkg
export CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
```

请使用绝对路径；路径可能含空格时应给变量加引号。

## `run.sh` 命令

在 `active-isaaclab-2.1.1/` 中运行：

| 命令 | 行为 |
|---|---|
| `./run.sh --install` | 可编辑安装 core、assets、tasks 包 |
| `./run.sh --install all` | 安装上述包并编译/安装 UIPC |
| `./run.sh --python <args>` | 调用所选 Isaac Sim/当前环境 Python |
| `./run.sh --sim <args>` | 启动 Isaac Sim，并把项目包目录作为扩展目录 |
| `./run.sh --test <pytest-args>` | 在已配置的 core、task、tactile-bench 根目录运行 pytest |
| `./run.sh --format` | 运行 pre-commit；必要时可能安装 `pre-commit` |
| `./run.sh --vscode` | 存在时调用 Isaac Lab 的 VS Code 设置工具 |
| `./run.sh --conda [NAME]` | 使用包装脚本的旧辅助方法创建 Python 3.10 Conda 环境 |
| `./run.sh --help` | 打印帮助；当前脚本打印后以状态码 1 退出 |

新安装应优先使用 Isaac Lab 上游环境说明，而不是 `--conda`。包装脚本按以下顺序选择 Python：已激活 Conda 环境、Isaac Lab `_isaac_sim/python.sh`、安装了 `isaacsim-rl` 的系统 Python。

如果目标脚本注册了 Isaac Lab `AppLauncher` 参数，且用户没有指定渲染模式，`run.sh --python` 会添加 `--rendering_mode performance`。

## 常见 Isaac 启动参数

准确参数来自固定版本的 Isaac Lab `AppLauncher`，请查看目标脚本 `--help`。本项目常用：

| 参数 | 用途 |
|---|---|
| `--headless` | 不显示应用窗口运行 |
| `--enable_cameras` | 启用相机传感器，包括无界面任务运行 |
| `--device cuda:0` | 在支持时选择 Isaac 仿真设备 |
| `--rendering_mode performance` | 降低非视觉运行的渲染开销 |

实验自己的 `--render_viewport` 与 AppLauncher 的 headless 参数不是同一选项。

## 触觉基准常见参数

| 参数 | 含义 | 建议 |
|---|---|---|
| `--output_dir PATH` | 长期保存的运行输出 | 始终显式设置；每次运行使用新路径 |
| `--workspace_dir PATH` | UIPC 临时/工作目录 | 与输出分开，每个进程唯一 |
| `--render_viewport` | 在线渲染场景 | 无界面吞吐运行时省略 |
| `--render_every N` | 每 N 步渲染一次 | 增大 N 可减少显示开销 |
| `--no_save` | 在支持脚本中禁用输出 | 只用于观察/调试 |
| `--loop_forever` | 持续重复直到停止 | 不适合可复现的有限数据集 |
| `--log_every N` | 支持脚本中的日志间隔 | 增大 N 可减少终端输出 |
| `--fail_on_verdict_fail` / `--fail_on_failure` | 验收失败时返回失败 | 复现或 CI 运行时使用 |

参数名由各脚本定义。从另一阶段复用参数前请先查看 `--help`。

## V1 控制项

主要类别：

- 几何：`--shape`、`--tool_radius_mm`、柔性膜尺寸；
- 运动：`--indent_depth_mm`、`--rub_distance_mm`、各阶段步数；
- 离散化：`--front_segments_y`、`--front_segments_z`、`--thickness_segments`、`--tet_edge_length_r`；
- 本构估计：法向/剪切刚度与阻尼、摩擦；
- 输出：触觉宽高、保存和预览间隔。

降低网格分辨率适合安装冒烟测试，但会改变数值实验。不能把低分辨率与默认结果当作只有运行耗时不同来比较。

## V6.2 控制项

必需外部路径：

```text
--contract_dir <通过验收的 V5.7f 契约>
```

主要耦合控制项：

- `--uipc_substeps_per_record`（最小为 8）；
- `--uipc_feedback_relaxation`；
- `--uipc_feedback_force_limit_n`；
- 夹爪驱动刚度、阻尼、力上限、速度和目标开度；
- 物体尺寸、质量、摩擦和初始位姿；
- 求解器超时与慢帧报告阈值。

修改这些值会改变物理/耦合场景。请保存完整命令与生成元数据。

## 配置优先级

对多数实验脚本：

1. 显式命令行值覆盖解析器默认值；
2. 环境变量选择外部工具/位置；
3. 模块级资产/配置默认值补充其余字段；
4. 生成元数据记录该脚本实际实现的有效配置子集。

不同历史脚本不保证使用相同名称、默认值或优先级。未经检查源码和输出元数据，不要在版本间直接复制配置字典。
