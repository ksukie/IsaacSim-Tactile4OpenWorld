<p align="right">
  <a href="../../en/help/troubleshooting.md">English</a> · <strong>简体中文</strong>
</p>

# 故障排查

请从最小失败层级开始检查。在完成导入、轻量测试和 V1 检查前，不要直接调试完整 V6.2 运行。

## 收集基础诊断信息

在 `active-isaaclab-2.1.1/` 中运行：

```bash
printf 'ISAACLAB_PATH=%s\n' "$ISAACLAB_PATH"
nvidia-smi
nvcc --version
cmake --version
git --version

./run.sh --python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.platform())"
./run.sh --python -c "import isaaclab, openworldtactile, openworldtactile_assets, openworldtactile_uipc, uipc; print('uipc=', uipc.__file__)"
```

保存完整命令与回溯。原生崩溃时，请附上最后一段仿真器/UIPC 日志，并说明同一命令在无界面模式下是否也失败。

## `ISAACLAB_PATH` 或 Python 错误

常见现象：

- `Unable to find the Isaac Sim directory`；
- `Unable to find any Python executable`；
- `ModuleNotFoundError: isaaclab`；
- `run.sh` 输出了意外的 Python 路径。

检查：

```bash
test -d "$ISAACLAB_PATH"
test -f "$ISAACLAB_PATH/isaaclab.sh"
ls -l "$ISAACLAB_PATH/_isaac_sim/python.sh"
./run.sh --python -c "import sys; print(sys.executable)"
```

通过 pip 安装 Isaac Sim 时，请先激活包含 `isaacsim-rl` 的环境。二进制安装时，请按 Isaac Lab 指南创建 `_isaac_sim` 连接。不要将 `ISAACLAB_PATH` 指向本 OpenWorldTactile 仓库。

## 项目包导入失败

重新安装到 `run.sh` 输出的解释器：

```bash
./run.sh --install all
./run.sh --python -m pip show \
  openworldtactile openworldtactile-assets openworldtactile-tasks openworldtactile-uipc
```

pip 可能用连字符规范化包显示名称。如果只有一个包缺失，应检查对应安装错误，不要手工把源码目录加入 `PYTHONPATH`。

如果 `openworldtactile_tasks` 在导入可选 UIPC 任务时失败，请先验证 `uipc`。UIPC 任务注册带保护逻辑，其他任务仍可能正常出现。

## UIPC 构建失败

### CMake 找不到 vcpkg

```bash
export VCPKG_ROOT=/absolute/path/to/vcpkg
export CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
test -f "$CMAKE_TOOLCHAIN_FILE"
```

内置 CMake 逻辑需要有效工具链。vcpkg 目录路径与 `vcpkg.cmake` 文件路径不是同一个值。

### CMake 版本过低

内置 libuipc `CMakeLists.txt` 要求 CMake 3.26 或更高：

```bash
cmake --version
```

确保当前 Shell 找到的 `cmake` 正是构建时使用的版本。

### CUDA 编译器或架构错误

同时检查系统 Toolkit 与驱动：

```bash
which nvcc
nvcc --version
nvidia-smi
```

如果设置了 `CMAKE_CUDA_ARCHITECTURES`，确认它是目标 GPU 对应的有效 CMake 架构编号。可以取消错误覆盖，让 CMake 使用本机选择，或设置正确值。Isaac Sim 运行时 CUDA 库与编译 libuipc 使用的系统 Toolkit 属于不同层级，两者都必须与驱动兼容。

### `uipc` 编译成功但无法导入

```bash
./run.sh --python -c "import sys; print(sys.executable); import uipc; print(uipc.__file__)"
```

常见原因包括使用了不同 Python 编译、C++ 运行时不兼容、缺少共享库，或复用了其他机器的二进制。请在目标环境中通过 `./run.sh --install all` 重建。不要用任意 `LD_LIBRARY_PATH` 修改掩盖 ABI 错误。

### 构建很慢或内存不足

libuipc 会编译 CUDA/C++ 依赖，资源开销较大。关闭其他 GPU/编译工作负载，确认磁盘和内存充足。当前包安装脚本内部选择并行编译，包装器没有已记录的并行度参数。反馈问题时请保留失败的编译器命令行。

## Isaac Sim 无法打开或首次启动卡住

- 先运行外部安装中的独立 Isaac Sim 示例。
- 确认 GPU 与驱动满足 Isaac Sim 4.5 要求。
- 首次启动可能需要构建着色器/扩展缓存。
- 无显示服务器时添加 `--headless`。
- 相机任务即使无界面运行也需要 `--enable_cameras`。
- 检查 Isaac/Omniverse 缓存所需磁盘空间。

如果最小上游 Isaac Sim/Isaac Lab 示例也失败，应先解决外部环境，再调试 OpenWorldTactile。

## V1 失败或输出异常

### 输出目录为空

确认没有使用 `--no_save` 或 `--loop_forever`。`--loop_forever` 会主动关闭保存。请设置显式可写的 `--output_dir`，并使用独立可写的 `--workspace_dir`。

### 缺少 `fxyz.npy`

阅读终端尾部并查找异常。V1 没有统一写入 `error.json` 的失败契约，因此崩溃后可能完全没有文件。先重新运行快速开始的低分辨率用例，再尝试默认分辨率。

### 力场全零或包含非有限值

```bash
./run.sh --python -c "import numpy as np; a=np.load('outputs/v1-smoke/fxyz.npy', allow_pickle=False); print(a.shape, np.isfinite(a).all(), float(np.abs(a).max()))"
```

确认轨迹包含压入、输出包含记录帧，且元数据与命令一致。出现非有限值代表运行失败；请保留工作目录和命令用于诊断，不要使用该数据。

### 数组存在但缺少预览视频

OpenCV 可能没有可用 MP4 编码器。`fxyz.npy`、`metadata.json` 和 PNG 帧仍是数值/源输出。可直接使用 PNG，或安装与当前 OpenCV 构建兼容的编码器。

## V6.2 契约与运行失败

### “Frozen 7f contract is incomplete”

`--contract_dir` 至少必须包含 V6.2 检查的 `vertex_area.npy`、`front_surface_mask.npy`、`rest_surface_pad_local.npy` 和 `verdict.json` 等文件。请使用完整[契约流程](../guides/experiments.md#所需形变契约)，不要创建空占位文件。

### “Frozen 7f deformation contract did not pass”

打开 `verdict.json` 检查失败条件。使用文档规定的 22 × 26 网格和相互独立的工作目录重新生成 V5.7d/7e 源运行。不要通过编辑 verdict 绕过验收。

### UIPC 子步值被拒绝

V6.2 要求 `--uipc_substeps_per_record >= 8`。该最小值属于接触/耦合设计，不是性能建议。

### 出现 `uipc_timeout.json`

原生 UIPC 子步超过 `--uipc_substep_timeout_sec`。检查 JSON 中的阶段/帧/子步、`uipc_substep_time_sec.npy` 和终端 `[V62_SLOW_FRAME]` 信息。超时代表运行失败或不完整。修改求解器/物理参数前，先用默认参数和新工作目录测试。

### 出现 `error.json`

阅读其中的回溯与 `termination_reason`。可保留部分数组用于诊断，但不能把它们作为成功完整运行交给验收流程。

### 进程完成但验收失败

验收器会报告失败的物理或结构条件。检查是否使用 `--max_formal_frames` 截断运行、修改了耦合/驱动/物体参数，或缺少可选检查所需的离线渲染输出。进程结束不等于场景通过。

## 找不到任务 ID

单独检查注册：

```bash
./run.sh --python -c "import gymnasium as gym; import openworldtactile_tasks; print([x for x in gym.registry if x.startswith('OpenWorldTactile-')])"
```

如果这里存在任务，但 Isaac Lab train/random/play 脚本中不存在，说明启动器没有导入 `openworldtactile_tasks`。按[任务与训练](../guides/tasks-and-training.md#连接-isaac-lab-启动器)说明，在应用启动后的外部项目占位符处添加导入。

如果只缺少 UIPC RGB 任务，说明其受保护导入失败。只有在 Isaac 应用启动后，才直接导入该模块查看底层回溯。

## 相机任务错误或显存不足

- 添加 `--enable_cameras`。
- 从 `--num_envs 1` 开始。
- 确认所选任务确实提供请求的相机/触觉输出。
- 修改图像/传感器配置前先减少环境数。
- 反馈 OOM 时记录显存与命令。

降低传感器分辨率会改变观测空间；除非同步更新模型/配置，否则原有检查点会失效。

## HDF5 查看/导出错误

### 找不到组或数据集

先检查文件：

```bash
./run.sh --python -c "import h5py; f=h5py.File('path/to/file.hdf5'); f.visititems(lambda n,o: print(n, getattr(o,'shape','group'))); f.close()"
```

然后传入实际的 `--image-group`、`--streams` 或 `--datasets`。历史 HDF5 文件并不共享统一模式。

### JPEG 解码失败

文件可能使用了没有预期 `compress_len` 的填充压缩行、包含截断帧，或 uint8 数据并非 JPEG。修改导出器前请检查类型、形状和记录器元数据。

### OpenCV 窗口无法打开

在无界面系统中使用 `--export-dir ... --no-show`。

## 历史 SDK 或资产错误

归档路线有意排除了供应商相机 SDK 和部分外部资产。`OWT_SDK_ROOT`/`OWT_ASSET_ROOT` 只能指向合法取得且兼容的副本。不能通过把来源不明的 DLL/SO 下载进公开仓库来修复缺失 SDK。

## 提交可复现问题

请包含：

- 仓库修订与路线（`active` 或 `archive`）；
- 操作系统、GPU、驱动、CUDA Toolkit、Python、Isaac Sim、Isaac Lab 和 libuipc 版本；
- 完整命令和移除敏感信息后的环境变量；
- 最小失败示例，以及上游 Isaac 示例是否可用；
- 完整回溯/日志尾部，以及生成的 `error.json`、超时、元数据或 verdict 文件；
- 预期行为与实际行为。

不要在公开 issue 中附加专有 SDK 二进制、凭据或大型私有数据集。
