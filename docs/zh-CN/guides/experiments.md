<p align="right">
  <a href="../../en/guides/experiments.md">English</a> · <strong>简体中文</strong>
</p>

# 运行实验

主线源码树同时包含推荐入口、验证探针和历史研究阶段。请使用下面明确记录的入口，不要只根据类似版本号的文件名选择脚本。

以下命令均假定已经执行：

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab
cd /absolute/path/to/IsaacSim-Tactile4OpenWorld/active-isaaclab-2.1.1
mkdir -p outputs
```

## 推荐入口

| 入口 | 用途 | 前置条件 | 输出 |
|---|---|---|---|
| `OpenWorldTactile_v1.py` | 首次 UIPC 柔性膜/接触运行与力场生成 | 完整 UIPC 安装 | `fxyz.npy`、元数据、图像/视频预览 |
| `OpenWorldTactile_v6_2_grasp.py` | 当前 Piper 闭环抓取研究流程 | UIPC 和通过验收的 V5.7f 形变契约 | 详细 PhysX/UIPC 耦合数组和验收元数据 |
| 已注册的 `openworldtactile_tasks` 环境 | 强化学习 | 匹配的 Isaac Lab 训练脚本与智能体包 | 框架对应的日志/检查点 |
| `tools/data/*.py` | 检查生成的 HDF5 episode | `h5py`、OpenCV、NumPy | 解码后的图像目录 |

V2 至 V6.1b 以及大量 `v5_new_*` 脚本记录了当前流程的研发过程。除非指南明确要求，否则应将它们视为研究探针。

## V1 固定柔性膜基准

V1 完整位于当前主线中，不依赖预先生成的契约，因此最适合作为第一个仿真实验。

### 完整默认运行

```bash
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v1.py \
  --headless \
  --shape dots \
  --indent_depth_mm 0.8 \
  --rub_distance_mm 3.0 \
  --output_dir "$PWD/outputs/v1-dots" \
  --workspace_dir "$PWD/outputs/v1-dots-workspace"
```

可选压头形状包括 `sphere`、`cylinder`、`dots`、`cross_lines`、`wave1` 和 `random`。每次独立 UIPC 运行应使用新的 `--workspace_dir`；不要让多个进程共用一个工作目录。

### 无界面与交互模式

- 添加 `--headless` 进行无窗口运行。
- 去掉 `--headless` 并添加 `--render_viewport` 可观察运动。
- 使用 `--render_every N` 每 N 个仿真步渲染一次。
- 使用 `--no_save` 只观察而不写输出。
- `--loop_forever` 会重复轨迹，并自动关闭保存。

保存的触觉张量形状为 `[T, 300, 300, 3]`。三个通道依次表示局部切向 Y、局部切向 Z 和局部法向 X，对外记为 `fx`、`fy`、`fz`。单位是 `sim_constitutive_force`，不是牛顿。详见[数据与输出](data-and-outputs.md#v1-输出契约)。

## V6.2 Piper 闭环抓取

V6.2 将 PhysX 控制的 Piper 和自由圆柱与一张 Pad 局部坐标系下的 UIPC 柔性膜耦合。它是进阶复现实验，不是单命令入门演示。

### 所需形变契约

如果 `--contract_dir` 中没有通过验收的 V5.7f 契约，V6.2 会拒绝启动。构建该契约需要一次 V5.7d 无接触运动记录和五次相互独立的 V5.7e 压入记录。

设置输出根目录：

```bash
export OWT_CONTRACT_ROOT="$PWD/outputs/v62-contract"
mkdir -p "$OWT_CONTRACT_ROOT"
```

生成刚体运动记录：

```bash
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v5_new_7d_backface_follow.py \
  --headless \
  --no_save_camera_rgb \
  --fail_on_verdict_fail \
  --output_dir "$OWT_CONTRACT_ROOT/rigid" \
  --workspace_dir "$OWT_CONTRACT_ROOT/rigid-workspace"
```

生成五次独立法向压入记录：

```bash
for repeat in 0 1 2 3 4; do
  ./run.sh --python experiments/tactile-bench/OpenWorldTactile_v5_new_7e_indenter_deformation.py \
    --headless \
    --no_save_camera_rgb \
    --membrane_cells_y 22 \
    --membrane_cells_z 26 \
    --indentation_levels_mm 0,0.2,0.5 \
    --fail_on_verdict_fail \
    --output_dir "$OWT_CONTRACT_ROOT/normal-$repeat" \
    --workspace_dir "$OWT_CONTRACT_ROOT/normal-$repeat-workspace"
done
```

构建并验证冻结契约：

```bash
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v5_new_7f_deformation_contract_probe.py \
  --rigid_input_dir "$OWT_CONTRACT_ROOT/rigid" \
  --normal_input_dir "$OWT_CONTRACT_ROOT/normal-0" \
  --repeat_input_dir "$OWT_CONTRACT_ROOT/normal-1" \
  --repeat_input_dir "$OWT_CONTRACT_ROOT/normal-2" \
  --repeat_input_dir "$OWT_CONTRACT_ROOT/normal-3" \
  --repeat_input_dir "$OWT_CONTRACT_ROOT/normal-4" \
  --output_dir "$OWT_CONTRACT_ROOT/verified" \
  --fail_on_verdict_fail
```

确认 `verified/verdict.json` 中的 `deformation_contract_passed` 为 `true`。V6.2 可用的契约目录至少必须包含 `vertex_area.npy`、`front_surface_mask.npy`、`rest_surface_pad_local.npy` 和通过验收的 `verdict.json`；V6.2 会在启动场景前检查这四个文件。

### 运行 V6.2

```bash
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v6_2_grasp.py \
  --headless \
  --contract_dir "$OWT_CONTRACT_ROOT/verified" \
  --workspace_dir "$PWD/outputs/v62-grasp-workspace" \
  --output_dir "$PWD/outputs/v62-grasp" \
  --sim_hz 60 \
  --uipc_substeps_per_record 8 \
  --uipc_feedback_relaxation 1.0 \
  --uipc_feedback_force_limit_n 0.25 \
  --slow_frame_threshold_sec 0.5 \
  --log_every 10
```

`--uipc_substeps_per_record` 必须保持在 8 或更高，脚本会拒绝更小的值。完整默认运动包含数百个记录帧。如果只想诊断初始化，可添加 `--max_formal_frames 20`，但不能把截断结果解释为抓取成功。

交互运行时去掉 `--headless` 并添加 `--render_viewport`。需要可复现数据时不要使用 `--loop_forever`。

### 渲染并验收 V6.2 输出

```bash
./run.sh --python experiments/tactile-bench/render_tactile_field_offline.py \
  --input_dir "$PWD/outputs/v62-grasp" \
  --video_fps 15

./run.sh --python experiments/tactile-bench/validate_v6_2_once.py \
  --input_dir "$PWD/outputs/v62-grasp" \
  --penetration_tolerance_mm 0.15 \
  --precontact_deformation_tolerance_mm 0.25 \
  --minimum_object_lift_mm 20 \
  --fail_on_failure
```

验收器会检查结构契约、耦合递推、接触响应、穿透容差、抬升、释放和离线力场一致性。持续维护的输出清单和解释规则见[数据与输出](data-and-outputs.md#v62-输出分组)；其他诊断数组以生成脚本为准。

## 其他实验分组

| 目录 | 内容 | 稳定性 |
|---|---|---|
| `experiments/simulation-prototypes/` | Taxim、FOTS、FEM、力分量和传感器集成探针 | 探索性；运行前检查依赖 |
| `experiments/manipulation/` | 拾取、放置、插孔、摩擦和 HDF5 记录脚本 | 面向特定场景的研究代码 |
| `experiments/benchmarks/` | 刚体/UIPC/触觉性能对比 | 基准工具；结果依赖硬件与设置 |
| `experiments/tactile-bench/` | V1–V6.2 UIPC 触觉演进与离线验收器 | V1 与 V6.2 是文档入口；其余为里程碑/探针 |

可以通过[生成型入口清单](../../internal/ENTRYPOINT_MATRIX.md)查找脚本并检查静态导入。清单中的“可直接运行”仅表示发现了主守卫和参数解析器，不是运行保证。

## 可复现运行清单

对需要比较或发表的结果，请记录：

- 仓库修订版本与实验文件名；
- Isaac Sim、Isaac Lab、Python、CUDA Toolkit、驱动、GPU 和 libuipc 版本；
- 完整命令与所有非默认参数；
- 随机种子、输入资产、契约哈希和标定文件；
- 输出元数据与验收结论；
- 是否无界面运行、是否渲染、是否截断或续跑。

除非应用了有记录的标定，否则不要把 `sim_constitutive_force` 或 TU 力场标记为牛顿。
