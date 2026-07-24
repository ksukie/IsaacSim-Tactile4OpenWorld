<p align="right">
  <a href="../../en/guides/data-and-outputs.md">English</a> · <strong>简体中文</strong>
</p>

# 数据与输出

OpenWorldTactile 实验会将结果写入各脚本 `--output_dir` 指定的目录。请始终显式设置该参数。许多默认值指向 `/tmp`，该目录可能被操作系统清理，不适合长期保存数据集。

## 数据约定

除非生成脚本的 `metadata.json` 另有说明：

- 世界坐标和局部坐标位置以米保存；
- 时序数组的第一维表示帧/时间（`T`）；
- 四元数遵循生成数据所用 Isaac Lab API 的约定；
- `fxyz` 的三个通道顺序为 `[fx, fy, fz]`；
- 触觉基准将 `fx` 映射到局部 Y 剪切、`fy` 映射到局部 Z 剪切、`fz` 映射到局部 X 法向压力；
- `sim_constitutive_force` 与 TU 是仿真量值单位，不是经过标定的牛顿。

解释数组前应先阅读 `metadata.json`。不同历史脚本不一定使用相同模式或坐标契约。

## V1 输出契约

`OpenWorldTactile_v1.py` 完成后会写入：

| 路径 | 内容 |
|---|---|
| `fxyz.npy` | float32 触觉场，形状 `[T, H, W, 3]`；默认 `[T, 300, 300, 3]` |
| `metadata.json` | 单位、通道顺序、几何、材料、轨迹和逐帧统计 |
| `preview_force.png` | 最后一个保存力场的可视化 |
| `preview_sequence.mp4` | 视频写入器可用时生成的预览序列 |
| `preview_frames/*.png` | 按 `--preview_every` 间隔保存的单帧预览 |

禁用 pickle 后安全加载：

```python
import json
from pathlib import Path

import numpy as np

run_dir = Path("outputs/v1-smoke")
fxyz = np.load(run_dir / "fxyz.npy", allow_pickle=False)
metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

assert fxyz.ndim == 4 and fxyz.shape[-1] == 3
assert np.isfinite(fxyz).all()
print(fxyz.shape, metadata["force_units"], metadata["channel_order"])
```

`--save_every` 会改变保存帧数，但不会改变仿真推进；`--preview_every` 只改变预览频率。

## V6.2 输出分组

V6.2 的运行契约更大，主要分组包括：

| 分组 | 代表文件 | 含义 |
|---|---|---|
| 触觉估计 | `surface_displacement_pad_local.npy`、`force_pad_local.npy`、`tactile_force_channels.npy` | Pad 局部形变与 TU 触觉估计 |
| UIPC 反力 | `uipc_reaction_force_w.npy`、`uipc_reaction_torque_w.npy` | 世界坐标下的原始 UIPC 时间平均反力/矩 |
| 实际耦合 | `applied_uipc_force_w.npy`、`applied_uipc_force_substeps_w.npy` | 经松弛和限幅后施加给 PhysX 的反力 |
| 接触诊断 | `contact_active.npy`、`minimum_signed_gap_mm.npy`、`uipc_reaction_vertex_count.npy` | 接触状态与穿透诊断 |
| 运动 | `object_pose_w.npy`、`object_velocity_w.npy`、`pad_pose_w.npy`、`gripper_opening_mm.npy` | 物体、Pad 和夹爪历史 |
| 计时 | `frame_wall_time_sec.npy`、`uipc_step_time_sec.npy`、`uipc_substep_time_sec.npy` | 墙钟与求解器耗时 |
| 运行描述 | `metadata.json`，以及可选的 `error.json`/`uipc_timeout.json` | 参数、结构、终止原因与错误 |

子步数组形状为 `[T, S, ...]`，`S` 通常为 8。比较原始 UIPC 反力与实际施加给 PhysX 的限幅力时，必须考虑接触锥投影、反馈松弛和力上限。

生成离线力场/视频和验收结果：

```bash
./run.sh --python experiments/tactile-bench/render_tactile_field_offline.py \
  --input_dir "$PWD/outputs/v62-grasp" \
  --video_fps 15

./run.sh --python experiments/tactile-bench/validate_v6_2_once.py \
  --input_dir "$PWD/outputs/v62-grasp" \
  --fail_on_failure
```

完整流程见[运行实验](experiments.md#v62-piper-闭环抓取)。

## 检查 HDF5 文件

HDF5 记录器分布在面向特定场景的操作脚本中，仓库没有统一的全局 HDF5 模式。选择数据集前应先检查文件。

打印所有组、数据集、形状和类型：

```bash
./run.sh --python -c "import h5py; p='path/to/episode.hdf5'; f=h5py.File(p,'r'); f.visititems(lambda n,o: print(n, getattr(o,'shape','group'), getattr(o,'dtype',''))); f.close()"
```

内置工具可识别的常见路径包括：

```text
observations/images/<stream>
observations/tactile/fxyz
observations/tactile1
observations/tactile1_fxyz_float32
observations/tactile1_height_map_float32
observations/tactile1_deformation_float32
```

这些只是候选路径，不表示每个 episode 都包含全部数据集。

## 查看或导出相机流

并排播放 `observations/images` 下的所有流：

```bash
./run.sh --python tools/data/view_hdf5_images.py path/to/episode.hdf5
```

选择数据流和播放速度：

```bash
./run.sh --python tools/data/view_hdf5_images.py path/to/episode.hdf5 \
  --streams top wrist \
  --fps 15 \
  --every 2
```

不打开窗口，直接导出：

```bash
./run.sh --python tools/data/view_hdf5_images.py path/to/episode.hdf5 \
  --export-dir outputs/decoded-camera \
  --no-show
```

专用导出器提供相同的非交互操作：

```bash
./run.sh --python tools/data/export_hdf5_images.py \
  path/to/episode.hdf5 outputs/decoded-camera \
  --every 1
```

## 导出触觉数据集

自动检测已知触觉数据集并渲染为图像：

```bash
./run.sh --python tools/data/export_hdf5_tactile_images.py \
  path/to/episode.hdf5 outputs/decoded-tactile
```

选择数据集并对帧进行降采样：

```bash
./run.sh --python tools/data/export_hdf5_tactile_images.py \
  path/to/episode.hdf5 outputs/decoded-tactile \
  --datasets observations/tactile/fxyz \
  --every 5 \
  --format png \
  --scale 4
```

对于三通道浮点力场，导出器会写入组合 RGB 可视化，以及 `fx`、`fy`、`fz` 各通道的伪彩色图。归一化只用于观察，导出颜色不代表物理量值。

## 数据集质量检查

将运行结果用于分析或训练前，至少检查：

- 数组存在，维数与通道顺序符合文档；
- 所有浮点数组均为有限值；
- 同步观测的时间维长度一致；
- 需要时同时包含接触与空载帧；
- 已记录单位、坐标系、仿真器和标定字段；
- 不存在 `error.json`、超时文件或失败验收，或已明确处理这些情况；
- 保留源码修订、命令、随机种子和资产/契约标识。

原始数值数组与元数据应和可视化一起保存。预览图不能替代源力场。

## 存储建议

- 每次运行写入新目录。
- UIPC 工作目录与长期保存的输出目录分开。
- 不要向源码仓库提交生成的数据集、视频、缓存、检查点或 UIPC 构建目录。
- 重启或清理前，将成功结果移出 `/tmp`。
- 在机器间传输大型数据集时使用校验和。
- 分发数据集时保留所有第三方输入资产的许可证与来源。
