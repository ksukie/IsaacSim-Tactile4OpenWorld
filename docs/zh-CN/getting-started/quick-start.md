<p align="right">
  <a href="../../en/getting-started/quick-start.md">English</a> · <strong>简体中文</strong>
</p>

# 快速开始

本流程用于验证已安装的当前主线，并生成一份小规模触觉力场结果。开始前请先完成[安装指南](installation.md)，包括 `./run.sh --install all`。

以下命令均从主线目录开始：

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab
cd /absolute/path/to/IsaacSim-Tactile4OpenWorld/active-isaaclab-2.1.1
```

## 1. 验证当前 Python 环境

```bash
./run.sh --python -c "import sys; print(sys.executable)"
./run.sh --python -c "import isaaclab, openworldtactile, openworldtactile_assets, openworldtactile_uipc, uipc; print('imports: OK')"
```

第一条命令应输出当前 Isaac Lab/Isaac Sim 环境中的 Python 解释器。如果显示了意外的解释器，请先修正环境再运行仿真。

## 2. 运行轻量算法检查

以下测试不会启动仿真器：

```bash
./run.sh --python -m unittest discover \
  -s experiments/tactile-bench \
  -p "test_membrane_local_frame.py" -v

./run.sh --python -m unittest discover \
  -s experiments/tactile-bench \
  -p "test_tu_tactile_field.py" -v

./run.sh --python -m unittest discover \
  -s experiments/tactile-bench \
  -p "test_v5_new_7g_deformation_force_estimator.py" -v
```

这些测试检查坐标转换、触觉场构建和离线形变估计器，但不能证明 Isaac Sim 或 UIPC 已能在 GPU 上正常推进。

## 3. 运行 V1 触觉冒烟测试

下面的低分辨率用例用于验证环境。它让球体压入并离开一张 UIPC 柔性膜，保存 `300 x 300 x 3` 触觉场：

```bash
mkdir -p outputs

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

首次启动 Isaac Sim 时可能需要构建着色器和扩展缓存；UIPC 第一次求解也可能较慢。

## 4. 检查结果

预期文件：

```text
outputs/v1-smoke/
├── fxyz.npy
├── metadata.json
├── preview_force.png
├── preview_sequence.mp4
└── preview_frames/
```

验证数组与元数据：

```bash
./run.sh --python -c "import json, numpy as np; p='outputs/v1-smoke'; a=np.load(f'{p}/fxyz.npy'); m=json.load(open(f'{p}/metadata.json')); assert a.ndim == 4 and a.shape[1:] == (300, 300, 3); assert np.isfinite(a).all(); print('shape=', a.shape, 'units=', m['force_units'], 'max_abs=', float(np.abs(a).max()))"
```

预期单位标签是 `sim_constitutive_force`，这些值不是经过标定的牛顿。非零最大值表示运行产生了响应，但仅凭这一点不能证明物理精度。

如果出现 `error.json`，或缺少任何预期文件，请查看[故障排查](../help/troubleshooting.md)。

## 5. 可选：观察仿真

去掉 `--headless` 并启用视口渲染。`--no_save` 可避免观察时写入输出：

```bash
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v1.py \
  --shape sphere \
  --indent_depth_mm 0.6 \
  --rub_distance_mm 0.0 \
  --front_segments_y 24 \
  --front_segments_z 30 \
  --thickness_segments 3 \
  --tet_edge_length_r 0.05 \
  --render_viewport \
  --render_every 5 \
  --loop_forever \
  --no_save
```

关闭 Isaac Sim 窗口或在终端按 `Ctrl+C` 即可结束循环。

## 下一步

| 目标 | 指南 |
|---|---|
| 使用 V1 完整分辨率设置 | [实验指南：V1](../guides/experiments.md#v1-固定柔性膜基准) |
| 运行进阶 V6.2 Piper 抓取 | [实验指南：V6.2](../guides/experiments.md#v62-piper-闭环抓取) |
| 训练已注册任务 | [任务与训练](../guides/tasks-and-training.md) |
| 检查 NumPy、视频或 HDF5 结果 | [数据与输出](../guides/data-and-outputs.md) |
| 理解力轴与包调用流程 | [架构](../reference/architecture.md) |
