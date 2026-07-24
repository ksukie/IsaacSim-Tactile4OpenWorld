<p align="right">
  <a href="../../en/guides/tasks-and-training.md">English</a> · <strong>简体中文</strong>
</p>

# 任务与训练

`openworldtactile_tasks` 提供 Gymnasium 注册和智能体配置文件，具体训练实现来自外部 Isaac Lab 2.1.1 仓库。

## 前置条件

1. 完成[完整安装](../getting-started/installation.md)，包括 UIPC。
2. 通过 Isaac Lab 安装任务所需的强化学习库。例如：

   ```bash
   "$ISAACLAB_PATH/isaaclab.sh" -i skrl
   "$ISAACLAB_PATH/isaaclab.sh" -i rl_games
   "$ISAACLAB_PATH/isaaclab.sh" -i rsl_rl
   ```

3. 先使用一个或少量环境。相机和 UIPC 观测的显存与计算开销显著高于特权状态任务。

## 列出已注册环境

读取 Gym 注册表前先导入项目任务包：

```bash
./run.sh --python -c "import gymnasium as gym; import openworldtactile_tasks; print('\n'.join(sorted(x for x in gym.registry if x.startswith('OpenWorldTactile-'))))"
```

请以当前安装版本的实际输出为准。当前源码注册：

| 任务 ID | 观测/工作流 | 智能体配置 |
|---|---|---|
| `OpenWorldTactile-Ball-Rolling-Tactile-Depth-v1` | 触觉深度/相机滚球 | skrl |
| `OpenWorldTactile-Ball-Rolling-Tactile-RGB-v0` | 触觉 RGB 滚球 | skrl |
| `OpenWorldTactile-Ball-Rolling-Taxim-Fots-v0` | Taxim 与 FOTS 滚球 | skrl |
| `OpenWorldTactile-Ball-Rolling-Tactile-RGB-Uipc-v0` | UIPC 触觉 RGB 滚球 | skrl；仅当可选导入成功时注册 |
| `OpenWorldTactile-Ball-Rolling-Privileged-v0` | 特权状态滚球 | rl_games、rsl_rl、skrl |
| `OpenWorldTactile-Ball-Rolling-Privileged-Reset-with-IK-solver_v0` | 使用 IK 重置的特权滚球 | rl_games、rsl_rl、skrl |
| `OpenWorldTactile-Ball-Rolling-Privileged-Without-Reaching_v0` | 不含 reaching 阶段的特权滚球 | rl_games、rsl_rl、skrl |
| `OpenWorldTactile-Factory-PegInsert-Direct-v0` | 直接式插销装配 | rl_games |
| `OpenWorldTactile-Factory-GearMesh-Direct-v0` | 直接式齿轮啮合 | rl_games |
| `OpenWorldTactile-Factory-NutThread-Direct-v0` | 直接式螺母旋拧 | rl_games |
| `OpenWorldTactile-Pole-Balancing-Base-v0` | 相机/触觉杆平衡 | skrl |
| `OpenWorldTactile-Repose-Cube-Allegro-v0` | Allegro 手内方块重定向 | rl_games、rsl_rl、skrl |

## 连接 Isaac Lab 启动器

未经修改的 Isaac Lab 上游脚本会导入 `isaaclab_tasks`，但不会自动导入本项目的外部任务包。请使用外部项目启动器，或创建上游脚本的项目本地副本，并在 Isaac Sim 启动后、上游任务导入/扩展模板占位符附近添加：

```python
import isaaclab_tasks  # 上游任务注册
import openworldtactile_tasks  # OpenWorldTactile 任务注册
```

遵循 Isaac Sim“先启动应用再导入”的脚本，不要在 `AppLauncher` 之前导入任务模块。Isaac Lab 外部项目模板已在正确位置提供占位符。

下面的 `path/to/owt_*.py` 表示按上述方法准备的启动器。本仓库当前提供任务定义和配置，不重复维护每个上游强化学习启动脚本的副本。

## 冒烟测试环境

创建 Isaac Lab `scripts/environments/random_agent.py` 的项目本地副本并添加上述导入，然后运行小规模用例：

```bash
./run.sh --python path/to/owt_random_agent.py \
  --task OpenWorldTactile-Ball-Rolling-Privileged-v0 \
  --num_envs 1 \
  --headless
```

相机任务需要启用相机：

```bash
./run.sh --python path/to/owt_random_agent.py \
  --task OpenWorldTactile-Ball-Rolling-Tactile-Depth-v1 \
  --num_envs 1 \
  --enable_cameras \
  --headless
```

开始长时间训练前，应先确认环境能够重置和推进。

## 使用 skrl 训练

准备 Isaac Lab `scripts/reinforcement_learning/skrl/train.py` 的项目版本，然后运行：

```bash
./run.sh --python path/to/owt_skrl_train.py \
  --task OpenWorldTactile-Ball-Rolling-Tactile-Depth-v1 \
  --num_envs 8 \
  --enable_cameras \
  --headless
```

正式训练前可用 `--max_iterations N` 做短集成测试。日志按上游训练器约定写入工作目录下的 `logs/skrl/`。

## 使用 rl_games 训练

准备 `scripts/reinforcement_learning/rl_games/train.py` 的项目版本：

```bash
./run.sh --python path/to/owt_rl_games_train.py \
  --task OpenWorldTactile-Factory-PegInsert-Direct-v0 \
  --num_envs 8 \
  --enable_cameras \
  --headless
```

## 使用 rsl_rl 训练

请选择已注册 `rsl_rl_cfg_entry_point` 的任务，例如特权状态滚球：

```bash
./run.sh --python path/to/owt_rsl_rl_train.py \
  --task OpenWorldTactile-Ball-Rolling-Privileged-v0 \
  --num_envs 32 \
  --headless
```

## 回放检查点

使用同一强化学习后端对应的项目版 `play.py`，传入完全一致的任务和检查点：

```bash
./run.sh --python path/to/owt_skrl_play.py \
  --task OpenWorldTactile-Ball-Rolling-Tactile-Depth-v1 \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --num_envs 1 \
  --enable_cameras
```

任务 ID、算法系列、观测配置和包修订版本都应与训练时一致。

## 扩展规模与复现

- 测量显存和步进耗时后再增加 `--num_envs`。
- 相机任务即使无界面运行也需要 `--enable_cameras`。
- UIPC 任务适合的并行环境数远小于刚体/特权状态任务。
- 将训练器导出的环境与智能体配置和检查点一起保存。
- 记录随机种子、包修订、任务 ID、后端、命令、GPU/驱动、Isaac 版本及 Hydra 覆盖项。
- 在独立回放中验证检查点；训练损失下降本身不等于任务成功。

这些任务属于研究环境。存在注册配置并不代表本次发布已重新运行所有后端/任务组合。
