# Task package / 任务包

[English](#english) · [简体中文](#简体中文)

<a id="english"></a>

## English

`openworldtactile_tasks` contains Gymnasium registrations, environment configurations, and RL-agent configuration files for the project's ball-rolling, factory assembly, pole-balancing, and in-hand manipulation tasks.

After installation, import the package before inspecting the registry:

```bash
cd active-isaaclab-2.1.1
./run.sh --python -c "import gymnasium as gym; import openworldtactile_tasks; print('\n'.join(sorted(x for x in gym.registry if x.startswith('OpenWorldTactile-'))))"
```

Upstream Isaac Lab training scripts do not automatically import this external task package. A project-aware launcher must start Isaac Sim first and then import both `isaaclab_tasks` and `openworldtactile_tasks`. The [tasks and training guide](../../../../docs/en/guides/tasks-and-training.md) lists every registered task, the compatible agent configurations, smoke-test commands, training examples, checkpoint playback, and scaling guidance.

Treat the Python environment configuration as authoritative for action, observation, reward, reset, and termination semantics. The nearby task descriptions are concise overviews rather than a replacement for the source configuration.

<a id="简体中文"></a>

## 简体中文

`openworldtactile_tasks` 包含项目的 Gymnasium 注册、环境配置和强化学习智能体配置，覆盖滚动小球、工厂装配、杆平衡与灵巧手在手操作任务。

安装后，先导入任务包再检查注册表：

```bash
cd active-isaaclab-2.1.1
./run.sh --python -c "import gymnasium as gym; import openworldtactile_tasks; print('\n'.join(sorted(x for x in gym.registry if x.startswith('OpenWorldTactile-'))))"
```

上游 Isaac Lab 训练脚本不会自动导入此外部任务包。项目专用启动器必须先启动 Isaac Sim，再同时导入 `isaaclab_tasks` 和 `openworldtactile_tasks`。[任务与训练指南](../../../../docs/zh-CN/guides/tasks-and-training.md)列出了全部任务 ID、兼容的智能体配置、冒烟测试命令、训练示例、检查点回放和扩展建议。

动作、观测、奖励、重置和终止语义以 Python 环境配置为准。各任务目录中的说明是简要概览，不能替代源码配置。
