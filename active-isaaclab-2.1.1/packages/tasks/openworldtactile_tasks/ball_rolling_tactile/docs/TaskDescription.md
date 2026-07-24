# Tactile ball rolling / 触觉滚动小球

[English](#english) · [简体中文](#简体中文)

<a id="english"></a>

## English

The agent controls a Franka Panda arm with a single GelSight Mini end effector and must roll a ball to a sampled planar goal. Relative task-space actions move the end effector; observations combine robot/task state with a tactile image representation.

Registered variants:

| Environment ID | Tactile observation |
|---|---|
| `OpenWorldTactile-Ball-Rolling-Tactile-Depth-v1` | tactile depth/camera image |
| `OpenWorldTactile-Ball-Rolling-Tactile-RGB-v0` | tactile RGB image |
| `OpenWorldTactile-Ball-Rolling-Taxim-Fots-v0` | Taxim rendering plus FOTS marker simulation |
| `OpenWorldTactile-Ball-Rolling-Tactile-RGB-Uipc-v0` | UIPC-backed tactile RGB; registered only when optional imports succeed |

The reward combines object contact/reaching, progress toward the ball goal, success, end-effector safety/orientation, action-rate, and joint-velocity terms. Several variants initialize the end effector near the ball so training does not spend most samples before tactile contact.

Camera-based variants require `--enable_cameras`. Begin with one environment, verify reset and stepping, then scale while watching GPU memory. For launcher preparation and training commands, see [Tasks and training](../../../../../../docs/en/guides/tasks-and-training.md). The environment classes are authoritative for exact tensor shapes and reward weights.

<a id="简体中文"></a>

## 简体中文

智能体控制带单个 GelSight Mini 末端执行器的 Franka Panda 机械臂，将小球滚动到平面内随机采样的目标位置。相对任务空间动作控制末端执行器，观测由机器人/任务状态和触觉图像表示组成。

已注册的变体：

| 环境 ID | 触觉观测 |
|---|---|
| `OpenWorldTactile-Ball-Rolling-Tactile-Depth-v1` | 触觉深度/相机图像 |
| `OpenWorldTactile-Ball-Rolling-Tactile-RGB-v0` | 触觉 RGB 图像 |
| `OpenWorldTactile-Ball-Rolling-Taxim-Fots-v0` | Taxim 渲染与 FOTS 标记点仿真 |
| `OpenWorldTactile-Ball-Rolling-Tactile-RGB-Uipc-v0` | UIPC 触觉 RGB；仅在可选导入成功时注册 |

奖励综合考虑接触/到达小球、向目标推进、成功、末端高度与姿态安全、动作变化率和关节速度。部分变体会在重置时把末端初始化到小球附近，减少训练在发生触觉接触前浪费的样本。

相机变体必须启用 `--enable_cameras`。建议先用单环境确认可重置和步进，再观察显存逐步扩展。启动器准备和训练命令见[任务与训练](../../../../../../docs/zh-CN/guides/tasks-and-training.md)。精确张量形状和奖励权重以环境类为准。
