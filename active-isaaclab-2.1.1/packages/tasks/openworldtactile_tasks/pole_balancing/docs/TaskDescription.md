# Pole balancing / 杆平衡

[English](#english) · [简体中文](#简体中文)

<a id="english"></a>

## English

The Franka Panda arm uses a single GelSight Mini end effector to keep a rigid pole upright while moving it toward a target pose. The environment combines proprioceptive/task state with a tactile camera observation.

Environment ID: `OpenWorldTactile-Pole-Balancing-Base-v0`

The policy issues relative task-space end-effector commands. Reward terms encourage contact with the pole, target height, upright orientation, target tracking, survival, smooth action changes, and low joint velocity. Episodes terminate on workspace, height, distance, or orientation violations and on the time limit.

This is a camera task: pass `--enable_cameras`, first use `--num_envs 1`, and verify resets before training. See [Tasks and training](../../../../../../docs/en/guides/tasks-and-training.md) for the skrl launcher workflow. The Python environment configuration is authoritative for exact spaces, weights, and thresholds.

<a id="简体中文"></a>

## 简体中文

Franka Panda 机械臂使用单个 GelSight Mini 末端执行器，在将刚性杆移动到目标位姿的同时保持其直立。环境观测由本体/任务状态和触觉相机图像组成。

环境 ID：`OpenWorldTactile-Pole-Balancing-Base-v0`

策略输出末端执行器的相对任务空间命令。奖励项鼓励接触杆、到达目标高度、保持直立、跟踪目标、存活、动作平滑和较低关节速度。当工作空间、高度、距离或姿态越界，或达到时间上限时，回合终止。

这是相机任务：必须传入 `--enable_cameras`，先用 `--num_envs 1` 验证重置和步进，再开始训练。skrl 启动流程见[任务与训练](../../../../../../docs/zh-CN/guides/tasks-and-training.md)。精确空间、权重和阈值以 Python 环境配置为准。
