# Privileged-state ball rolling / 特权状态滚动小球

[English](#english) · [简体中文](#简体中文)

<a id="english"></a>

## English

This task family is the lower-cost baseline for tactile ball rolling. A Franka Panda arm rolls a ball to a sampled planar goal, but the policy receives privileged simulator state such as the current ball position instead of tactile images. It is useful for checking task dynamics, reward design, and trainer integration before adding camera or deformable-sensor cost.

Registered variants:

- `OpenWorldTactile-Ball-Rolling-Privileged-v0`: standard reset and full reaching-plus-goal reward;
- `OpenWorldTactile-Ball-Rolling-Privileged-Reset-with-IK-solver_v0`: uses IK to reset the end effector near the ball;
- `OpenWorldTactile-Ball-Rolling-Privileged-Without-Reaching_v0`: removes the reaching term to isolate ball-to-goal control.

Actions are relative end-effector task-space commands. Rewards cover reaching/contact, ball-to-goal progress and success, safe end-effector pose, smooth actions, and joint velocity. IK reset reduces the learned reaching problem but adds reset-time compute, so compare throughput and task success rather than assuming it is always faster.

See [Tasks and training](../../../../../../docs/en/guides/tasks-and-training.md) for supported RL backends and launcher setup. Exact observations, term weights, and termination thresholds are defined by the environment configuration classes.

<a id="简体中文"></a>

## 简体中文

本任务族是触觉滚动小球的低成本基线。Franka Panda 机械臂将小球滚动到平面内随机目标，但策略直接获得当前小球位置等仿真特权状态，不使用触觉图像。它适合在引入相机或可形变传感器开销前检查任务动力学、奖励设计和训练器集成。

已注册的变体：

- `OpenWorldTactile-Ball-Rolling-Privileged-v0`：标准重置，包含到达和目标推进的完整奖励；
- `OpenWorldTactile-Ball-Rolling-Privileged-Reset-with-IK-solver_v0`：使用 IK 将末端重置到小球附近；
- `OpenWorldTactile-Ball-Rolling-Privileged-Without-Reaching_v0`：移除到达奖励，单独研究小球到目标的控制。

动作是末端执行器的相对任务空间命令。奖励包含到达/接触、向目标推进与成功、末端姿态安全、动作平滑和关节速度。IK 重置降低了需要学习的到达难度，但增加了重置计算量，应同时比较吞吐率和任务成功率，不能默认其一定更快。

支持的强化学习后端和启动器配置见[任务与训练](../../../../../../docs/zh-CN/guides/tasks-and-training.md)。精确观测、各项权重和终止阈值以环境配置类为准。
