# Isaac Lab 入门与训练使用说明

这份文档是给自己复习用的简明笔记，记录 Isaac Sim、Isaac Lab 的关系，以及如何在当前机器上启动、训练、查看日志和加载训练好的模型。

## 1. Isaac Lab 是什么

Isaac Lab 是 NVIDIA 提供的机器人学习框架，主要用来做机器人强化学习、模仿学习、任务环境搭建和仿真训练。

可以简单理解为：

```text
Isaac Lab = 用来训练机器人的工具箱
```

它常见用途包括：

- 训练四足机器人走路，例如 Unitree Go2、ANYmal。
- 训练机械臂抓取、搬运、移动物体。
- 同时并行运行很多个机器人环境，加快强化学习训练。
- 定义任务、奖励函数、观测量、动作空间。
- 对接强化学习库，例如 `rsl_rl`、`rl_games`、`skrl`。
- 把训练好的策略模型重新加载到仿真里，让机器人自己动起来。

Isaac Lab 本身不是一个单独的图形软件。它更多是一套 Python 代码、训练脚本、任务配置和机器人环境。

## 2. Isaac Sim 和 Isaac Lab 的关系

你打开窗口后看到的标题通常是：

```text
Isaac Sim 5.1.0
```

这是正常的。因为真正负责显示画面、物理仿真、机器人模型加载的是 Isaac Sim。

两者关系可以这样理解：

```text
Isaac Sim = 底层仿真器，负责画面、物理、传感器、机器人模型
Isaac Lab = 上层训练框架，负责任务、奖励、训练脚本、策略模型
```

类比一下：

```text
Isaac Sim 像游戏引擎
Isaac Lab 像基于这个游戏引擎写的机器人训练项目
```

所以当你运行：

```bash
./isaaclab.sh -p scripts/demos/quadrupeds.py
```

实际发生的是：

```text
Isaac Lab 脚本启动 Isaac Sim
Isaac Lab 创建场景和机器人
Isaac Sim 负责显示和仿真
```

训练时也是一样：Isaac Lab 负责训练逻辑，Isaac Sim 负责底层仿真。

## 3. 当前安装环境

当前机器环境大致如下：

```text
系统：Ubuntu 24.04.4 LTS
Python：3.11.15
Isaac Sim：5.1.0
Isaac Lab：v2.3.2
GPU：NVIDIA GeForce RTX 5090 D v2
显存：约 24 GB
驱动：590.48.01
训练环境名：isaaclab
Isaac Lab 路径：~/IsaacLab-v2.3.2
```

进入 Isaac Lab 前，建议先执行：

```bash
conda activate isaaclab
cd ~/IsaacLab-v2.3.2
export OMNI_KIT_ACCEPT_EULA=YES
```

其中：

- `conda activate isaaclab`：进入安装 Isaac Sim / Isaac Lab 的 Python 环境。
- `cd ~/IsaacLab-v2.3.2`：进入 Isaac Lab 项目目录。
- `export OMNI_KIT_ACCEPT_EULA=YES`：接受 Isaac Sim / Omniverse 的 EULA，避免启动时卡住确认。

## 4. 常用启动与验证命令

### 4.1 验证 Isaac Lab 是否能启动

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

如果看到类似：

```text
[INFO]: Setup complete...
```

说明 Isaac Lab 可以正常启动。

注意：这个脚本可能不会自动退出，因为它会保持仿真循环运行。确认成功后可以按：

```bash
Ctrl + C
```

停止程序。

### 4.2 打开一个空场景 GUI

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

这个命令会打开 Isaac Sim 图形窗口。第一次启动可能比较慢，因为要加载扩展和编译 shader。

### 4.3 打开四足机器人展示 demo

```bash
./isaaclab.sh -p scripts/demos/quadrupeds.py
```

这个 demo 会加载多个四足机器人模型。它主要用于展示机器人资产，不一定代表训练好的行走策略。

如果窗口提示“无响应”，通常先点“等待”。第一次加载可能要几分钟。

## 5. 如何查看可用任务

查看所有环境任务：

```bash
./isaaclab.sh -p scripts/environments/list_envs.py
```

只查看 Go2 相关任务：

```bash
./isaaclab.sh -p scripts/environments/list_envs.py --keyword Go2
```

你当前查到的 Go2 任务包括：

```text
Isaac-Velocity-Flat-Unitree-Go2-v0
Isaac-Velocity-Flat-Unitree-Go2-Play-v0
Isaac-Velocity-Rough-Unitree-Go2-v0
Isaac-Velocity-Rough-Unitree-Go2-Play-v0
```

这些名字可以这样理解：

```text
Flat = 平地
Rough = 崎岖地形
Unitree-Go2 = 机器人型号
Velocity = 速度跟踪任务
Play = 用来播放/展示策略的环境
```

一般规律：

```text
不带 Play 的任务：用于训练
带 Play 的任务：用于加载模型并观看效果
```

## 6. 如何训练 Unitree Go2

训练 Go2 平地行走任务：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 1000
```

每个参数的意思：

```text
scripts/reinforcement_learning/rsl_rl/train.py
训练脚本，使用 rsl_rl 这个强化学习库。

--task Isaac-Velocity-Flat-Unitree-Go2-v0
训练 Unitree Go2 在平地上跟踪目标速度。

--headless
无界面训练，不打开 Isaac Sim 窗口，速度更快。

--num_envs 1024
同时并行训练 1024 个机器人环境。

--max_iterations 1000
训练 1000 轮。
```

训练过程可以理解为：

```text
1. Isaac Sim 创建很多个 Go2 机器人。
2. 每个机器人收到一个目标速度。
3. 策略网络输出每个关节该怎么动。
4. 仿真器判断机器人是否走得稳、是否跟上目标速度、是否摔倒。
5. 根据奖励函数给分。
6. 强化学习算法根据得分更新策略网络。
7. 重复很多轮后，机器人逐渐学会走路。
```

如果显存不够，可以把环境数量调小：

```bash
--num_envs 512
```

或者：

```bash
--num_envs 256
```

## 7. 训练日志怎么看

训练时终端会不断输出类似内容：

```text
Learning iteration 403/1000
Computation: 53750 steps/s
Mean action noise std: 0.56
Mean value_function loss: 0.0126
Mean surrogate loss: -0.0065
Mean entropy loss: 9.7585
Mean reward: 17.81
Mean episode length: 972.48
Episode_Reward/track_lin_vel_xy_exp: 0.7408
Episode_Reward/track_ang_vel_z_exp: 0.5538
Metrics/base_velocity/error_vel_xy: 0.9502
Metrics/base_velocity/error_vel_yaw: 0.4628
Episode_Termination/time_out: 0.9482
Episode_Termination/base_contact: 0.0518
Total timesteps: 9928704
Iteration time: 0.46s
ETA: 00:04:41
```

主要看这些：

```text
Learning iteration 403/1000
当前第 403 轮，总共训练 1000 轮。
```

```text
Computation: 53750 steps/s
每秒模拟 53750 步，越高说明训练越快。
```

```text
Mean reward: 17.81
平均奖励，越高越好。刚开始可能是负数，后面变高说明机器人在进步。
```

```text
Mean episode length: 972.48
平均每回合能坚持多少步。接近最大步数说明机器人不容易摔倒。
```

```text
Episode_Reward/track_lin_vel_xy_exp
前后/左右速度跟踪奖励。越高说明机器人越能按目标速度走。
```

```text
Episode_Reward/track_ang_vel_z_exp
转向速度跟踪奖励。越高说明机器人越能按目标转弯。
```

```text
Episode_Reward/lin_vel_z_l2
上下乱跳惩罚。负数越小，说明身体上下跳动越少。
```

```text
Episode_Reward/ang_vel_xy_l2
身体翻滚、俯仰晃动惩罚。负数越小，说明身体更稳。
```

```text
Episode_Reward/dof_torques_l2
关节用力惩罚。目的是让机器人不要靠过大的力乱甩腿。
```

```text
Episode_Reward/action_rate_l2
动作变化惩罚。目的是让动作更平滑。
```

```text
Metrics/base_velocity/error_vel_xy
平面速度误差。越小越好。
```

```text
Metrics/base_velocity/error_vel_yaw
转向速度误差。越小越好。
```

```text
Episode_Termination/time_out
正常跑到回合结束的比例。越高越好。
```

```text
Episode_Termination/base_contact
机身碰地或摔倒结束的比例。越低越好。
```

一句话判断训练是否变好：

```text
Mean reward 逐渐升高
Mean episode length 接近最大值
速度误差逐渐下降
base_contact 比例较低
```

## 8. 训练完成后如何加载模型运行

训练完成后，模型会保存在：

```bash
~/IsaacLab-v2.3.2/logs/rsl_rl/
```

你这次 Go2 训练的示例路径是：

```bash
logs/rsl_rl/unitree_go2_flat/2026-06-11_17-54-14/model_999.pt
```

其中：

```text
unitree_go2_flat = Go2 平地任务
2026-06-11_17-54-14 = 本次训练的时间目录
model_999.pt = 第 999 轮保存的模型，通常是最终模型
```

加载最终模型并打开 GUI 观看：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/unitree_go2_flat/2026-06-11_17-54-14/model_999.pt \
  --real-time
```

参数解释：

```text
play.py
播放训练好的策略，不再继续训练。

--task Isaac-Velocity-Flat-Unitree-Go2-Play-v0
使用 Go2 平地行走的展示环境。

--num_envs 1
只显示 1 个机器人，方便观察。

--checkpoint ...
指定要加载的模型文件。

--real-time
尽量按真实时间速度播放。
```

如果想先无界面确认模型能加载：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/unitree_go2_flat/2026-06-11_17-54-14/model_999.pt \
  --headless
```

如果想看多个机器人一起跑：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --num_envs 16 \
  --checkpoint logs/rsl_rl/unitree_go2_flat/2026-06-11_17-54-14/model_999.pt \
  --real-time
```

如果最后一个模型效果不理想，也可以试试中间模型：

```bash
model_800.pt
model_900.pt
model_950.pt
```

有时中间某个 checkpoint 的视觉效果可能比最后一个更稳。

## 9. Isaac Sim 窗口基础操作

打开 GUI 后，常用操作：

```text
鼠标右键拖动：旋转视角
鼠标中键拖动：平移视角
鼠标滚轮：缩放
按住右键 + W/A/S/D：像游戏一样移动相机
选中物体后按 F：聚焦到该物体
```

左侧工具栏：

```text
播放按钮：开始仿真
暂停按钮：暂停仿真
重置按钮：重置场景
箭头工具：选择物体
移动工具：移动物体
```

右侧 `Stage` 面板：

```text
可以查看 World、Robot、关节、身体部件等层级。
```

底部 `Console` 面板：

```text
可以查看 w