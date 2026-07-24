# Isaac Sim 与 Isaac Lab 简介

## 最重要的结论

Isaac Sim 和 Isaac Lab 不是同一个东西。

```text
Isaac Sim = 机器人仿真软件
Isaac Lab = 基于 Isaac Sim 的机器人训练框架
```

更通俗地说：

```text
Isaac Sim 负责“把机器人和世界仿真出来”
Isaac Lab 负责“让机器人在仿真里学习和训练”
```

所以你运行 Isaac Lab 的脚本时，屏幕上打开的窗口仍然会显示：

```text
Isaac Sim 5.1.0
```

这是正常的。因为图形界面、物理仿真和机器人显示都是 Isaac Sim 提供的。

## Isaac Sim 是什么

Isaac Sim 是 NVIDIA 的机器人仿真平台。

它主要负责：

- 显示机器人和场景。
- 运行物理仿真。
- 加载机器人模型。
- 模拟相机、激光雷达、接触传感器等传感器。
- 支持 GPU 加速仿真和 RTX 渲染。
- 使用 USD 格式管理机器人、场景和资产。

可以把 Isaac Sim 理解成一个“机器人世界模拟器”。

例如你看到的四足机器人、地面、灯光、相机视角、碰撞效果，都是 Isaac Sim 在负责。

## Isaac Lab 是什么

Isaac Lab 是 NVIDIA 提供的机器人学习框架。

它建立在 Isaac Sim 之上，主要用于机器人训练和实验。

它主要负责：

- 定义机器人学习任务。
- 设置观测量，也就是机器人能看到或感知到什么。
- 设置动作空间，也就是机器人能控制哪些关节。
- 设置奖励函数，也就是怎样判断机器人做得好不好。
- 并行运行大量机器人环境，加快训练速度。
- 对接强化学习算法库，例如 `rsl_rl`。
- 保存训练好的模型。
- 重新加载模型，让机器人在 Isaac Sim 中运行。

可以把 Isaac Lab 理解成一个“机器人训练工具箱”。

## 两者的关系

Isaac Lab 离不开 Isaac Sim。Isaac Lab 本身不负责底层渲染和物理，而是调用 Isaac Sim 来完成这些事情。

关系可以写成：

```text
Isaac Lab
  ↓ 调用
Isaac Sim
  ↓ 运行
机器人仿真世界
```

或者：

```text
Isaac Sim 提供仿真环境
Isaac Lab 在这个环境里训练机器人
```

类比一下：

```text
Isaac Sim 像游戏引擎
Isaac Lab 像基于游戏引擎写的训练系统
```

## 为什么打开的是 Isaac Sim 窗口

因为 Isaac Lab 没有一个单独叫“Isaac Lab”的图形窗口。

当你执行：

```bash
./isaaclab.sh -p 某个脚本.py
```

实际过程是：

```text
1. Isaac Lab 启动 Python 脚本
2. 脚本创建任务、机器人和场景
3. Isaac Lab 调用 Isaac Sim
4. Isaac Sim 打开窗口并显示仿真画面
```

所以窗口标题显示 Isaac Sim 是正确的。

判断你是不是在用 Isaac Lab，不是看窗口标题，而是看你是否从 Isaac Lab 目录运行了 Isaac Lab 的脚本。

例如：

```bash
cd ~/IsaacLab-v2.3.2
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py
```

这就是在用 Isaac Lab。

## 训练时谁负责什么

训练机器人时，两者分工大概是：

```text
Isaac Sim：
负责仿真世界、机器人运动、碰撞、传感器、画面显示。

Isaac Lab：
负责训练任务、奖励函数、强化学习流程、模型保存和模型加载。
```

训练流程可以简单理解为：

```text
1. Isaac Lab 创建训练任务
2. Isaac Sim 同时仿真很多个机器人
3. 机器人不断尝试动作
4. Isaac Lab 根据表现给奖励
5. 强化学习算法更新策略模型
6. 训练完成后得到 checkpoint 模型
7. 再用 Isaac Lab 加载模型，在 Isaac Sim 里运行
```

## train.py 和 play.py 的区别

Isaac Lab 中常见两个脚本：

```text
train.py = 训练模型
play.py = 加载模型并运行
```

例如：

```text
train.py
让机器人从不会走开始，通过试错学习。

play.py
加载已经训练好的模型，让机器人按照策略自己动起来。
```

也就是说：

```text
先用 train.py 训练
再用 play.py 观看效果
```

## 一句话理解

如果只记一句话：

```text
Isaac Sim 是仿真器，Isaac Lab 是训练框架。
Isaac Lab 用 Isaac Sim 来训练和测试机器人。
```

