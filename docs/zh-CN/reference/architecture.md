<p align="right">
  <a href="../../en/reference/architecture.md">English</a> · <strong>简体中文</strong>
</p>

# 架构

IsaacSim-Tactile4OpenWorld 是围绕外部 Isaac Sim 与 Isaac Lab 环境构建的扩展项目。当前主线组合了传感器模型、资产、任务定义、UIPC 接触仿真和研究实验，但不包含完整 Isaac 平台。

## 系统上下文

```text
外部运行时
  Isaac Sim 4.5 + Isaac Lab 2.1.1 + CUDA/PyTorch
        │
        ▼
OpenWorldTactile 主线
  核心传感器接口 ── 资产/配置 ── 任务环境
        │                │           │
        └────── UIPC 集成 + libuipc ─┘
                         │
                         ▼
                    实验与数据工具
```

独立的 `archive-isaaclab-2.3.2/` 路线不属于这条主线，必须使用不同环境。

## 主线包

| 包 | 职责 | 主要导出/示例 |
|---|---|---|
| `openworldtactile` | GelSight 类传感器接口与触觉渲染/标记点方法 | `GelSightSensor`、`GelSightSensorCfg`、Taxim、FOTS、FEM 实现 |
| `openworldtactile_assets` | USD 数据与预定义 Isaac Lab 配置 | Piper、Franka、Allegro、GelSight Mini、OpenWorldTactile Pad 路径 |
| `openworldtactile_uipc` | Isaac Lab 场景状态与 libuipc 接触/形变之间的桥接 | `UipcSim`、`UipcObject`、连接约束、网格工具、UIPC 强化学习基类 |
| `openworldtactile_tasks` | Gymnasium 任务注册与智能体配置 | 滚球、工厂装配、杆平衡、Allegro 重定向 |

`active-isaaclab-2.1.1/run.sh` 以可编辑方式安装这些包，因此纯 Python 源码修改无需重新构建 wheel 即可生效；原生 UIPC 修改仍需要重新编译。

## 触觉传感器路径

高层 `GelSightSensor` 路线将 Isaac 传感器相机与一种或多种仿真方法组合：

```text
场景几何/接触状态
        │
        ├── 内部深度/RGB 相机
        ├── Taxim 光学渲染
        ├── FOTS 标记点运动
        └── FEM/UIPC 形变源（配置后）
        │
        ▼
GelSightSensorData.output
  tactile_rgb / marker_motion / height_map / camera_depth /
  camera_rgb / 方法特定 tactile_force_field
```

并非每种配置都提供全部输出。`data_types` 选择所需通道，具体光学/标记点仿真器决定其含义与形状。

## UIPC 固定柔性膜路径（V1）

V1 使用 UIPC 求解形变，并根据膜面生成稠密触觉场：

```text
Isaac/PhysX 场景步进
       │
       ▼
UIPC 柔性膜与运动学压头更新
       │
       ▼
当前表面 − 记录的静止表面
       │
       ▼
基于形变的本构估计
       │
       ▼
守恒投影至 [T, 300, 300, 3]
       │
       └── 元数据 + NumPy + 图像/视频预览
```

输出标签为 `sim_constitutive_force`，未标定为牛顿。

## 闭环耦合路径（V6.2）

V6.2 明确区分状态所有权：

- PhysX 独占 Piper 关节系统与自由圆柱状态。
- UIPC 独占传感器 Pad 坐标系中的一张可形变膜。
- 圆柱只以运动学外部边界形式映射到 UIPC。
- UIPC 反力转换到世界坐标，经可接受接触锥投影、松弛/限幅后，在下一个子步施加给 PhysX。
- 冻结 V5.7g 触觉估计器读取膜形变，但不驱动 PhysX。

每个 60 Hz 记录间隔至少包含八个交替 PhysX/UIPC 子步：

```text
上一子步限幅 UIPC 反力 -> PhysX 步进
PhysX 圆柱位姿 -> Pad 局部 UIPC 边界
UIPC 求解 -> 原始反力
接触锥投影 + 松弛 + 限幅
-> 下一 PhysX 子步
```

原始反力、可接受反力、实际施加力、运动、间隙和计时会分别保存，分析输出时必须区分这些量。

## 坐标与通道约定

当前触觉基准 Pad 约定为：

```text
坐标系：Pad 局部
+X：表面外法向
+Y、+Z：切向
形变：当前表面 − 静止表面
fxyz 通道：[局部 Y 剪切，局部 Z 剪切，局部 X 法向]
```

世界坐标数组通常带 `_w` 后缀；Pad/传感器局部数组带 `_pad_local`、`_pad_l` 或在元数据中显式记录坐标系。历史脚本可能不同，因此文件名本身不是通用契约。

## 任务路径

```text
导入 openworldtactile_tasks
       │
       ▼
注册 Gymnasium 环境
       │
       ├── 环境配置入口
       └── 各强化学习后端智能体配置入口
       │
       ▼
能够识别项目的 Isaac Lab random/train/play 启动器
       │
       ▼
日志、配置快照、检查点、可选视频
```

外部启动器必须在启动 Isaac Sim 后导入 `openworldtactile_tasks`。详见[任务与训练](../guides/tasks-and-training.md)。

## 主线与归档边界

| 路线 | 运行基线 | 作用 |
|---|---|---|
| `active-isaaclab-2.1.1/` | Isaac Lab 2.1.1 / Isaac Sim 4.5 | 持续维护的项目主线 |
| `archive-isaaclab-2.3.2/` | 独立 Isaac Lab 2.3.2 环境 | 历史 GelSight/SDK 实验与迁移参考 |

归档中的代码、资产或补丁应逐项迁移。将整个归档覆盖到主线会混合不兼容 API 和未记录的外部依赖。

## 稳定性边界

- 用户指南记录的命令行接口是首选入口，但仍属于研究接口。
- 实验文件名表示研究历史，不是语义化包版本。
- 公开 Python 导出尚无正式弃用策略。
- 输出模式是逐脚本契约；所有历史实验之间不存在统一模式。
- 仓库静态检查只验证结构与可解析性，不验证仿真物理或硬件安全。

发表实验时应固定仓库修订版本，并将命令、元数据和验收结果与数据一起保存。
