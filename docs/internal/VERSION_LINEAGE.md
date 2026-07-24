# 项目版本谱系

## 两个不能混用的版本维度

1. Isaac Lab 基线：当前 OpenWorldTactile 对应 2.1.1，legacy OpenWorldTactile 对应 2.3.2。
2. OpenWorldTactile 实验版本：V1–V6.2 是 2.1.1 路线内部的实验演进，不是 Isaac Lab 版本号。

## Isaac Lab 2.1.1：OpenWorldTactile/UIPC/AgileX

目录：`../active-isaaclab-2.1.1/`

共同底座：

```text
packages/uipc/libuipc
  -> packages/uipc
     -> UIPC_Pad.usda / Piper USD
        -> experiments/tactile-bench
```

实验主线：

```text
V1    固定单膜压入、摩擦和 deformation-based fxyz
 -> V2    力估计核心模块化
 -> V2.1  dense membrane 参考路线
 -> V2.2  camera-observed membrane
 -> V2.3  RGB marker tracking 剪切估计
 -> V3    自动验证套件规划
 -> V4    Piper/OpenWorldTactile 挂载和接触验证
 -> V5    抓取、接触、形变、力估计与触觉场迭代
 -> V6    自由物体抓取、提升和完整周期
 -> V7    数据质量规划
 -> V8    真实力标定规划
```

V4 主要阶段：

- V4/V4.1/V4.2：挂载接触、夹爪开合和局部三轴力。
- V4.3–V4.6：资产契约、合成力、纹理压印和压力场。
- V4.7–V4.9c：真实 UIPC 接触、Pad 资产和 AgileX 挂载检查。

V5 主要阶段：

- V5/V5.1–V5.3：机器人挂载、抓取、接触间隙和位置对齐。
- V5_new_1–3h：挂载确认、PhysX 抓取、Fz 基线和视觉膜整理。
- V5_new_4–4.5：UIPC 原生力、投影、接触区域和梯度来源检查。
- V5_new_5–6：压力/剪切代理及本构形变触觉。
- V5_new_7a–7e：膜形变、接触和 attachment 跟随验证。
- V5_new_7f：冻结 deformation contract。
- V5_new_7g：deformation-based 三轴力估计、局部坐标和剪切验证。
- V5_new_8：抓取触觉集成。
- V5_new_9：TU tactile field 渲染。

V6 主要阶段：

- V6.0：自由物体抓取与提升。
- V6.1：自由物体完整周期验证。
- V6.1b：短提升保持诊断。
- V6.2：简化抓取触觉组合；是当前说明中的默认参考入口，不替代或删除历史版本。

当前直接导入链：

```text
V6.0 / V6.1 / V6.1b / V6.2
  -> OpenWorldTactile_v5_new_9_tu_tactile_field_rendering.py
     -> OpenWorldTactile_v5_new_7g_deformation_force_estimator.py
     -> tu_tactile_field.py
```

此外，7g 可视化和测试会直接导入 `membrane_local_frame.py`、7f/7g 实现及同目录辅助模块。因此 `OpenWorldTactileBench/` 继续作为一个完整移动单元，不按版本拆目录。

V3、V7、V8 主要以规划和说明文档存在；它们不能按连续、均已实现的代码版本理解。完整脚本文件名见 `OWTBENCH_VERSION_INDEX.md`。

## Isaac Lab 2.3.2：OpenWorldTactile/GelSight

目录：`../archive-isaaclab-2.3.2/`

独立数据链：

```text
upstream tactile depth / SDF / contact
  -> GelSight RGB / OpenWorldTactile RGB
     -> OpenWorldTactile SDK hue / flow
        -> fxyz
```

该路线保留 `packages/contrib`、少量 `packages/assets-patches` 配置参考、实验脚本、外部 SDK 挂载说明和实验记录。SDK 本体不随公开仓库分发。它不是 2.1.1 UIPC 主线的后续版本，也不应覆盖到 2.1.1 的 `packages/` 中。

## 保留策略

- 所有现有实验版本代码均保留，项目自有名称统一为 OpenWorldTactile，实验演进与相对引用关系保持一致。
- “默认参考入口”只用于导航，不代表删除其他版本。
- 最终载荷由 `../tools/repository/FINAL_MANIFEST.csv` 记录路径、大小和 SHA-256。
- 本谱系仅描述文件和静态依赖，不代表已完成运行环境验证。
