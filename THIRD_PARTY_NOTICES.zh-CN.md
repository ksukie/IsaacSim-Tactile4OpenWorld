<p align="right">
  <a href="THIRD_PARTY_NOTICES.md">English</a> · <strong>简体中文</strong>
</p>

# 第三方通知

IsaacSim-Tactile4OpenWorld 是围绕 OpenWorldTactile 框架构建的多许可证研究发行物。根目录 [`LICENSE`](LICENSE) 为 BSD-3-Clause；除非文件或子目录另有声明，它覆盖 OpenWorldTactile 原创贡献。该许可证不会替代上游通知、重新许可第三方作品，也不授予商标权。

## 随仓库分发的组件

| 组件 | 分发路径 | 上游 | 许可证与本地通知 | 说明 |
|---|---|---|---|---|
| Isaac Lab 与 ORBIT 衍生代码 | `active-isaaclab-2.1.1/packages/{core,assets,tasks}/`、部分实验代码及 `archive-isaaclab-2.3.2/packages/contrib/` | <https://github.com/isaac-sim/IsaacLab> | BSD-3-Clause；文件 SPDX 头及 [`archive-isaaclab-2.3.2/LICENSE`](archive-isaaclab-2.3.2/LICENSE) | 必须保留现有 Isaac Lab/ORBIT 版权行。OpenWorldTactile 修改不表示 NVIDIA 或 Isaac Lab 对项目背书。 |
| libuipc | `active-isaaclab-2.1.1/packages/uipc/libuipc/` | <https://github.com/spiriMirror/libuipc> | Apache-2.0；[`LICENSE`](active-isaaclab-2.1.1/packages/uipc/libuipc/LICENSE) | 内置源码报告版本为 0.9.0-alpha。下列子组件对其自身文件使用不同许可证。 |
| libuipc Python 绑定 | `active-isaaclab-2.1.1/packages/uipc/libuipc/python/` | libuipc 上游 | GPL-3.0-only；[`LICENSE`](active-isaaclab-2.1.1/packages/uipc/libuipc/python/LICENSE) | 不得将该子树描述为 MIT 或 BSD。分发组合二进制时必须满足适用 GPL 条款。 |
| TetGen | `active-isaaclab-2.1.1/packages/uipc/libuipc/external/tetgen/` | <https://wias-berlin.de/software/tetgen/> | AGPL-3.0-or-later 或单独商业许可证；[`LICENSE`](active-isaaclab-2.1.1/packages/uipc/libuipc/external/tetgen/LICENSE) | 这是强 copyleft 许可证。专有二进制使用 TetGen 时需要从权利人另行取得权利。 |
| MuDA | `active-isaaclab-2.1.1/packages/uipc/libuipc/external/muda/` | <https://github.com/MuGdxy/muda> | Apache-2.0；[`LICENSE`](active-isaaclab-2.1.1/packages/uipc/libuipc/external/muda/LICENSE) | 子树中的 Catch2 使用自己的许可证。 |
| Catch2 2.13.10 | `active-isaaclab-2.1.1/packages/uipc/libuipc/external/muda/external/catch2/` | <https://github.com/catchorg/Catch2> | BSL-1.0；[`LICENSE_1_0.txt`](active-isaaclab-2.1.1/packages/uipc/libuipc/external/muda/external/catch2/LICENSE_1_0.txt) | 生成的单头文件发行物；必须保留其头部通知。 |
| Octree | `active-isaaclab-2.1.1/packages/uipc/libuipc/external/octree/Octree/` | <https://github.com/attcs/Octree> | MIT；[`LICENSE`](active-isaaclab-2.1.1/packages/uipc/libuipc/external/octree/Octree/LICENSE) | 版权所有 Attila Csikós/attcs。 |
| SimpleBVH | `active-isaaclab-2.1.1/packages/uipc/libuipc/src/geometry/bvh/BVH.*` | <https://github.com/geometryprocessing/SimpleBVH> | MIT；[`LICENSE-SimpleBVH`](active-isaaclab-2.1.1/packages/uipc/libuipc/src/geometry/bvh/LICENSE-SimpleBVH) | 在 libuipc 中有修改；审计时核对的上游修订为 `69920e15db5281ce89d3618604ed63ea9e4e0bce`。 |
| NSEssentials Morton 代码 | `active-isaaclab-2.1.1/packages/uipc/libuipc/src/geometry/bvh/Morton.*` | <https://github.com/nschertler/NSEssentials> | BSD-3-Clause；[`LICENSE-NSEssentials`](active-isaaclab-2.1.1/packages/uipc/libuipc/src/geometry/bvh/LICENSE-NSEssentials) | 版权所有 Nico Schertler；审计时核对的上游修订为 `e5bceb01708368756528fdc95054bd3d05b7b745`。 |
| AgileX Piper ROS | `active-isaaclab-2.1.1/vendor/agilex-piper/piper_ros/` | <https://github.com/agilexrobotics/piper_ros> | MIT；[`LICENSE`](active-isaaclab-2.1.1/vendor/agilex-piper/piper_ros/LICENSE) | 版权所有 RosenYin。 |
| MoveIt 1.1.11 快照 | `active-isaaclab-2.1.1/vendor/agilex-piper/piper_ros/src/piper_moveit/moveit-1.1.11/` | <https://github.com/moveit/moveit> | BSD 类条款；[`LICENSE.txt`](active-isaaclab-2.1.1/vendor/agilex-piper/piper_ros/src/piper_moveit/moveit-1.1.11/LICENSE.txt) | 保留各文件与许可证文件中的全部版权通知。 |
| GPU Taxim/Taxim 衍生实现 | `active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/gpu_taxim/sim/` | <https://github.com/CMURoboTouch/Taxim> | MIT；[`LICENSE`](active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/gpu_taxim/sim/LICENSE) | 版权所有 CMURoboTouch。使用该方法时应引用 Taxim。 |
| FOTS 衍生标记点运动 | `active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/fots/sim/marker_motion.py` | <https://github.com/Rancho-zhao/FOTS> | MIT；[`LICENSE`](active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/fots/LICENSE) | 文件已为 OpenWorldTactile 修改。上游 README 声明 MIT，但 2026-07-22 审计时上游 `LICENSE` 链接不可用。本地保留条款使预期授权可见；商业再分发前，发布维护者仍应复核上游历史。 |
| ManiSkill-ViTac2025 衍生触觉仿真器 | `active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/fem_based/sim/tactile_sensor_sapienipc_modified.py` | <https://github.com/chuanyune/ManiSkill-ViTac2025> | Apache-2.0；[`LICENSE`](active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/fem_based/sim/LICENSE) | 文件名与头部表明其经过修改；须保留来源永久链接与修改说明。 |
| GelSight Mini 模型及衍生传感器资产 | `active-isaaclab-2.1.1/packages/assets/openworldtactile_assets/data/Sensors/GelSight_Mini/` 及包含 GelSight 的 Franka 装配 | <https://github.com/gelsightinc/gsrobotics> | GPL-3.0-only；两个资产根目录均有本地 `LICENSE` | 上游以 GPL-3.0 分发 GelSight Mini CAD 模型。转换后的 USD 与组合资产形式按 GPL-3.0-only 保守分发。 |
| Franka 机器人描述与网格 | `active-isaaclab-2.1.1/packages/assets/openworldtactile_assets/data/Robots/Franka/` 下包含 GelSight 的 Franka 装配 | <https://github.com/frankarobotics/franka_ros> | Apache-2.0；[`LICENSE-APACHE-2.0`](active-isaaclab-2.1.1/packages/assets/openworldtactile_assets/data/Robots/Franka/LICENSE-APACHE-2.0) | 同时包含 GelSight 内容的装配必须满足两组适用条款。 |

## 未纳入公开发行的内容

2026-07-22 发布审计因再分发权利或来源记录不足，从仓库中移除了：

- 历史 `hardware-sdk/openworldtactile/` 包，包括 `SonixCamera.dll` 和 `libSonixCamera.so`；
- 来自 `danfergo/gelsight-simulation` 的触觉测试形状资产；审计时其上游仓库未提供明确许可证；
- 不透明的 `Policies/ball_rolling/IK_old.pt` 检查点；它缺少作者、训练来源和模型许可证；
- 错误声明 MIT 的生成型 `*.egg-info` 包元数据。

这些排除是有意的发布边界，并不表示原内容必然侵权。只有在具备可验证许可证或书面再分发许可、准确上游修订和所需通知时，才能恢复被排除组件。

## 研究归属

许可证义务与学术引用义务相互独立。已实现方法对应论文见 [`CITATIONS.zh-CN.md`](CITATIONS.zh-CN.md)；上游项目的 README 可能还包含其他引用要求。

> 本中文版本用于帮助理解，不能替代各组件的英文许可证原文；发生差异时以对应许可证文件为准。
