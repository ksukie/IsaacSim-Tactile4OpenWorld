<p align="right">
  <a href="../../en/guides/custom-integration.md">English</a> · <strong>简体中文</strong>
</p>

# 自定义集成

将 OpenWorldTactile 接入其他 Isaac Lab 场景时，请使用已安装的项目包。当前 API 属于研究阶段接口，应固定仓库修订版本，并从最接近需求的已有示例开始。

## 包结构

| 导入包 | 公开用途 |
|---|---|
| `openworldtactile` | `GelSightSensor`、对应配置/数据类及光学/标记点仿真方法 |
| `openworldtactile_assets` | 仓库内 USD 路径和预定义机器人/传感器配置 |
| `openworldtactile_uipc` | UIPC 仿真器、对象、连接约束、环境基类和网格工具 |
| `openworldtactile_tasks` | Gymnasium 注册以及任务/智能体配置 |

## 遵循 Isaac Sim 导入顺序

使用 Isaac/Omniverse 模块的脚本必须先启动应用，再导入大部分仿真模块：

```python
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

# 启动应用后再导入 Isaac Lab 与 OpenWorldTactile。
from openworldtactile import GelSightSensor
from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.sensors import GELSIGHT_MINI_TAXIM_CFG
from openworldtactile_uipc import UipcObject, UipcObjectCfg, UipcSim, UipcSimCfg
```

完整脚本应通过 `try/finally` 清理，并调用 `app.close()`。

## 复用资产配置

预定义对象是 Isaac Lab 配置实例。应通过复制或 `replace` 生成新配置，不要修改共享的模块级实例：

```python
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_CFG
from openworldtactile_assets.sensors import GELSIGHT_MINI_TAXIM_CFG

robot_cfg = AGILEX_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
sensor_cfg = GELSIGHT_MINI_TAXIM_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot/gelsight_mini_case",
    data_types=["tactile_rgb", "height_map", "camera_depth"],
)
```

按照现有任务的模式将配置加入 Isaac Lab 场景。可参考：

- [Taxim 传感器原型](../../../active-isaaclab-2.1.1/experiments/simulation-prototypes/check_taxim_sim.py)
- [FEM/UIPC 标记点原型](../../../active-isaaclab-2.1.1/experiments/simulation-prototypes/check_mani_skill_marker_franka.py)
- [Allegro 任务传感器配置](../../../active-isaaclab-2.1.1/packages/tasks/openworldtactile_tasks/inhand/config/allegro_hand/allegro_env_cfg.py)

只有 FEM 依赖成功导入时，包才会导出可选的 `GELSIGHT_MINI_TAXIM_FEM_CFG`。

## 传感器输出

`GelSightSensor.data.output` 是以所请求 `data_types` 为键的字典。不同实现可能提供：

| 键 | 含义 |
|---|---|
| `tactile_rgb` | 合成触觉 RGB 图像 |
| `marker_motion` | 标记点初始/当前位置 |
| `height_map` | 压入高度/深度表示 |
| `camera_depth` | 传感器内部相机深度 |
| `camera_rgb` | 传感器内部相机 RGB |
| `tactile_force_field` | 配置后得到的方法特定触觉场 |

形状取决于传感器相机和具体仿真方法。应检查实际张量/数组与配置，不要假定每种方法返回相同分辨率或力定义。

## UIPC 集成生命周期

最小完整 UIPC 示例会创建 `UipcSim`、创建关联到该仿真器的 `UipcObject`，重置 Isaac 仿真，调用 `uipc_sim.setup_sim()`，并在每次 Isaac 仿真步后更新渲染网格：

```text
创建 SimulationContext
创建 UipcSim(UipcSimCfg(...))
创建 UipcObject(..., uipc_sim) 实例
sim.reset()
uipc_sim.setup_sim()

循环：
    写入运动学/输入状态
    sim.step(...)
    uipc_sim.update_render_meshes()
    更新 Isaac/UIPC 对象缓冲区
    读取形变/接触输出
```

固定柔性膜的紧凑参考见 [V1](../../../active-isaaclab-2.1.1/experiments/tactile-bench/OpenWorldTactile_v1.py)，Pad 局部外部边界耦合见 [V6.2](../../../active-isaaclab-2.1.1/experiments/tactile-bench/OpenWorldTactile_v6_2_grasp.py)。V6.2 的耦合规则具有专用约束，不应只复制其中一部分。

## 坐标与力契约

编写新估计器或数据集前，应明确：

1. 静止膜面与当前膜面所属坐标系；
2. 法向与切向轴方向；
3. 世界坐标到传感器坐标的变换约定；
4. 力通道顺序、符号和单位；
5. 输出是物理反力、限幅后施加力，还是触觉估计量；
6. 重置/静止表面记录时机以及刚体传感器运动的处理方法。

当前触觉基准的 Pad 约定中，局部 `+X` 为外法向，局部 `+Y/+Z` 为切向。未经确认资产自身坐标系，不要把该结论直接用于其他资产。

## 添加任务

1. 在 `packages/tasks/openworldtactile_tasks/` 中实现环境/配置。
2. 使用 Gymnasium 注册唯一的 `OpenWorldTactile-...-vN` ID。
3. 只添加确有对应配置文件的智能体入口。
4. 确保项目启动器在应用启动后导入 `openworldtactile_tasks`。
5. 先用一个环境冒烟测试重置和若干步推进。
6. 用中英文记录观测/动作空间、资产、单位、奖励、终止条件和所需相机/UIPC 参数。

## 扩展检查清单

- 路径应来自 `OWT_ASSETS_DATA_DIR` 或显式配置，不使用机器特定绝对路径。
- 生成输出放在包和资产目录以外。
- 并行 UIPC 仿真使用不同工作目录。
- 即使发生异常也要关闭仿真应用。
- 为坐标变换与数值后处理添加轻量测试。
- 保留第三方 SPDX 头、通知和引用。
- 在发布稳定性策略前，将公开类名视为与修订版本绑定的研究 API。
