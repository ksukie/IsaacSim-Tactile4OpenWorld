# Isaac Lab 2.3.2 legacy 指南

## 定位

[`archive-isaaclab-2.3.2/`](../../archive-isaaclab-2.3.2/) 保存 OpenWorldTactile 的 legacy 传感器与相机 SDK 集成路线，根部 [`VERSION`](../../archive-isaaclab-2.3.2/VERSION) 标记为 2.3.2。它包含 58 个公开版本载荷文件，不包含完整 Isaac Lab 2.3.2 本体。

这条路线用于保留旧实验、传感器集成和迁移参考，不是 2.1.1 UIPC 主线的后续版本。

## 目录导航

| 位置 | 内容 |
|---|---|
| [`packages/contrib/`](../../archive-isaaclab-2.3.2/packages/contrib/) | `openworldtactile_sensor` 与上游 Isaac Lab contrib 的 legacy 集成 |
| [`packages/assets-patches/`](../../archive-isaaclab-2.3.2/packages/assets-patches/) | 少量历史资产配置参考 |
| [`experiments/franka/current/`](../../archive-isaaclab-2.3.2/experiments/franka/current/) | Franka 接触、SDF、RGB 与力场实验 |
| [`experiments/sensors/`](../../archive-isaaclab-2.3.2/experiments/sensors/) | OpenWorldTactile/GelSight 传感器演示 |
| [`experiments/rgb-pipeline/`](../../archive-isaaclab-2.3.2/experiments/rgb-pipeline/) | RGB 与 fxyz 组合流程 |
| [`hardware-sdk/`](../../archive-isaaclab-2.3.2/hardware-sdk/) | 外部 OpenWorldTactile 相机 SDK 的挂载位置与许可证边界说明；SDK 本体不分发 |
| [`notes/`](../../archive-isaaclab-2.3.2/notes/) | 来源实验记录与说明 |

根部还保留三个辅助入口：

- [`inspect_gelsight_cfg.py`](../../archive-isaaclab-2.3.2/tools/inspect_gelsight_cfg.py)：检查 GelSight 配置。
- [`set_openworldtactile_camera_view.py`](../../archive-isaaclab-2.3.2/tools/set_openworldtactile_camera_view.py)：设置 OpenWorldTactile 相机视角。
- [`run-original-gelsight.sh`](../../archive-isaaclab-2.3.2/tools/run-original-gelsight.sh)：包装原 GelSight 保存流程。

## 如何选择入口

legacy 没有一个可以代表全部用途的“总入口”：

- 研究 Franka 接触点、SDF、剪切或力场时，从 `experiments/franka/current/` 选择。
- 检查单个 OpenWorldTactile/GelSight 传感器时，从 `sensors/` 选择。
- 研究双传感器、夹持、提升或 RGB-fxyz 联动时，从 `experiments/rgb-pipeline/` 选择。
- 需要逐文件查看静态导入、资产与主守卫时，使用 [脚本入口矩阵](ENTRYPOINT_MATRIX.md)。

## 环境边界

运行 legacy 内容至少涉及以下外部条件：

- 完整 Isaac Lab 2.3.2 与匹配的 Isaac Sim/Python 环境。
- `OWT_ASSET_ROOT` 指向的 GelSight R1.5 触觉资产，以及 `${ISAACLAB_NUCLEUS_DIR}/Factory` 或等价本地映射。
- 从权利人或授权渠道取得的 OpenWorldTactile SDK、Python 依赖、相机驱动、USB 设备和与目标平台匹配的原生库。
- 对 `isaaclab_contrib` 和少量 `isaaclab_assets` 修改进行人工迁入或扩展注册；当前目录不是完整上游仓库。

SDK 本体因缺少可验证的再分发授权而不随仓库提供。取得合法副本后可放入本版本目录的 `hardware-sdk/openworldtactile/`；相关适配也允许显式覆盖：

```bash
export OWT_SDK_ROOT=/absolute/path/to/hardware-sdk/openworldtactile
export OWT_ASSET_ROOT=/absolute/or/nucleus/path/to/openworldtactile-assets
```

外部原生库可取得并不等于目标操作系统、CPU 架构、驱动或相机已经兼容。本次没有加载 SDK、连接硬件或启动仿真。

项目自有目录、模块、脚本、显示名称和外部资产配置入口均使用 OpenWorldTactile。

## 迁移原则

- 先在独立的 Isaac Lab 2.3.2 环境恢复 legacy，再决定是否向新基线移植。
- 不把整个历史归档的 `packages/` 直接覆盖进 2.1.1 主线。
- 迁移时逐项处理 API、Nucleus 资产、SDK 和硬件依赖，不以“脚本能解析”代替运行验证。
- 当前树中不存在的 `stable_refs`、`high_risk`、`superseded` 等历史范围不能从现有文件推断或还原。

详细缺口见 [依赖缺口与迁移边界](DEPENDENCY_GAPS.md)，来源边界见 [版本关系](VERSION_RELATIONSHIP.md)。
