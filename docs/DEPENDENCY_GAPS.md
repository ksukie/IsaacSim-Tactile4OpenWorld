# 依赖缺口与迁移边界

## 结论

- 最终工作树包含 3,749 个版本载荷文件：主线 3,691 个，legacy 58 个。
- 两个版本容器都是项目关联内容，不是完整的官方 Isaac Lab 本体。
- 本轮只验证文件、哈希、Python AST、文档链接和静态资产引用；没有验证运行环境。

## 需要外部准备的条件

| 条件 | 适用路线 | 当前状态 | 迁移建议 |
|---|---|---|---|
| 完整 Isaac Lab 本体 | 两条路线 | 仓库内未提供 | 分别准备 2.1.1 与 2.3.2 环境，不要混装 |
| Isaac Sim / Omniverse 运行时 | 两条路线 | 仓库内未提供 | 按对应 Isaac Lab 基线选择版本 |
| CUDA、GPU 驱动、PyTorch ABI | 两条路线 | 未做运行验证 | 在目标机单独确认组合兼容性 |
| libuipc 构建产物 | 主线 | 源码存在，目标机构建状态未知 | 在目标环境按项目说明重新构建 |
| GelSight R1.5 触觉 USD 与标定数据 | legacy | 外部环境资产 | 设置 `OWT_ASSET_ROOT` 或提供默认 Nucleus 映射 |
| Factory 示例资产 | legacy | 外部环境资产 | 由 `${ISAACLAB_NUCLEUS_DIR}/Factory` 或等价映射提供 |
| 相机 SDK 依赖与驱动 | legacy | 不随公开仓库分发，仅保留外部挂载说明 | 从权利人处合法取得后，在目标平台单独验证 |

## 可移植性入口

| 位置 | 配置项 | 当前行为 |
|---|---|---|
| `check_taxim_sim.py` | `OWT_SDK_ROOT` | 默认查找外部放入归档容器的 `hardware-sdk/openworldtactile`，可显式覆盖 |
| `openworldtactile_finger_sensor.py` | `OWT_SDK_ROOT` | 默认从归档根目录查找外部 SDK，可显式覆盖 |
| `OpenWorldTactile_v2_8.py` | `PYTHON_BIN` | 默认使用 `sys.executable`，可显式覆盖 |
| `visuotactile_sensor_cfg.py` | `OWT_ASSET_ROOT` | 默认使用 OpenWorldTactile Nucleus 位置，可映射到任意外部资产根 |

完整登记见 [`PORTABILITY_ADAPTATIONS.csv`](../tools/repository/PORTABILITY_ADAPTATIONS.csv)。当前外部和运行时路径见 [`external_path_references.csv`](../tools/repository/external_path_references.csv) 与 [`PATH_ADAPTATION_PLAN.csv`](../tools/repository/PATH_ADAPTATION_PLAN.csv)。

## 运行验证边界

静态 AST 成功只说明 Python 源码可解析，不证明以下事项：

- Isaac Lab 或 Isaac Sim API 与目标版本完全兼容；
- USD、Nucleus、CUDA 或 UIPC 在目标机可用；
- 外部取得的供应商相机库、USB 设备或驱动能够加载；
- 仿真、标定、数据采集或力估计结果正确。

这些事项需要在独立环境验证，不属于本次名称与目录重构范围。
