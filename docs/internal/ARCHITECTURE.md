# IsaacSim-Tactile4OpenWorld 仓库架构

## 总体结构

IsaacSim-Tactile4OpenWorld 同时保存一条 OpenWorldTactile 当前研究主线和一条独立 legacy 路线。两者依赖不同 Isaac Lab 基线，不能作为同一套运行环境直接混用。

目录映射、路径兼容处理与静态验证范围见 [重构记录](REPOSITORY_REFACTOR.md)。

| 顶层位置 | 职责 | 是否独立运行环境 |
|---|---|---|
| [`active-isaaclab-2.1.1/`](../../active-isaaclab-2.1.1/) | OpenWorldTactile、UIPC、AgileX/Piper 当前主线 | 是，依赖 Isaac Lab 2.1.1 |
| [`archive-isaaclab-2.3.2/`](../../archive-isaaclab-2.3.2/) | OpenWorldTactile/GelSight 传感器、外部 SDK 边界与历史实验 | 是，依赖 Isaac Lab 2.3.2 |
| [`docs/`](../) | 中英双语用户指南与内部维护记录 | 否 |
| [`tools/repository/`](../../tools/repository/) | 最终清单和纯静态校验工具 | 否 |

```text
<repository>/
├─ active-isaaclab-2.1.1/
│  ├─ packages/{core,assets,tasks,uipc}/
│  ├─ experiments/{tactile-bench,manipulation,simulation-prototypes,benchmarks}/
│  ├─ tools/data/
│  └─ vendor/agilex-piper/
├─ archive-isaaclab-2.3.2/
│  ├─ packages/{contrib,assets-patches}/
│  ├─ experiments/{franka,sensors,rgb-pipeline,basics}/
│  ├─ hardware-sdk/README.md
│  └─ notes/
├─ docs/{en,zh-CN,internal}/
└─ tools/repository/
```

## 两条依赖链

主线：

```text
Isaac Lab 2.1.1 + Isaac Sim
  -> openworldtactile / openworldtactile_assets / openworldtactile_tasks
  -> openworldtactile_uipc -> libuipc
  -> OpenWorldTactileBench 与其他 demos
```

legacy：

```text
Isaac Lab 2.3.2 + Isaac Sim
  -> isaaclab_contrib.sensors.openworldtactile_sensor
  -> GelSight RGB / OpenWorldTactile RGB
  -> externally obtained hardware-sdk/openworldtactile
  -> fxyz 与实验输出
```

legacy 的触觉 USD 和标定数据由 `OWT_ASSET_ROOT` 指定；默认映射到 `${ISAACLAB_NUCLEUS_DIR}/OpenWorldTactile`。这使外部资产位置与项目内部命名解耦。

## 版本边界

- Isaac Lab `2.1.1` 与 `2.3.2` 是运行基线。
- OpenWorldTactile `V1–V6.2` 是主线内部的实验阶段，不是 Isaac Lab 版本号。
- `legacy` 表示独立保留的集成路线，不表示它是主线的后续版本。

## 维护规则

- 仓库与发行名称使用 IsaacSim-Tactile4OpenWorld；项目自有 Python 包、目录、脚本、配置和技术名称统一使用 OpenWorldTactile。
- Isaac Lab、Isaac Sim、UIPC、libuipc、GelSight、CUDA 和 PyTorch 等真实外部名称保持其本来含义。
- 外部 SDK、Nucleus 资产和解释器位置通过 `OWT_*` 或既有通用环境变量配置；未确认授权的 SDK 不进入公开载荷。
- 根 BSD-3-Clause 只覆盖原创贡献，第三方边界以 `THIRD_PARTY_NOTICES.md` 为准。
- 修改任一载荷文件后，应有意更新 `FINAL_MANIFEST.csv`，随后重新执行最终静态校验。
