# 自研与第三方边界

## OpenWorldTactile 集成层

以下位置主要承载项目的组织、实验和集成贡献，但其中带有既有版权/SPDX 头的文件仍遵循该文件自己的许可证：

- `active-isaaclab-2.1/packages/{core,assets,tasks}/` 的 OpenWorldTactile 集成代码；
- `active-isaaclab-2.1/packages/uipc/openworldtactile_uipc/` 与构建集成；
- `active-isaaclab-2.1/experiments/` 中的 OpenWorldTactile 实验路线；
- `archive-isaaclab-2.3/experiments/`、历史笔记和项目适配；
- 根文档与 `tools/repository/` 发布工具。

## 明确的上游或第三方范围

- Isaac Lab、ORBIT、Isaac Sim 与 Omniverse API；
- `packages/uipc/libuipc/` 及其 MuDA、TetGen、Octree、Catch2 子组件；
- `vendor/agilex-piper/` 及内嵌 MoveIt 快照；
- Taxim、FOTS 和 ManiSkill-ViTac2025 衍生实现；
- GelSight Mini、Franka 和 Piper 衍生模型/机器人资产；
- 外部 Nucleus/Factory 资产及用户自行取得的相机 SDK。

完整的逐项来源、许可证和本地许可证文件见根目录 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。学术引用见 [`CITATIONS.md`](../CITATIONS.md)。

## 不随公开仓库分发

- 权利未确认的历史相机 SDK 与 Sonix 原生库；
- 未声明许可证的 tactile test shapes；
- 无训练 provenance 与模型许可证的旧 `IK_old.pt`；
- 构建产物、生成的 `*.egg-info`、日志和私有配置。

项目命名统一不会改变第三方的真实产品名、版权、许可证、协议或商标归属。将第三方文件移动到 OpenWorldTactile 目录也不会把它变为项目原创内容。
