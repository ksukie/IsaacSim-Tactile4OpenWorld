# IsaacSim-Tactile4OpenWorld 文档中心

本目录说明 IsaacSim-Tactile4OpenWorld 仓库中的 OpenWorldTactile 主线与 legacy 路线、目录边界和发布准备状态；它不替代 Isaac Lab、Isaac Sim、UIPC 或相机 SDK 的官方文档。

## 第一次阅读

- [仓库架构](ARCHITECTURE.md)：顶层分区、两条依赖链和目录边界。
- [重构记录](REPOSITORY_REFACTOR.md)：目录映射、路径兼容处理与静态验证范围。
- [Isaac Lab 2.1.1 当前主线指南](MAINLINE_GUIDE.md)：OpenWorldTactile/UIPC 的代码位置、参考入口和环境边界。
- [Isaac Lab 2.3.2 legacy 指南](LEGACY_GUIDE.md)：OpenWorldTactile/GelSight 的用途入口和迁移边界。
- [脚本入口矩阵](ENTRYPOINT_MATRIX.md)：两个版本容器中 153 个 Python 脚本的角色、静态依赖和资产导航。

## 版本关系

- [版本关系](VERSION_RELATIONSHIP.md)：两个 Isaac Lab 基线与 OpenWorldTactile 实验版本之间的关系。
- [版本谱系](VERSION_LINEAGE.md)：两条路线及 OpenWorldTactile V1–V6.2 的演进与直接依赖。
- [版本矩阵](VERSION_MATRIX.md)：版本维度的紧凑对照。
- [OpenWorldTactileBench 版本索引](OWTBENCH_VERSION_INDEX.md)：现有实验脚本版本索引。

## 依赖与内容边界

- [依赖缺口](DEPENDENCY_GAPS.md)：完整 Isaac Lab 本体、外部资产、构建产物和目标机环境等迁移条件。
- [外部路径引用](EXTERNAL_PATH_REFERENCES.md)：外部资产和运行时输出路径分类。
- [缺失或外部资产](MISSING_OR_EXTERNAL_ASSETS.md)：USDA 可选引用及外部环境资产。
- [自研与第三方边界](THIRD_PARTY_BOUNDARIES.md)：项目代码、上游依赖和第三方内容范围。
- [开源发布审计](OPEN_SOURCE_READINESS.md)：许可证、第三方、二进制和资产审查的结果与残余风险。
- [第三方通知](../THIRD_PARTY_NOTICES.md)：组件路径、上游来源和许可证位置。
- [研究引用](../CITATIONS.md)：实现方法对应的原始论文。

## 最终校验

- [路径与内容确认](PATH_CONFIRMATION.md)：3,749 个版本载荷文件、最终 SHA-256 和静态检查结果。
- 最终清单、静态报告与可复现校验器位于 [`tools/repository/`](../tools/repository/)。

## 文档维护

- 移动路径时，同时更新本索引、根 README、对应使用指南和静态导航工具。
- 运行结果、硬件兼容性和外部资产可用性必须与“静态检查通过”分开记录。
- 未确认许可证或再分发权利的内容不得进入公开载荷；新增第三方内容时同步更新通知和引用。
