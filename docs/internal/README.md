# Internal maintainer records / 内部维护记录

These files preserve release audits, migration decisions, static inventories, and historical implementation notes. They support maintainers and reviewers; they are **not** the recommended path for installing or using the project.

这些文件用于保留发布审计、迁移决策、静态清单和历史实现记录，主要服务于维护者与审核者，**不是**用户安装和使用项目的推荐入口。

For maintained user documentation, choose a language:

用户文档请按语言访问：

- [English documentation](../README.md)
- [中文文档](../README.zh-CN.md)

## Contents / 内容分类

- Architecture and version history / 架构与版本历史: `ARCHITECTURE.md`, `VERSION_*.md`, `OWTBENCH_VERSION_INDEX.md`
- Release and dependency audits / 发布与依赖审计: `OPEN_SOURCE_READINESS.md`, `DEPENDENCY_GAPS.md`, `THIRD_PARTY_BOUNDARIES.md`
- Static inventories / 静态清单: `ENTRYPOINT_MATRIX.md`, `EXTERNAL_PATH_REFERENCES.md`, `MISSING_OR_EXTERNAL_ASSETS.md`, `PATH_CONFIRMATION.md`
- Migration records / 迁移记录: `REPOSITORY_REFACTOR.md`, `MAINLINE_GUIDE.md`, `LEGACY_GUIDE.md`

Generated records may be refreshed with the scripts documented in [`tools/repository`](../../tools/repository/README.md). Do not use a generated record as a substitute for the maintained guides under `docs/en/` and `docs/zh-CN/`.

生成记录可通过 [`tools/repository`](../../tools/repository/README.zh-CN.md) 中的脚本刷新。请勿用生成记录替代 `docs/en/` 和 `docs/zh-CN/` 下持续维护的用户指南。
