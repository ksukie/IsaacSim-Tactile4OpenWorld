# 仓库重构记录

## 目标

本次重构将按来源堆叠的目录整理为稳定的研究项目结构：当前 UIPC 主线和历史 GelSight/SDK 路线在根目录并列，代码包、实验、工具、供应商内容和历史笔记各归其位。

## 当前布局

```text
active-isaaclab-2.1/       当前研究主线：UIPC、触觉抓取与 AgileX/Piper 集成
archive-isaaclab-2.3/      历史路线：GelSight、外部 SDK 边界与 Franka/RGB 实验
docs/                       项目说明和发布边界
tools/repository/           清单、入口导航和纯静态检查
```

版本目录直接位于仓库根目录；不再保留只包含一层版本目录的 `active/` 或 `archive/` 包装目录。

## 目录映射

| 原职责 | 当前位置 |
|---|---|
| 2.1.1 自研扩展与 UIPC 集成 | `active-isaaclab-2.1/packages/{core,assets,tasks,uipc}/` |
| 2.1.1 演示、抓取、原型与基准 | `active-isaaclab-2.1/experiments/` |
| 2.1.1 数据处理脚本 | `active-isaaclab-2.1/tools/data/` |
| 2.1.1 AgileX/Piper 上游集成 | `active-isaaclab-2.1/vendor/agilex-piper/` |
| 2.3.2 contrib 与资产补丁 | `archive-isaaclab-2.3/packages/` |
| 2.3.2 Franka、传感器和 RGB 实验 | `archive-isaaclab-2.3/experiments/` |
| 2.3.2 OpenWorldTactile 相机 SDK | 本体不分发；挂载说明在 `archive-isaaclab-2.3/hardware-sdk/README.md` |
| 2.3.2 历史实验笔记 | `archive-isaaclab-2.3/notes/` |

## 已同步内容

- 更新 `run.sh`、实验脚本、SDK 自动发现和测试中的相对路径。
- 更新根 README、架构、路线指南、版本关系和自动生成的入口/路径报告。
- 保留 Python 包与模块名，避免仅因文件系统整理改变导入 API。
- 删除已确认为空的旧 `active`、`archive`、`source`、`scripts` 和 `external` 目录。
- 开源审计后移出无授权 SDK/原生库、无许可证测试形状、无 provenance 模型权重和生成的包元数据。
- 补齐根许可证、NOTICE、第三方清单、引用、贡献、安全政策和自动发布审计。

## 验证范围

重构完成后执行的检查为：

- 3,749 个版本载荷文件与 SHA-256 清单一致；
- 153 个 Python 文件可被 AST 解析；
- 顶层项目文档本地 Markdown 链接有效；
- 4 个 USDA 相对引用均可解析；
- 未发现旧版本容器路径或历史项目名称片段。

这些检查不启动 Isaac Sim、不编译 libuipc、不安装依赖，也不连接 GPU 或相机。因此它们证明的是仓库结构和静态引用一致性，不是仿真、硬件或性能结果。

## 后续维护

目录移动必须先更新运行时路径，再更新本文档、根 README 和 `tools/repository/` 中的导航基线；最后重建清单并执行静态校验。不要为了整理目录而合并两条 Isaac Lab 基线，或将归档路线的依赖直接覆盖到当前主线。
