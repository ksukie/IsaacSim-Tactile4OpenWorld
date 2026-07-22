# 仓库静态校验工具

本目录只保存最终 IsaacSim-Tactile4OpenWorld 工作树所需的清单、报告和纯静态工具，不属于任一 Isaac Lab 版本载荷。

- `FINAL_MANIFEST.csv`：3,746 个最终版本载荷文件的相对路径、大小和 SHA-256。
- `PORTABILITY_ADAPTATIONS.csv`：SDK、Python 解释器和外部触觉资产的可配置入口。
- `ENTRYPOINT_INVENTORY.csv`：两个版本 `experiments/`（及当前版本 `tools/`）下 153 个 Python 文件的静态入口、导入和资产引用清单。
- `PATH_ADAPTATION_PLAN.csv`：当前外部或运行时路径引用的逐项分类。
- `external_path_references.csv`：外部环境与运行时路径扫描结果。
- `usda_reference_check.csv`：USDA 相对引用与外部引用检查结果。
- `markdown_link_check.csv`：顶层项目文档的本地链接检查结果。
- `build_static_navigation.py`：从当前源码重新生成入口矩阵、入口清单和路径分类。
- `finalize_layout.py`：按最终清单复核载荷哈希，并执行 AST、链接、USDA、路径冲突、压缩文件及遗留名称片段检查。
- `audit_open_source.py`：检查发布政策文件、关键许可证、包元数据、凭据形态、生成目录和禁止进入公开仓库的原生/不透明载荷。

常用静态命令：

```bash
py tools/repository/audit_open_source.py
py tools/repository/build_static_navigation.py
py tools/repository/finalize_layout.py
```

只有在确认最终载荷有意变更后，才重新写入清单：

```bash
py tools/repository/finalize_layout.py --write-manifest
```

这些工具不导入项目模块、不安装依赖、不编译，也不启动仿真。
