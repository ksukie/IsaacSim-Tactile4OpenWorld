<p align="right">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

# 仓库静态校验工具

本目录包含静态发布清单、报告和校验脚本，供维护者使用，不属于任一 Isaac Lab 运行时载荷。

## 生成记录

| 文件 | 用途 |
|---|---|
| `FINAL_MANIFEST.csv` | 两个版本运行时载荷中每个文件的路径、大小和 SHA-256 |
| `PORTABILITY_ADAPTATIONS.csv` | SDK、Python 和外部资产的可配置入口 |
| `ENTRYPOINT_INVENTORY.csv` | Python 入口、导入和资产引用的静态清单 |
| `PATH_ADAPTATION_PLAN.csv` | 外部/运行时路径引用的分类与处理方案 |
| `external_path_references.csv` | 外部路径扫描结果 |
| `usda_reference_check.csv` | USDA 相对引用与外部引用检查 |
| `markdown_link_check.csv` | 持续维护的顶层文档本地链接检查 |

## 命令

在仓库根目录使用普通 Python 3 解释器运行：

```bash
python tools/repository/audit_open_source.py
python tools/repository/build_static_navigation.py
python tools/repository/finalize_layout.py
```

这些检查不会导入项目包、安装依赖、编译 UIPC 或启动 Isaac Sim。

只有确认所有运行时载荷变化都是有意修改后，才重写载荷清单：

```bash
python tools/repository/finalize_layout.py --write-manifest
```

`build_static_navigation.py` 会刷新生成清单和 [`docs/internal/ENTRYPOINT_MATRIX.md`](../../docs/internal/ENTRYPOINT_MATRIX.md)。`finalize_layout.py` 会校验清单哈希、AST 语法、文档链接、USDA 引用、路径冲突、压缩文件和旧名称残留。`audit_open_source.py` 会检查发布政策、许可证、包元数据、疑似凭据、生成目录及禁止进入公开载荷的原生/不透明文件。
