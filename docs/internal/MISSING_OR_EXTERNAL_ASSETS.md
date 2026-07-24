# 缺失或外部资产

## 发布载荷完整性

当前工作树相对 `tools/repository/FINAL_MANIFEST.csv` 缺少 43 个主线文件，其中包括 Franka/GelSight Single Adapter 网格和 UIPC 示例网格。它们与下述“允许缺失且具有内部 fallback”的 USDA 占位引用不是同一类问题。发布者必须先恢复运行必需文件，或在逐项核实用途、来源和许可证后有意调整载荷；不能把清单不一致解释为外部运行时依赖。

根 `.gitignore` 的 `*.obj` 规则会忽略资产网格。若这些 OBJ 应进入公开仓库，应添加范围精确的否定规则，并确认提交实际包含文件内容。

## USDA 文本引用

USDA 文本引用静态检查结果：

- 已在最终工作树中解析：2 条。
- 明确外部引用：0 条。
- 允许缺失且声明内部 fallback：2 条。
- 意外缺失的相对引用：0 条。

`UIPC_Pad.usda` 中 `./uipc/membrane.tet` 和 `./uipc/membrane_surface.obj` 是可选占位路径；文件同时声明 `uipc:external_mesh_required = 0` 和内部 `membrane_sim_mesh` fallback，因此单独列为可选引用，不判定为重构破坏。

Legacy 文档和脚本中的 `${ISAACLAB_NUCLEUS_DIR}` 资产属于外部环境依赖，详见外部路径清单。
