# 缺失或外部资产

USDA 文本引用静态检查结果：

- 已在最终工作树中解析：2 条。
- 明确外部引用：0 条。
- 允许缺失且声明内部 fallback：2 条。
- 意外缺失的相对引用：0 条。

`UIPC_Pad.usda` 中 `./uipc/membrane.tet` 和 `./uipc/membrane_surface.obj` 是可选占位路径；文件同时声明 `uipc:external_mesh_required = 0` 和内部 `membrane_sim_mesh` fallback，因此单独列为可选引用，不判定为重构破坏。

Legacy 文档和脚本中的 `${ISAACLAB_NUCLEUS_DIR}` 资产属于外部环境依赖，详见外部路径清单。
