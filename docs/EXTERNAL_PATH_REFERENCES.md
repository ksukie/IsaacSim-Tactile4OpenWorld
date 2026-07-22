# 外部路径引用

静态扫描记录源码与项目说明中的明确外部环境路径和运行时临时路径，不导入项目模块，也不运行仿真。

当前支持的主要配置入口：

- `OWT_ASSET_ROOT`：legacy 触觉 USD 与标定数据根目录。
- `OWT_SDK_ROOT`：OpenWorldTactile SDK 的显式位置。
- `ISAACLAB_NUCLEUS_DIR`：Isaac Lab 提供的 Nucleus 资产根目录。
- `PYTHON_BIN`：需要子进程解释器时的可选覆盖。

完整扫描结果见 [`external_path_references.csv`](../tools/repository/external_path_references.csv)，逐项用途和迁移建议见 [`PATH_ADAPTATION_PLAN.csv`](../tools/repository/PATH_ADAPTATION_PLAN.csv)，已实现的配置机制见 [`PORTABILITY_ADAPTATIONS.csv`](../tools/repository/PORTABILITY_ADAPTATIONS.csv)。项目自有配置不再绑定来源机用户名；扫描到的 3 条 `/home/` 绝对路径均位于 `active-isaaclab-2.1/vendor/agilex-piper/` 上游内容中，仍作为第三方迁移提示登记。
