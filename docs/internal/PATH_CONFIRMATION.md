# IsaacSim-Tactile4OpenWorld 静态确认状态

> 当前状态：**未通过最终载荷清单校验**。本文记录 2026-07-22 工作树的实际检查结果；恢复清单缺失文件并完整运行 `finalize_layout.py` 后，本文件会由工具自动重写。

- 载荷清单：`tools/repository/FINAL_MANIFEST.csv`，预期 3,746 项。
- 当前可枚举载荷：3,703 项；主线 3,645 项、历史归档 58 项。
- 清单差异：主线缺少 43 项、额外文件 0 项。缺失分类见 `OPEN_SOURCE_READINESS.md` 与 `MISSING_OR_EXTERNAL_ASSETS.md`。
- 本轮改写的 12 个载荷文档，其大小和 SHA-256 已同步到清单；其他清单行未被降级或移除。
- Python 脚本静态 AST：153 个，解析错误 0。
- 根级、`docs/` 和仓库工具说明的本地 Markdown 链接：408 条，断链 0。
- 额外纳入子包用户入口后的维护文档链接：438 条，断链 0。
- `docs/en/` 与 `docs/zh-CN/`：各 14 个文件，相对路径差异 0。
- USDA 引用：4 条，意外缺失相对引用 0。
- 外部或运行时路径引用：279 条，见 `tools/repository/external_path_references.csv`。
- 压缩交付件：0；大小写路径冲突：0；禁止的旧名称片段：0。
- 最长绝对路径：305 个字符，对应 `active-isaaclab-2.1.1/vendor/agilex-piper/piper_ros/src/piper_moveit/moveit-1.1.11/moveit_planners/pilz_industrial_motion_planner/test/test_robots/prbt/test_data/unittest_joint_limits_aggregator_testdata/test_joint_limits_violate_position_max.yaml`。

这些检查不安装依赖、不编译 UIPC、不启动 Isaac Lab/Isaac Sim，也不连接硬件。载荷缺失问题解决前，不得引用旧版“文件缺失 0”的确认结论。
