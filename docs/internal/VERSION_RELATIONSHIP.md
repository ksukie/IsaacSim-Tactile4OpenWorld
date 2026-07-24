# 版本关系

## Isaac Lab 运行基线

- [`active-isaaclab-2.1.1/`](../../active-isaaclab-2.1.1/)：Isaac Lab 2.1.1 对应的当前 OpenWorldTactile、UIPC 与 AgileX 主线。
- [`archive-isaaclab-2.3.2/`](../../archive-isaaclab-2.3.2/)：Isaac Lab 2.3.2 对应的 OpenWorldTactile/GelSight 集成路线。

两个容器均只包含对应基线的项目关联内容，不表示其中包含完整官方 Isaac Lab。本地运行时应为两条路线准备相互独立的环境。

## OpenWorldTactile 实验版本

V1 至 V6.2 共享 `packages/uipc/libuipc` 及同目录辅助模块。V6 系列会直接导入 V5.9、V5.7g 和其他公共模块，因此 `experiments/tactile-bench/` 保持为完整单元，不按实验版本拆分。

V3、V7、V8 主要是规划或说明阶段；V6.2 是当前文档的默认参考入口，但不替代其他实验脚本。完整关系见 [版本谱系](VERSION_LINEAGE.md) 与 [OpenWorldTactileBench 版本索引](OWTBENCH_VERSION_INDEX.md)。
