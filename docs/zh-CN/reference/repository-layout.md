<p align="right">
  <a href="../../en/reference/repository-layout.md">English</a> · <strong>简体中文</strong>
</p>

# 仓库结构

## 顶层

```text
IsaacSim-Tactile4OpenWorld/
├── README.md / README.zh-CN.md
├── active-isaaclab-2.1.1/
├── archive-isaaclab-2.3.2/
├── docs/
├── tools/repository/
├── CONTRIBUTING*.md、SECURITY*.md、CITATIONS*.md
├── LICENSE、NOTICE、THIRD_PARTY_NOTICES*.md
└── CITATION.cff
```

| 位置 | 面向对象 | 内容 |
|---|---|---|
| 根 README | 所有用户 | 项目概览与文档最短入口 |
| `docs/en/`、`docs/zh-CN/` | 用户 | 持续维护、相互镜像的用户指南 |
| `active-isaaclab-2.1.1/` | 用户/研究人员 | 当前可安装包、实验、资产与数据工具 |
| `archive-isaaclab-2.3.2/` | 迁移研究人员 | 带外部依赖边界的历史路线 |
| `docs/internal/`、`tools/repository/` | 维护者 | 生成型清单、发布审计记录与静态检查 |

## 当前主线

```text
active-isaaclab-2.1.1/
├── run.sh
├── packages/
│   ├── core/       -> openworldtactile
│   ├── assets/     -> openworldtactile_assets + USD/数据
│   ├── tasks/      -> openworldtactile_tasks
│   └── uipc/       -> openworldtactile_uipc + 内置 libuipc
├── experiments/
│   ├── tactile-bench/
│   ├── manipulation/
│   ├── simulation-prototypes/
│   └── benchmarks/
├── tools/data/
└── vendor/agilex-piper/
```

### 包

- `packages/core/openworldtactile/`：传感器类和 Taxim/FOTS/FEM 方法实现。
- `packages/assets/openworldtactile_assets/`：Python 配置和包含机器人/传感器 USD 的 `data/`。
- `packages/tasks/openworldtactile_tasks/`：环境实现、Gym 注册与智能体配置。
- `packages/uipc/openworldtactile_uipc/`：Isaac/UIPC 集成。
- `packages/uipc/libuipc/`：内置第三方源码，使用自身许可证与上游文档。

### 实验

- `tactile-bench/`：带版本标签的 UIPC 触觉研究、本地 API、离线估计器、验收器和保留研究记录。
- `manipulation/`：针对拾取、放置、插孔、摩擦和记录的场景脚本。
- `simulation-prototypes/`：聚焦单一传感器/方法的集成检查。
- `benchmarks/`：性能对比工具。

大量触觉基准文件不是相互独立的包。它们使用同目录导入与共享资产，因此不支持只复制单个脚本到其他位置。

### 数据工具

`tools/data/` 包含独立的 HDF5 图像/触觉查看器和导出器，只处理已完成文件，本身不采集数据。

### 第三方目录

`vendor/agilex-piper/` 是上游集成内容。不要把其中的 README、路径或构建说明当作 OpenWorldTactile 用户文档；除非是保留来源的明确上游补丁，否则应避免修改。

## 历史归档

```text
archive-isaaclab-2.3.2/
├── VERSION
├── packages/{contrib,assets-patches}/
├── experiments/{basics,franka,rgb-pipeline,sensors}/
├── hardware-sdk/
├── notes/
└── tools/
```

`active-isaaclab-2.1.1/run.sh` 不会安装该归档。详见[历史路线](../guides/legacy.md)。

## 文档

```text
docs/
├── README.md / README.zh-CN.md
├── en/
│   ├── getting-started/
│   ├── guides/
│   ├── reference/
│   └── help/
├── zh-CN/              # 与 en/ 主题完全镜像
├── internal/           # 维护者/生成型记录
└── media/
```

每篇持续维护的用户指南都有英文与简体中文版本，并在顶部提供语言切换。内部发布记录与保留的实验笔记不属于用户指南契约。

## 生成文件应放在哪里

| 文件类型 | 推荐位置 |
|---|---|
| 实验数组/JSON/预览 | 显式 `active-isaaclab-2.1.1/outputs/<run>/` 或外部数据目录 |
| UIPC 工作目录 | 每次运行独立的临时目录 |
| 强化学习日志/检查点 | 包源码以外的训练器日志目录 |
| 构建产物 | 已忽略且不提交的包构建目录 |
| 公开文档图片 | 许可证明确且有意纳入版本控制时放入 `docs/media/` |

不要将生成结果写入 `packages/**/data/`；该目录用于版本化资产。

## 查找内容

在仓库根目录运行：

```bash
# 按名称查找脚本或资产
rg --files | rg 'OpenWorldTactile_v6_2|GelSight_Mini'

# 查找任务注册
rg -n 'gym.register|OpenWorldTactile-' active-isaaclab-2.1.1/packages/tasks

# 查找命令行参数
rg -n 'add_argument.*output_dir' active-isaaclab-2.1.1/experiments

# 查找公开类定义
rg -n '^class ' active-isaaclab-2.1.1/packages/{core,uipc}
```

完整静态脚本列表见[内部入口矩阵](../../internal/ENTRYPOINT_MATRIX.md)。
