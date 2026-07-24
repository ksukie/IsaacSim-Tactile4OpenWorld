<p align="right">
  <a href="../../en/help/faq.md">English</a> · <strong>简体中文</strong>
</p>

# 常见问题

## 这是独立的 Isaac Sim 或 Isaac Lab 发行版吗？

不是。项目会把扩展安装到外部 Isaac Lab 2.1.1 / Isaac Sim 4.5 环境中。请先完成[安装指南](../getting-started/installation.md)。

## 第一次应该运行哪个脚本？

运行[快速开始](../getting-started/quick-start.md)中的低分辨率 V1 用例。它会实际求解 UIPC 柔性膜并生成可检查的小型输出，不需要 V6.2 的契约链。

## 这里的“开放世界”是什么意思？

它表示将触觉实验扩展到不同机器人、传感器模型、物体和接触条件；不表示项目实现了通用世界模型、开放词汇推理或任意场景下的零样本控制。

## 为什么同时有 2.1.1 和 2.3.2 目录？

它们对应不同 Isaac Lab 基线。`active-isaaclab-2.1.1/` 是持续维护的 OpenWorldTactile/UIPC 主线，`archive-isaaclab-2.3.2/` 保留旧 GelSight/SDK 路线。两者不能混装到同一环境。

## V1、V5、V6.2 是软件发布版本吗？

不是。它们是触觉基准的研究阶段。数字更大不代表脚本更稳定或更易运行，详见[实验版本谱系](../reference/experiment-lineage.md)。

## 为什么 V6.2 需要之前的 V5.7d/e/f 输出？

V6.2 使用冻结且经过验证的形变契约与估计器。前置运行用于验证坐标系刚体运动消除、柔性膜拓扑/面积、法向响应和重复性；脚本会主动拒绝不完整或未通过的契约。

## 触觉力是牛顿吗？

默认不是。V1 输出 `sim_constitutive_force`，冻结估计器使用 TU 量值。V6.2 还会单独保存物理 UIPC 反力/实际耦合数组，在元数据明确时其单位为 N 或 N·m。不要只根据文件名推断单位，应读取元数据。

## 不编译 UIPC 能使用项目吗？

部分光学/标记点代码和非 UIPC 资产可通过 `./run.sh --install` 安装，但文档中的 V1/V6.2 工作流和 UIPC 任务变体需要 `./run.sh --install all`。任务扩展也声明了 UIPC 集成依赖。

## 支持 Windows 吗？

Isaac 上游产品和 libuipc 各自有 Windows 路线，但本仓库的端到端安装目前只记录 Linux/Bash。尚未发布经过测试的 PowerShell 包装器/工具链矩阵。

## 仓库包含预训练策略吗？

仓库包含智能体配置，不包含覆盖全部任务且经过验证的策略集。一个历史不透明检查点因缺少训练来源和许可证而被主动排除。请使用对应任务/后端自行训练，或提供兼容且许可证明确的检查点。

## 包含物理相机 SDK 吗？

不包含。由于无法验证再分发授权，历史供应商二进制已移除。请从授权来源取得 SDK，并仅为受影响的历史工作配置 `OWT_SDK_ROOT`。

## 可以连接真实机器人或相机硬件吗？

部分归档代码引用了硬件/SDK 流程，但项目没有安全认证。请先隔离检查和测试，在代码库外配置物理限位与急停流程，绝不能把静态检查解释为硬件验证。

## 结果写到哪里？

每个实验有自己的默认目录，通常位于 `/tmp`。请显式设置 `--output_dir`，并让 `--workspace_dir` 与其分离。详见[数据与输出](../guides/data-and-outputs.md)。

## 为什么 Isaac Lab 训练器找不到任务 ID？

外部启动器必须在启动 Isaac Sim 后导入 `openworldtactile_tasks`。未经修改的上游启动器只导入上游任务注册，详见[连接 Isaac Lab 启动器](../guides/tasks-and-training.md#连接-isaac-lab-启动器)。

## 哪些文档是正式用户文档？

持续维护的用户指南位于 `docs/en/` 与 `docs/zh-CN/`。如果说明与源码冲突，以当前源码和脚本 `--help` 为准。`docs/internal/` 和实验旁的笔记用于来源/维护记录，不是安装指南。

## 可以商业使用吗？

项目原创代码采用 BSD-3-Clause，但仓库包含多种许可证，特定绑定、资产和第三方子树还适用 GPL/AGPL 等条款。请阅读[第三方通知](../../../THIRD_PARTY_NOTICES.zh-CN.md)，并就具体分发方案咨询专业法律意见；本文档不构成法律建议。

## 如何引用？

项目元数据使用 [`CITATION.cff`](../../../CITATION.cff)，具体方法论文见[研究引用](../../../CITATIONS.zh-CN.md)。研究中应同时引用本项目和实际使用仿真方法的原始论文。

## 如何反馈有效问题？

使用[故障排查](troubleshooting.md#提交可复现问题)末尾的诊断模板。请提供准确版本、命令、回溯和最小复现，不要附加凭据或专有 SDK 文件。
