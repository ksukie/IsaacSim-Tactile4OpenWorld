<p align="right">
  <a href="../../en/reference/compatibility.md">English</a> · <strong>简体中文</strong>
</p>

# 兼容性与验证状态

项目当前只以一组主线组合为目标。其他组合可能可用，但在完成复现和文档记录前，不属于用户指南支持范围。

## 主线矩阵

| 层级 | 目标/参考 | 状态 |
|---|---|---|
| 操作系统 | Ubuntu 22.04 同类 Linux 环境 | 来自 Isaac Lab 2.1.1 / Isaac Sim 4.5 上游推荐组合；项目运行矩阵尚未自动化 |
| Isaac Sim | 4.5.0 | 固定的主线目标 |
| Isaac Lab | 2.1.1 | 固定的主线目标 |
| Python | 3.10 | Isaac Sim 4.5 与项目包元数据要求 |
| libuipc | 仓库内源码标记为 0.9.0-alpha | 在本机编译；不分发可移植二进制 |
| UIPC 构建工具链 | 内置指南要求 CMake 3.26+、CUDA 12.4+、vcpkg、C++20 工具链 | 必须在目标机验证 |
| GPU/驱动 | NVIDIA RTX 级 GPU，以及同时兼容 Isaac Sim 和所选 CUDA Toolkit 的驱动 | 遵循 NVIDIA 对应版本要求 |

官方参考：

- [Isaac Lab 2.1.1 使用 Isaac Sim 二进制的安装指南](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/setup/installation/binaries_installation.html)
- [Isaac Sim 4.5 系统与驱动要求](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html)

2.1.1 上游指南记录了 Ubuntu 22.04 与 Isaac Sim 4.5 的测试组合。旧项目笔记提到 Ubuntu 24.04 迁移环境，但本次发布不将该记录表述为经过测试的公开基线。

## “已验证”的含义

| 层级 | 证据 | 不能证明 |
|---|---|---|
| 静态 | 文件存在、Python AST 可解析、本地文档链接有效、资产引用已登记 | 导入、原生库加载、GPU 执行、数值正确性 |
| 导入 | 包和 `uipc` 可在一个 Python 环境导入 | 仿真器启动或物理正确 |
| 单元 | 低依赖数值测试通过 | Isaac/PhysX/UIPC 集成 |
| 冒烟 | 短 V1 或任务运行能够初始化和推进 | 完整场景验收或可复现性 |
| 场景 | 完整命令完成，且验收器在已记录环境中通过 | 其他 GPU、驱动或参数组合支持 |
| 硬件 | 指定硬件/SDK 组合在安全控制下测试 | 安全认证或通用设备兼容性 |

仓库提供静态检查，但不得将其描述为运行验证。公开文档提供导入、单元、V1 冒烟和 V6.2 场景检查命令，实际结果仍取决于用户的外部环境。

## 版本路线

| 目录 | 基线 | 用途 | 支持状态 |
|---|---|---|---|
| `active-isaaclab-2.1.1/` | Isaac Lab 2.1.1 | 当前包、UIPC、任务和实验 | 持续维护的主线 |
| `archive-isaaclab-2.3.2/` | Isaac Lab 2.3.2 | 历史 GelSight/SDK 复现与迁移 | 归档；不提供常规安全/运行维护 |

不要在同一 Python 环境中混合两条路线。OpenWorldTactile `V1`–`V6.2` 是 2.1.1 路线内部的实验阶段，与 Isaac Lab `2.3.2` 归档标签没有先后关系。

## 已知外部边界

仓库不提供：

- Isaac Sim 或完整 Isaac Lab 仓库；
- 预编译的 `uipc` 原生模块；
- NVIDIA 驱动、CUDA Toolkit、vcpkg、编译器或系统包；
- 经授权的历史 OpenWorldTactile 相机 SDK；
- 归档实验引用的全部 Nucleus/Factory/GelSight 资产；
- 覆盖所有任务的预训练策略；
- 硬件安全认证。

主线包含项目 USD 资产、libuipc 源码以及采用各自许可证的部分第三方源码/资产。详见[第三方通知](../../../THIRD_PARTY_NOTICES.zh-CN.md)。

## 平台说明

### Linux

Linux/Bash 是文档路线。包装脚本、`/tmp` 默认值、Shell 循环和内置 libuipc Linux 说明都假定类 Unix 环境。

### Windows

Isaac Sim 与 Isaac Lab 上游提供 Windows 工作流，libuipc 也保留上游 Windows 构建说明，但本仓库尚未将这些步骤整合为经过验证的 PowerShell 包装流程。原生 Windows 用户需要自行适配安装和命令；贡献支持时请提供完整工具链信息。

### 容器与远程/无界面系统

Isaac Sim 上游支持容器与无界面部署，但本仓库不提供项目容器镜像。需要自行处理 GPU 透传、EULA、可写输出/UIPC 工作目录挂载、着色器缓存，以及资产和 vcpkg 依赖可能需要的网络访问。

## 更改固定版本

更改 Isaac Sim、Isaac Lab、Python、CUDA 或 libuipc 时，应按迁移处理：

1. 创建独立环境；
2. 从源码重建 UIPC；
3. 运行导入与低依赖测试；
4. 运行 V1 冒烟测试；
5. 运行 V6.2 前重新生成并验收 V5.7f 契约；
6. 运行场景验收并记录完整矩阵；
7. 同步更新中英文兼容性与安装文档。

不能仅凭一次导入成功或截断运行就扩大支持矩阵。
