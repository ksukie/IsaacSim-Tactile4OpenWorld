<p align="right">
  <a href="../../en/guides/legacy.md">English</a> · <strong>简体中文</strong>
</p>

# 历史 Isaac Lab 2.3.2 路线

`archive-isaaclab-2.3.2/` 保留旧版 GelSight、RGB/力场、Franka 和相机 SDK 集成路线，用于来源追踪和迁移。它不是当前主线，也不是完整的 Isaac Lab 仓库。

## 仅在以下情况使用

- 复现与归档脚本绑定的结果；
- 研究旧版 OpenWorldTactile/GelSight 传感器集成；
- 将某个具体方法或资产迁移到受维护的基线；
- 使用独立取得合法许可证的历史相机 SDK。

新的 UIPC 触觉基准工作应从 `active-isaaclab-2.1.1/` 开始。

## 目录内容

| 路径 | 用途 |
|---|---|
| `packages/contrib/` | 历史 `isaaclab_contrib` 触觉传感器集成 |
| `packages/assets-patches/` | 少量历史资产配置补丁 |
| `experiments/franka/current/` | Franka 接触、SDF、RGB、剪切和力场实验 |
| `experiments/sensors/` | 单个 OpenWorldTactile/GelSight 传感器探针 |
| `experiments/rgb-pipeline/` | RGB 与 fxyz 组合流程 |
| `hardware-sdk/` | 外部 SDK 的记录型挂载位置；不分发 SDK 二进制 |
| `notes/` | 未维护的内部研究记录，不是用户安装说明 |

历史路线不存在代表全部用途的唯一入口脚本。

## 外部要求

需要独立准备：

- 单独的 Isaac Lab 2.3.2 及匹配的 Isaac Sim 环境；
- 目标脚本所需的 `isaaclab_contrib`/资产修改；
- 通过 Nucleus 或本地映射提供的 GelSight R1.5 与 Factory 资产；
- 当目标脚本导入时，从授权渠道取得且与平台兼容的 OpenWorldTactile 相机 SDK；
- 兼容的原生库、相机驱动和硬件。

不要把 2.1.1 和 2.3.2 路线安装到同一环境后假定 API 可互换。

## 配置外部资产

归档源码支持时，可设置：

```bash
export OWT_ASSET_ROOT=/absolute/or/nucleus/path/to/openworldtactile-assets
export OWT_SDK_ROOT=/absolute/path/to/hardware-sdk/openworldtactile
```

默认 SDK 挂载位置为：

```text
archive-isaaclab-2.3.2/hardware-sdk/openworldtactile/
```

请阅读双语 [SDK 边界说明](../../../archive-isaaclab-2.3.2/hardware-sdk/README.md)。除非获得可验证的再分发授权，否则不得把已移除的供应商 DLL/SO 恢复到公开仓库。

## 恢复流程

1. 创建干净、独立的 Isaac Lab 2.3.2 环境。
2. 选择一个目标脚本，并在[内部入口清单](../../internal/ENTRYPOINT_MATRIX.md)中检查其导入和资产。
3. 仅将所需的 `packages/contrib/` 和 `packages/assets-patches/` 内容迁入外部项目或兼容扩展结构，不要整体覆盖上游仓库。
4. 配置 `OWT_ASSET_ROOT`，仅在确有需要时配置 `OWT_SDK_ROOT`。
5. 连接相机或硬件前，先用 2.3.2 解释器运行目标脚本的 `--help`。
6. 先验证不连接硬件的纯仿真用例。
7. 记录所有人工补丁、外部资产修订、SDK 版本、平台和命令。

归档目录没有标准安装器，也不包含完整上游源码，因此具体迁移步骤取决于目标脚本。静态解析成功不能证明恢复后的运行时正确。

## 操作安全

归档代码可能访问相机、USB 设备、原生库或机器人相关接口。连接物理设备前应在隔离环境中检查。不要把该研究归档当作硬件安全层。

## 迁移建议

- 每次只向新主线分支迁移一个功能。
- 用显式配置替换硬编码的 Nucleus/SDK 路径。
- 重新验证坐标系、单位、图像格式和 Isaac Lab API 变化。
- 保留原始来源与许可证通知。
- 任何迁移后公开的工作流都应补充持续维护的中英文用户文档。
