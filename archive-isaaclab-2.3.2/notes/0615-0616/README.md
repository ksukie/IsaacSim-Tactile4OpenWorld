# Franka tactile experiment archive / Franka 触觉实验归档

[English](#english) · [简体中文](#简体中文)

<a id="english"></a>

## English

This folder preserves the raw research notes from the 2026-06-15 to 2026-06-16 Franka tactile experiments. They describe an archived contact-to-force-map-to-RGB/SDK pipeline and may depend on the separately licensed camera SDK. They are historical provenance, not supported setup instructions.

Key source notes:

- `当前最终实现.md`: the implementation selected at the end of that experiment cycle;
- `0615-0616所有测试脚本.md`: inventory of the corresponding Franka test scripts;
- the remaining files: focused notes on contact projection, SDF/RGB conversion, and simulation behavior.

The archived pipeline was approximately:

```text
ContactSensor → COP/force map → pressure and texture displacement
→ tactile RGB renderer → external SDK FXYZ → Isaac Sim UI
```

For supported installation and experiments, use the current mainline documentation. The archive boundary and missing-SDK behavior are described in [Legacy and archive](../../../docs/en/guides/legacy.md).

<a id="简体中文"></a>

## 简体中文

本目录保留 2026-06-15 至 2026-06-16 Franka 触觉实验的原始研究记录，内容涉及已归档的“接触→力图→RGB/SDK”流程，并可能依赖需要单独授权的相机 SDK。这些文件用于历史追溯，不是当前受支持的安装说明。

主要记录：

- `当前最终实现.md`：该轮实验结束时选定的实现；
- `0615-0616所有测试脚本.md`：对应 Franka 测试脚本清单；
- 其他文件：接触投影、SDF/RGB 转换和仿真行为的专题记录。

归档流程大致为：

```text
ContactSensor → COP/力图 → 压力与纹理位移
→ 触觉 RGB 渲染器 → 外部 SDK FXYZ → Isaac Sim UI
```

安装和实验请使用当前主线文档。归档边界和缺少 SDK 时的行为见[旧版与归档](../../../docs/zh-CN/guides/legacy.md)。
