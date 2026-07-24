# External camera SDK mount / 外部相机 SDK 挂载点

[English](#english) · [简体中文](#简体中文)

<a id="english"></a>

## English

Some archived Isaac Lab 2.3.2 experiments expect a separately obtained OpenWorldTactile camera SDK at:

```text
archive-isaaclab-2.3.2/hardware-sdk/openworldtactile/
```

The SDK is not distributed here. A previously bundled copy was excluded on 2026-07-22 because no verifiable redistribution license accompanied it and it contained vendor-native `SonixCamera.dll` and `libSonixCamera.so` files. Obtain an authorized SDK from its rightsholder, verify platform and version compatibility, and place it at the path above or set `OWT_SDK_ROOT` where the archived script supports that variable.

Do not commit the SDK unless written redistribution rights, the complete license, source/version provenance, and binary notices have all been documented. These SDK-dependent scripts are retained for reproducibility and are not part of the supported mainline. See [Legacy and archive](../../docs/en/guides/legacy.md).

<a id="简体中文"></a>

## 简体中文

部分归档的 Isaac Lab 2.3.2 实验要求用户单独取得 OpenWorldTactile 相机 SDK，并放置在：

```text
archive-isaaclab-2.3.2/hardware-sdk/openworldtactile/
```

本仓库不分发该 SDK。此前捆绑的副本已于 2026-07-22 移除，原因是缺少可核验的再分发许可，且包含供应商原生文件 `SonixCamera.dll` 和 `libSonixCamera.so`。请从权利人或授权渠道取得 SDK，核对平台与版本后放入上述目录；如果归档脚本支持，也可设置 `OWT_SDK_ROOT`。

除非已经记录书面再分发授权、完整许可证、来源/版本信息和二进制声明，否则不要将 SDK 提交到仓库。依赖该 SDK 的脚本仅为复现历史结果而保留，不属于受支持主线。详见[旧版与归档](../../docs/zh-CN/guides/legacy.md)。
