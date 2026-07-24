# GelSight Mini model / GelSight Mini 模型

[English](#english) · [简体中文](#简体中文)

<a id="english"></a>

## English

This directory documents the GelSight Mini-derived sensor assets used by `openworldtactile_assets`.

### Provenance and license

- Sensor case and gel-pad source: [gelsightinc/gsrobotics](https://github.com/gelsightinc/gsrobotics), accessed 2025-02-15.
- Hardware reference: [GelSight Mini datasheet](https://www.gelsight.com/wp-content/uploads/productsheet/Mini/GelSight_Datasheet_GSMini.pdf), accessed 2025-02-15.
- The upstream CAD repository uses GPL-3.0. The derived model, converted USD files, textures, and calibration material in this subtree are distributed conservatively under GPL-3.0-only. Retain the adjacent `LICENSE` and the repository [third-party notices](../../../../../../../../THIRD_PARTY_NOTICES.md).
- Taxim-format calibration and rendering workflows also require the academic citation listed in the repository `CITATIONS.md`. Citation and software licensing are separate obligations.

### Model structure

The case and gel pad are separate meshes so applications can assign different physics and friction properties. This supports a rigid PhysX pad, a PhysX soft body, or a custom deformable solver such as UIPC. A translucent attachment plate connects the pad to the case in PhysX-based configurations, and a centered internal camera supports height-map, Taxim, and FOTS workflows.

Nominal dimensions (`length × width × height`):

- case: `32 mm × 28 mm × 24 mm`;
- gel pad: `25.25 mm × 20.75 mm × 4 mm`.

Camera-based use requires the matching sensor configuration and Isaac Sim camera rendering. If the attachment plate occludes the view, verify the renderer's translucency setting or use the configuration's visibility controls. Treat these dimensions as model metadata; consult the upstream hardware datasheet for physical-device specifications.

<a id="简体中文"></a>

## 简体中文

本目录说明 `openworldtactile_assets` 使用的 GelSight Mini 衍生传感器资产。

### 来源与许可证

- 传感器外壳和凝胶垫来源：[gelsightinc/gsrobotics](https://github.com/gelsightinc/gsrobotics)，访问日期 2025-02-15。
- 硬件参考：[GelSight Mini 数据表](https://www.gelsight.com/wp-content/uploads/productsheet/Mini/GelSight_Datasheet_GSMini.pdf)，访问日期 2025-02-15。
- 上游 CAD 仓库采用 GPL-3.0。本子树中的衍生模型、转换后的 USD、纹理和标定材料按 GPL-3.0-only 谨慎分发。再分发时须保留相邻的 `LICENSE` 和仓库[第三方声明](../../../../../../../../THIRD_PARTY_NOTICES.zh-CN.md)。
- Taxim 格式标定与渲染流程还需要进行仓库 `CITATIONS.md` 中列出的学术引用；论文引用与软件许可是两项独立义务。

### 模型结构

外壳和凝胶垫采用独立网格，应用可以为两者设置不同的物理与摩擦属性，因此可使用刚性 PhysX 垫、PhysX 软体或 UIPC 等自定义可形变求解器。基于 PhysX 的配置通过半透明连接板把凝胶垫固定到外壳，位于中心的内部相机支持高度图、Taxim 和 FOTS 流程。

标称尺寸（`长 × 宽 × 高`）：

- 外壳：`32 mm × 28 mm × 24 mm`；
- 凝胶垫：`25.25 mm × 20.75 mm × 4 mm`。

使用相机时需要匹配的传感器配置并启用 Isaac Sim 相机渲染。如果连接板遮挡视线，请检查渲染器的半透明设置或使用配置中的可见性控制。这些尺寸是仿真模型元数据；真实硬件规格应以上游数据表为准。
