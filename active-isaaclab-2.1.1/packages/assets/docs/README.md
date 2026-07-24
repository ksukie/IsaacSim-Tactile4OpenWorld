# Asset package / 资产包

[English](#english) · [简体中文](#简体中文)

<a id="english"></a>

## English

`openworldtactile_assets` provides Python configurations and local data for the robots, tactile sensors, props, and test scenes used by this project. Install it with `./run.sh --install` from `active-isaaclab-2.1.1/`.

Import the packaged data root instead of hard-coding a machine-specific path:

```python
from openworldtactile_assets import OWT_ASSETS_DATA_DIR

ball_usd = f"{OWT_ASSETS_DATA_DIR}/Props/ball_wood.usd"
```

The data tree is organized by purpose:

- `Robots/`: robot models and tactile-sensor assemblies;
- `Sensors/`: standalone tactile sensor models and calibration assets;
- `Props/`: objects, mounts, markers, and task geometry;
- `Policies/`: packaged policy artifacts when redistribution is permitted;
- `Test/`: assets used by package checks.

When adding an asset, keep its USD and dependent files together, resolve paths from `OWT_ASSETS_DATA_DIR`, document units and coordinate frames, and include source, license, conversion steps, and required attribution. Review the repository [third-party notices](../../../../THIRD_PARTY_NOTICES.md) before redistributing derived assets.

<a id="简体中文"></a>

## 简体中文

`openworldtactile_assets` 提供项目所用机器人、触觉传感器、道具和测试场景的 Python 配置与本地数据。请在 `active-isaaclab-2.1.1/` 中通过 `./run.sh --install` 安装。

请导入包内数据根目录，不要硬编码某台机器的绝对路径：

```python
from openworldtactile_assets import OWT_ASSETS_DATA_DIR

ball_usd = f"{OWT_ASSETS_DATA_DIR}/Props/ball_wood.usd"
```

数据目录按用途组织：

- `Robots/`：机器人模型和触觉传感器装配体；
- `Sensors/`：独立触觉传感器模型与标定资产；
- `Props/`：物体、安装件、标记和任务几何体；
- `Policies/`：在允许再分发时打包的策略文件；
- `Test/`：包检查使用的资产。

新增资产时，应将 USD 与依赖文件放在一起，通过 `OWT_ASSETS_DATA_DIR` 解析路径，记录单位和坐标系，并补充来源、许可证、转换步骤和署名要求。再分发衍生资产前请检查仓库的[第三方声明](../../../../THIRD_PARTY_NOTICES.zh-CN.md)。
