<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/guides/custom-integration.md">简体中文</a>
</p>

# Custom integration

Use the installed packages when adding OpenWorldTactile to another Isaac Lab scene. The APIs are research-stage interfaces; pin the repository revision and start from the closest maintained example.

## Package map

| Import package | Public role |
|---|---|
| `openworldtactile` | `GelSightSensor`, its configuration/data classes, and optical/marker simulation approaches |
| `openworldtactile_assets` | packaged USD paths and ready-made robot/sensor configurations |
| `openworldtactile_uipc` | UIPC simulator, objects, attachments, environment base classes, and mesh utilities |
| `openworldtactile_tasks` | Gymnasium registrations and task/agent configurations |

## Follow Isaac Sim import order

Scripts that use Isaac/Omniverse modules must launch the app before importing most simulation modules:

```python
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

# Isaac Lab and OpenWorldTactile imports follow app launch.
from openworldtactile import GelSightSensor
from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.sensors import GELSIGHT_MINI_TAXIM_CFG
from openworldtactile_uipc import UipcObject, UipcObjectCfg, UipcSim, UipcSimCfg
```

Wrap cleanup in `try/finally` and call `app.close()` in a complete script.

## Reuse an asset configuration

Ready-made objects are Isaac Lab configuration instances. Copy or replace them instead of mutating a shared module-level instance:

```python
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_CFG
from openworldtactile_assets.sensors import GELSIGHT_MINI_TAXIM_CFG

robot_cfg = AGILEX_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
sensor_cfg = GELSIGHT_MINI_TAXIM_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot/gelsight_mini_case",
    data_types=["tactile_rgb", "height_map", "camera_depth"],
)
```

Add these configurations to an Isaac Lab scene configuration using the same pattern as the existing tasks. Useful examples:

- [Taxim sensor prototype](../../../active-isaaclab-2.1.1/experiments/simulation-prototypes/check_taxim_sim.py)
- [FEM/UIPC marker prototype](../../../active-isaaclab-2.1.1/experiments/simulation-prototypes/check_mani_skill_marker_franka.py)
- [Allegro task sensor configuration](../../../active-isaaclab-2.1.1/packages/tasks/openworldtactile_tasks/inhand/config/allegro_hand/allegro_env_cfg.py)

The optional `GELSIGHT_MINI_TAXIM_FEM_CFG` symbol is exported only when its FEM dependency imports successfully.

## Sensor outputs

`GelSightSensor.data.output` is a dictionary keyed by requested `data_types`. Available implementations may provide:

| Key | Meaning |
|---|---|
| `tactile_rgb` | synthesized tactile RGB image |
| `marker_motion` | initial/current marker image positions |
| `height_map` | indentation height/depth representation |
| `camera_depth` | internal sensor-camera depth |
| `camera_rgb` | internal sensor-camera RGB |
| `tactile_force_field` | method-specific tactile field when configured |

Shapes depend on the selected sensor camera and simulation approach. Inspect the actual tensor/array and configuration rather than assuming every approach returns the same resolution or force definition.

## UIPC integration lifecycle

The smallest complete UIPC examples create a `UipcSim`, create `UipcObject` instances attached to it, reset the Isaac simulation, call `uipc_sim.setup_sim()`, and then update render meshes after each Isaac simulation step:

```text
create SimulationContext
create UipcSim(UipcSimCfg(...))
create UipcObject(..., uipc_sim) instances
sim.reset()
uipc_sim.setup_sim()

repeat:
    write kinematic/input state
    sim.step(...)
    uipc_sim.update_render_meshes()
    update Isaac/UIPC object buffers
    read deformation/contact output
```

Use [V1](../../../active-isaaclab-2.1.1/experiments/tactile-bench/OpenWorldTactile_v1.py) as the compact reference for a fixed membrane, and [V6.2](../../../active-isaaclab-2.1.1/experiments/tactile-bench/OpenWorldTactile_v6_2_grasp.py) for Pad-local external-boundary coupling. V6.2's coupling rules are specialized and should not be copied partially.

## Coordinate and force contracts

Define these before writing a new estimator or dataset:

1. owning frame for rest and current membrane surfaces;
2. normal and tangent axis directions;
3. world-to-sensor transformation convention;
4. force channel order, sign, and unit;
5. whether an output is a physical reaction, a limited applied force, or a tactile estimator value;
6. reset/rest-surface timing and handling of rigid sensor motion.

For the current tactile-bench Pad convention, local `+X` is the outward normal and local `+Y/+Z` are tangents. Do not reuse that statement for another asset without confirming its authored frame.

## Add a task

1. Implement the environment/config under `packages/tasks/openworldtactile_tasks/`.
2. Register a unique `OpenWorldTactile-...-vN` ID with Gymnasium.
3. Add only the agent entry points that have matching config files.
4. Ensure a project-aware launcher imports `openworldtactile_tasks` after app launch.
5. Smoke-test reset and several steps with one environment.
6. Document observation/action spaces, assets, units, reward, termination, and required camera/UIPC flags in both languages.

## Extension checklist

- Use paths derived from `OWT_ASSETS_DATA_DIR` or explicit configuration, not machine-specific absolute paths.
- Put generated outputs outside package/asset directories.
- Give concurrent UIPC simulations separate workspaces.
- Close the simulation app even when an exception occurs.
- Add lightweight tests for frame transforms and numeric post-processing.
- Preserve third-party SPDX headers, notices, and citations.
- Treat public class names as revision-pinned research APIs until a stability policy is published.
