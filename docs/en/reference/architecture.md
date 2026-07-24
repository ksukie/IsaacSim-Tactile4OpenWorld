<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/reference/architecture.md">简体中文</a>
</p>

# Architecture

IsaacSim-Tactile4OpenWorld is an extension project around external Isaac Sim and Isaac Lab installations. The current mainline combines sensor models, assets, task definitions, UIPC contact simulation, and research experiments without vendoring the full Isaac platform.

## System context

```text
External runtime
  Isaac Sim 4.5 + Isaac Lab 2.1.1 + CUDA/PyTorch
        │
        ▼
OpenWorldTactile mainline
  core sensor interfaces ── assets/configs ── task environments
        │                         │                    │
        └────────── UIPC integration + libuipc ──────┘
                              │
                              ▼
                 experiments and data tools
```

The independent `archive-isaaclab-2.3.2/` route is outside this mainline and must use a separate environment.

## Mainline packages

| Package | Responsibility | Important exports/examples |
|---|---|---|
| `openworldtactile` | GelSight-style sensor interface and tactile rendering/marker approaches | `GelSightSensor`, `GelSightSensorCfg`, Taxim, FOTS, FEM implementations |
| `openworldtactile_assets` | USD data and ready-made Isaac Lab configurations | Piper, Franka, Allegro, GelSight Mini, OpenWorldTactile pad paths |
| `openworldtactile_uipc` | bridge between Isaac Lab scene state and libuipc contact/deformation | `UipcSim`, `UipcObject`, attachments, mesh utilities, UIPC RL base classes |
| `openworldtactile_tasks` | Gymnasium task registrations and agent configs | ball rolling, factory assembly, pole balancing, Allegro reorientation |

Packages are installed editable by `active-isaaclab-2.1.1/run.sh`, so source changes take effect in the selected Python environment without rebuilding pure-Python wheels. Native UIPC changes still require recompilation.

## Tactile sensor path

The high-level `GelSightSensor` path composes an Isaac sensor camera with one or more simulation approaches:

```text
scene geometry/contact state
        │
        ├── internal depth/RGB camera
        ├── Taxim optical rendering
        ├── FOTS marker motion
        └── FEM/UIPC deformation source (when configured)
        │
        ▼
GelSightSensorData.output
  tactile_rgb / marker_motion / height_map / camera_depth /
  camera_rgb / method-specific tactile_force_field
```

Not every configuration provides every output. `data_types` selects requested channels, and the chosen optical/marker simulator determines their meaning and shape.

## UIPC fixed-membrane path (V1)

V1 uses UIPC for deformation and derives a dense tactile field from the membrane surface:

```text
Isaac/PhysX scene step
       │
       ▼
UIPC membrane and kinematic indenter update
       │
       ▼
current surface − recorded rest surface
       │
       ▼
deformation-based constitutive estimate
       │
       ▼
conservative splat to [T, 300, 300, 3]
       │
       └── metadata + NumPy + image/video previews
```

The output is labeled `sim_constitutive_force`. It is not calibrated to Newton.

## Closed-loop coupling path (V6.2)

V6.2 separates state ownership:

- PhysX owns the Piper articulation and free cylinder state.
- UIPC owns one deformable membrane in the sensor Pad frame.
- The cylinder is mirrored into UIPC only as a kinematic external boundary.
- UIPC reaction is transformed to world coordinates, projected into an admissible contact cone, relaxed/limited, and applied to PhysX on the next substep.
- A frozen V5.7g tactile estimator consumes membrane deformation but does not drive PhysX.

Each 60 Hz record interval contains at least eight alternating PhysX/UIPC substeps:

```text
previous limited UIPC reaction -> PhysX step
PhysX cylinder pose -> Pad-local UIPC boundary
UIPC solve -> raw reaction
contact-cone projection + relaxation + limit
-> next PhysX substep
```

Raw reaction, admissible reaction, actually applied force, motion, gap, and timing are stored separately. This distinction is essential when analyzing the output.

## Coordinate and channel conventions

The current tactile-bench Pad convention is:

```text
frame: Pad local
+X: outward surface normal
+Y, +Z: tangent directions
displacement: current surface − rest surface
fxyz channels: [local-Y shear, local-Z shear, local-X normal]
```

World-space arrays usually include a `_w` suffix; Pad/sensor-local arrays include `_pad_local`, `_pad_l`, or an explicit metadata frame. Historical scripts may differ, so filenames alone are not a universal contract.

## Task path

```text
openworldtactile_tasks import
       │
       ▼
Gymnasium environment registration
       │
       ├── env config entry point
       └── backend-specific agent config entry points
       │
       ▼
project-aware Isaac Lab random/train/play launcher
       │
       ▼
logs, configuration dumps, checkpoints, optional videos
```

The external launcher must import `openworldtactile_tasks` after starting Isaac Sim. See [Tasks and training](../guides/tasks-and-training.md).

## Active and archive boundary

| Route | Runtime baseline | Role |
|---|---|---|
| `active-isaaclab-2.1.1/` | Isaac Lab 2.1.1 / Isaac Sim 4.5 | maintained project mainline |
| `archive-isaaclab-2.3.2/` | separate Isaac Lab 2.3.2 environment | historical GelSight/SDK experiments and migration reference |

Code, assets, or patches from the archive should be migrated individually. Overlaying the entire archive onto the mainline would mix incompatible APIs and undocumented external dependencies.

## Stability boundaries

- Command-line interfaces documented in the user guide are the preferred entry points, but remain research interfaces.
- Experiment filenames encode research history, not semantic package versions.
- Public Python exports are not yet covered by a formal deprecation policy.
- Output schemas are per-script contracts; there is no universal schema across all historical experiments.
- Static repository checks validate structure and parseability, not simulation physics or hardware safety.

Pin a repository revision for published experiments and retain the command, metadata, and validation verdict with results.
