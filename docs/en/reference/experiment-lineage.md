<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/reference/experiment-lineage.md">简体中文</a>
</p>

# Experiment lineage

Two independent version dimensions appear in this repository:

- Isaac Lab `2.1.1` and `2.3.2` identify external runtime baselines.
- OpenWorldTactile `V1` through `V6.2` identify research stages and probes inside the 2.1.1 tactile-bench.

`V6.2` is not “newer than Isaac Lab 2.3.2”; the numbers describe different things.

## User-facing status

| Stage | Recommended use | Status |
|---|---|---|
| V1 | first fixed-membrane UIPC run and dense fxyz output | documented starting point |
| V5.7d–V5.7f | build the coordinate/deformation contract required by later estimators | documented V6.2 prerequisite |
| V5.7g | frozen deformation-to-force estimator used by later stages | internal algorithm boundary; tested offline |
| V5.9 | TU field rendering helpers reused by V6.2 | implementation dependency, not the first user entry |
| V6.2 | closed-loop Piper/PhysX/UIPC grasp | current advanced scenario |
| all other version scripts | reproduce diagnostics or study method evolution | research history; inspect before use |

## Major stages

| Family | Main question addressed |
|---|---|
| V1 | Can one UIPC membrane produce a dense three-channel deformation-based tactile field? |
| V2–V2.8 | Can force estimation, camera-observed membrane data, marker tracking, texture/imprint generation, and sample packaging be separated into reusable steps? |
| V3 documentation | What validation suite is needed before mounting the sensor? |
| V4–V4.9c | Can the membrane, camera, and force field follow a robot-mounted sensor without rigid-motion artifacts, and can authored USD contact be validated? |
| early V5 | Can mounted contact and grasp/insertion scenarios be integrated? |
| V5 `new` 1–6 | Can mount alignment, PhysX grasp, contact gap, visual layers, gradient/sign calibration, pressure/shear proxies, and constitutive deformation be isolated? |
| V5.7a–V5.7g | Can membrane deformation, attachment/follow behavior, coordinate contracts, repeatability, and a frozen estimator be specified and verified? |
| V5.8–V5.9 | Can grasp integration and a locally rendered TU tactile field reuse the frozen estimator? |
| V6.0–V6.1b | Can grasp/lift/full-cycle behavior and failure diagnostics be validated? |
| V6.2 | Can raw UIPC contact reaction be coupled back into a PhysX-owned object and gripper while preserving separate tactile estimation? |

The original stage documents also discuss possible dataset-quality and real-force calibration phases. A planned phase is not an implemented or validated release feature.

## Dependency shape

V6.2 is not standalone:

```text
V5.7d + five V5.7e runs
          │
          ▼
     V5.7f contract
          │
          ├── V5.7g frozen estimator
          │          │
          │          ▼
          └──── V5.9 field helpers
                         │
                         ▼
                       V6.2
```

Copying only `OpenWorldTactile_v6_2_grasp.py` will omit local imports, assets, and its input contract.

## How to choose a script

1. For a new user, run V1.
2. For the current grasp result, follow the complete V6.2 workflow.
3. For an algorithmic question, select the narrowest probe whose name and retained note match the question.
4. Inspect imports, required files, defaults, and verdict logic before executing a historical stage.
5. Do not infer stability from a larger version number.

The maintainer-generated [entry-point matrix](../../internal/ENTRYPOINT_MATRIX.md) lists all detected experiment/tool scripts, direct-run signals, import roots, and referenced assets. It is an inventory, not a support matrix.
