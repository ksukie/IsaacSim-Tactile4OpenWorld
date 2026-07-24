<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/guides/legacy.md">简体中文</a>
</p>

# Legacy Isaac Lab 2.3.2 route

`archive-isaaclab-2.3.2/` preserves an older GelSight, RGB/force-field, Franka, and camera-SDK integration route. It is retained for provenance and migration work. It is not the current mainline and is not a complete Isaac Lab checkout.

## Use this route only when

- reproducing a result tied to the archived scripts;
- studying the former OpenWorldTactile/GelSight sensor integration;
- porting one specific method or asset to a maintained baseline;
- working with an independently licensed copy of the historical camera SDK.

New UIPC tactile-bench work should start in `active-isaaclab-2.1.1/`.

## Contents

| Path | Purpose |
|---|---|
| `packages/contrib/` | historical `isaaclab_contrib` tactile sensor integration |
| `packages/assets-patches/` | selected legacy asset configuration patches |
| `experiments/franka/current/` | Franka contact, SDF, RGB, shear, and force-map experiments |
| `experiments/sensors/` | individual OpenWorldTactile/GelSight sensor probes |
| `experiments/rgb-pipeline/` | combined RGB and fxyz workflows |
| `hardware-sdk/` | documented mount point for an externally obtained SDK; SDK binaries are not distributed |
| `notes/` | unmaintained internal research records, not user setup instructions |

There is no single legacy script that represents every workflow.

## External requirements

Prepare these independently:

- a separate Isaac Lab 2.3.2 and matching Isaac Sim environment;
- any `isaaclab_contrib`/asset changes required by the selected script;
- GelSight R1.5 and Factory assets referenced through Nucleus or a local mapping;
- an authorized, platform-compatible OpenWorldTactile camera SDK when the selected script imports it;
- compatible native libraries, camera drivers, and hardware.

Do not install the 2.1.1 and 2.3.2 routes into one environment and assume their APIs are interchangeable.

## Configure external assets

Where supported by the archived source:

```bash
export OWT_ASSET_ROOT=/absolute/or/nucleus/path/to/openworldtactile-assets
export OWT_SDK_ROOT=/absolute/path/to/hardware-sdk/openworldtactile
```

The default SDK mount point is:

```text
archive-isaaclab-2.3.2/hardware-sdk/openworldtactile/
```

Read the bilingual [SDK boundary notice](../../../archive-isaaclab-2.3.2/hardware-sdk/README.md). The removed vendor DLL/SO files must not be restored to the public repository without verifiable redistribution rights.

## Restoration workflow

1. Create a clean, separate Isaac Lab 2.3.2 environment.
2. Select one target script and inspect its imports/assets in the [internal entry-point inventory](../../internal/ENTRYPOINT_MATRIX.md).
3. Port only the required `packages/contrib/` and `packages/assets-patches/` content into an external project or compatible extension layout. Do not overwrite an upstream checkout wholesale.
4. Configure `OWT_ASSET_ROOT` and, only when required, `OWT_SDK_ROOT`.
5. Run the target script's `--help` using the 2.3.2 interpreter before connecting cameras or hardware.
6. Validate with a simulator-only, non-hardware case first.
7. Record all manual patches, external asset revisions, SDK version, platform, and command.

Because the archive does not include a standardized installer or a complete upstream tree, exact port steps depend on the target script. A successful static parse is not evidence that the restored runtime is correct.

## Operational safety

Archived code may access cameras, USB devices, native libraries, or robot-related interfaces. Review it in an isolated environment before connecting physical equipment. Do not use this research archive as a hardware-safety layer.

## Migration guidance

- Port one feature at a time into a new mainline branch.
- Replace hard-coded Nucleus/SDK paths with explicit configuration.
- Revalidate frames, units, image formats, and Isaac Lab API changes.
- Keep original attribution and license notices.
- Add maintained English and Simplified Chinese user documentation for any migrated public workflow.
