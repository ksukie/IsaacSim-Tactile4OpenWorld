<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/guides/data-and-outputs.md">简体中文</a>
</p>

# Data and outputs

OpenWorldTactile experiments write into the directory selected by each script's `--output_dir`. Always set that argument explicitly. Defaults often point to `/tmp`, which may be cleared by the operating system and is unsuitable for long-term datasets.

## Data conventions

Unless a script's `metadata.json` says otherwise:

- world and local positions are stored in meters;
- time-series arrays use the first dimension as frame/time (`T`);
- quaternions follow the convention documented by the producing Isaac Lab API;
- `fxyz` uses three channels in the order `[fx, fy, fz]`;
- the tactile-bench convention maps `fx` to local Y shear, `fy` to local Z shear, and `fz` to local X normal pressure;
- `sim_constitutive_force` and TU are simulation-valued units, not calibrated Newtons.

Read `metadata.json` before interpreting an array. Historical scripts do not all share one schema or coordinate contract.

## V1 output contract

A completed `OpenWorldTactile_v1.py` run writes:

| Path | Contents |
|---|---|
| `fxyz.npy` | float32 tactile field, shape `[T, H, W, 3]`; defaults to `[T, 300, 300, 3]` |
| `metadata.json` | units, channel order, geometry, material, trajectory, and per-frame statistics |
| `preview_force.png` | visualization of the last saved field |
| `preview_sequence.mp4` | preview sequence when the video writer is available |
| `preview_frames/*.png` | individual preview frames at `--preview_every` cadence |

Load it safely without pickle:

```python
import json
from pathlib import Path

import numpy as np

run_dir = Path("outputs/v1-smoke")
fxyz = np.load(run_dir / "fxyz.npy", allow_pickle=False)
metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

assert fxyz.ndim == 4 and fxyz.shape[-1] == 3
assert np.isfinite(fxyz).all()
print(fxyz.shape, metadata["force_units"], metadata["channel_order"])
```

`--save_every` changes the number of saved frames; it does not change simulation stepping. `--preview_every` changes only preview cadence.

## V6.2 output groups

V6.2 writes a larger run contract. The most useful groups are:

| Group | Representative files | Interpretation |
|---|---|---|
| Tactile estimate | `surface_displacement_pad_local.npy`, `force_pad_local.npy`, `tactile_force_channels.npy` | Pad-local deformation and TU-valued tactile estimate |
| UIPC reaction | `uipc_reaction_force_w.npy`, `uipc_reaction_torque_w.npy` | raw time-averaged UIPC reaction in world coordinates |
| Applied coupling | `applied_uipc_force_w.npy`, `applied_uipc_force_substeps_w.npy` | relaxed/limited reaction applied to PhysX |
| Contact diagnostics | `contact_active.npy`, `minimum_signed_gap_mm.npy`, `uipc_reaction_vertex_count.npy` | contact state and penetration diagnostics |
| Motion | `object_pose_w.npy`, `object_velocity_w.npy`, `pad_pose_w.npy`, `gripper_opening_mm.npy` | object, pad, and gripper history |
| Timing | `frame_wall_time_sec.npy`, `uipc_step_time_sec.npy`, `uipc_substep_time_sec.npy` | wall-clock and solver timing |
| Run description | `metadata.json`, optional `error.json`/`uipc_timeout.json` | parameters, structure, termination, and failures |

Substep arrays use shape `[T, S, ...]`, where `S` is normally 8. Do not compare raw UIPC reaction directly with the limited PhysX-applied force without accounting for the contact-cone projection, feedback relaxation, and force limit.

Generate an offline field/video and a verdict:

```bash
./run.sh --python experiments/tactile-bench/render_tactile_field_offline.py \
  --input_dir "$PWD/outputs/v62-grasp" \
  --video_fps 15

./run.sh --python experiments/tactile-bench/validate_v6_2_once.py \
  --input_dir "$PWD/outputs/v62-grasp" \
  --fail_on_failure
```

See [Running experiments](experiments.md#v62-closed-loop-piper-grasp) for the complete workflow.

## Inspect an HDF5 file

HDF5 recorders appear in scenario-specific manipulation scripts. There is no single repository-wide HDF5 schema; inspect the file before choosing datasets.

Print all groups, datasets, shapes, and dtypes:

```bash
./run.sh --python -c "import h5py; p='path/to/episode.hdf5'; f=h5py.File(p,'r'); f.visititems(lambda n,o: print(n, getattr(o,'shape','group'), getattr(o,'dtype',''))); f.close()"
```

Common paths recognized by the bundled tools include:

```text
observations/images/<stream>
observations/tactile/fxyz
observations/tactile1
observations/tactile1_fxyz_float32
observations/tactile1_height_map_float32
observations/tactile1_deformation_float32
```

These are candidates, not a promise that every episode contains every path.

## View or export camera streams

Play all streams below `observations/images` side by side:

```bash
./run.sh --python tools/data/view_hdf5_images.py path/to/episode.hdf5
```

Select streams and playback rate:

```bash
./run.sh --python tools/data/view_hdf5_images.py path/to/episode.hdf5 \
  --streams top wrist \
  --fps 15 \
  --every 2
```

Export without opening a window:

```bash
./run.sh --python tools/data/view_hdf5_images.py path/to/episode.hdf5 \
  --export-dir outputs/decoded-camera \
  --no-show
```

The dedicated exporter provides the same non-interactive operation:

```bash
./run.sh --python tools/data/export_hdf5_images.py \
  path/to/episode.hdf5 outputs/decoded-camera \
  --every 1
```

## Export tactile datasets

Auto-detect known tactile datasets and render them as images:

```bash
./run.sh --python tools/data/export_hdf5_tactile_images.py \
  path/to/episode.hdf5 outputs/decoded-tactile
```

Select a dataset and subsample frames:

```bash
./run.sh --python tools/data/export_hdf5_tactile_images.py \
  path/to/episode.hdf5 outputs/decoded-tactile \
  --datasets observations/tactile/fxyz \
  --every 5 \
  --format png \
  --scale 4
```

For a three-channel float field, the exporter writes a combined RGB visualization and one color-mapped image per `fx`, `fy`, and `fz` channel. Normalization is for viewing only; exported colors are not physical values.

## Dataset quality checks

Before using a run for analysis or training, check at least:

- arrays exist and have the documented rank/channel order;
- all floating-point arrays are finite;
- time dimensions agree across synchronized observations;
- contact and free-space frames are both present when expected;
- unit, coordinate-frame, simulator, and calibration fields are recorded;
- `error.json`, timeout files, or failed verdicts are absent or explicitly accounted for;
- source revision, command, seed, and asset/contract identifiers are preserved.

Store raw numeric arrays and metadata alongside visualizations. A preview image is not a substitute for the source field.

## Storage practices

- Write each run to a new directory.
- Keep the UIPC workspace separate from durable outputs.
- Do not commit generated datasets, videos, caches, checkpoints, or UIPC build directories to the source repository.
- Copy successful runs out of `/tmp` before reboot or cleanup.
- Use checksums when moving large datasets between machines.
- Retain the license and provenance of any third-party input assets distributed with a dataset.
