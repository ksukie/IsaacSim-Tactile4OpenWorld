# 7g membrane-local three-axis probe and 3-D force videos

> Status: draft visualization side branch. The renderer still needs final adaptation and validation
> against the restored frozen 7g estimator before the command below is considered canonical.

## Frozen coordinate contract

The only authoritative tactile membrane prim is:

```text
/World/envs/env_0/Robot/link8/UIPC_Pad/simulation/membrane_sim_mesh
```

Every physical probe reads that prim's world pose on every frame and evaluates

```text
x_i^M(t) = inverse(T_W_M(t)) x_i^W(t)
u_i^M(t) = x_i^M(t) - x_i^M(0)
```

`+X` is the outward normal, while `Y/Z` are the two membrane tangent axes. The frame audit writes
`membrane_from_pad_rotation.npy`, `pad_from_membrane_rotation.npy`, and
`membrane_frame_contract.json`. A case cannot enter the final renderer unless the rotation,
round-trip, and mean outward-normal checks pass.

## Required physical runs

Run normal press/release once with `OpenWorldTactile_v5_new_7e_indenter_deformation.py`.
Run the signed shear probe four times:

```bash
PY="${PYTHON_BIN:-python}"
PROBE=experiments/tactile-bench/OpenWorldTactile_v5_new_7g_lateral_shear_deformation_probe.py

$PY $PROBE --headless --no_save_camera_rgb \
  --lateral_axis y --lateral_direction positive \
  --output_dir /tmp/openworldtactile_7g_plus_y --workspace_dir /tmp/openworldtactile_7g_plus_y_ws \
  --fail_on_verdict_fail

$PY $PROBE --headless --no_save_camera_rgb \
  --lateral_axis y --lateral_direction negative \
  --output_dir /tmp/openworldtactile_7g_minus_y --workspace_dir /tmp/openworldtactile_7g_minus_y_ws \
  --fail_on_verdict_fail

$PY $PROBE --headless --no_save_camera_rgb \
  --lateral_axis z --lateral_direction positive \
  --output_dir /tmp/openworldtactile_7g_plus_z --workspace_dir /tmp/openworldtactile_7g_plus_z_ws \
  --fail_on_verdict_fail

$PY $PROBE --headless --no_save_camera_rgb \
  --lateral_axis z --lateral_direction negative \
  --output_dir /tmp/openworldtactile_7g_minus_z --workspace_dir /tmp/openworldtactile_7g_minus_z_ws \
  --fail_on_verdict_fail
```

Use the same normal preload, mesh, material, friction, amplitude, and frame counts for all four shear
runs. The `+Z/-Z` runs are independent physical UIPC experiments; a transformed or renamed `Y`
history is not accepted.

## Validation and rendering

```bash
python \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7g_local_3d_force_render.py \
  --normal_probe_dir /tmp/openworldtactile_7e_normal \
  --plus_y_probe_dir /tmp/openworldtactile_7g_plus_y \
  --minus_y_probe_dir /tmp/openworldtactile_7g_minus_y \
  --plus_z_probe_dir /tmp/openworldtactile_7g_plus_z \
  --minus_z_probe_dir /tmp/openworldtactile_7g_minus_z \
  --output_dir /tmp/openworldtactile_7g_membrane_3d \
  --fail_on_verdict_fail
```

The renderer first saves each case's membrane-local surface, displacement, per-vertex force, and
resultant. It then applies the normal press/unload gates and the signed shear gates. `+Z/-Z` also
requires orthogonal tangent crosstalk `RMS(Fy)/RMS(Fz) < 0.10`. Videos are generated only after every
gate passes, so the existence of the all-directions MP4 means real `+Z/-Z` source verdicts were
present and accepted.

## Outputs

Each `cases/<case>/` directory contains:

```text
membrane_surface_rest_local.npy
membrane_surface_current_local.npy
membrane_displacement_local.npy
vertex_force_membrane_local_tu.npy
force_membrane_local_tu.npy
membrane_frame_contract.json
membrane_from_pad_rotation.npy
pad_from_membrane_rotation.npy
```

The top-level videos are:

```text
membrane_force_3d_normal_press_release.mp4
membrane_force_3d_plus_y.mp4
membrane_force_3d_minus_y.mp4
membrane_force_3d_plus_z.mp4
membrane_force_3d_minus_z.mp4
membrane_force_3d_all_directions.mp4
```

All videos share one membrane-local view, one coordinate range, and one force-arrow scale. The
screen horizontal axis is membrane `Y`, vertical is membrane `Z`, and oblique depth is membrane `X`.
No frame is automatically normalized.

## Current evidence boundary

The previously recorded real `+Y/-Y` result remains valid evidence for that axis only. Until fresh
`--lateral_axis z` positive and negative runs pass and their verdicts are consumed by the renderer,
the repository must not claim that membrane-local three-axis validation is complete.
