# V5 new 7c pad/UIPC follow probe

## Scope

`OpenWorldTactile_v5_new_7c_pad_follow_probe.py` validates this chain only:

```text
Piper link8
  -> directly mounted UIPC_Pad USD and internal camera
  -> live articulation body pose composed with the authored pad local pose
  -> UIPC membrane world vertices
  -> Fabric/OmniHydra points
```

It intentionally has no anchor, attachment, contact, force, or pressure model.

The USD hierarchy remains:

```text
/World/envs/env_0/Robot/link8/UIPC_Pad
  /simulation/membrane_sim_mesh
  /sensors/camera
```

The hierarchy owns the pad and camera mount. The live UIPC bridge uses
`robot.data.body_link_pos_w/body_link_quat_w` because the current Isaac physics/Fabric runtime does
not reliably expose the moving articulation pose through `omni.usd.get_world_transform_matrix`.
The stage pad pose is recorded as a diagnostic comparison but never drives the membrane.

## GUI loop

```bash
cd "${OWT_ROOT}"

"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7c_pad_follow_probe.py \
  --loop_motion \
  --render_viewport \
  --visual_mode uipc_only \
  --render_every 1 \
  --render_sleep_sec 0.01 \
  --camera_save_every 10 \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7c_pad_follow_gui \
  --workspace_dir /tmp/openworldtactile_uipc_v5_new_7c_pad_follow_gui_ws
```

Do not add `--headless` when a visible Isaac Sim window is required. Close the window or press
`Ctrl+C` to stop an infinite loop.

`--visual_mode uipc_only` hides the duplicate black camera membrane, green backing membrane, and
texture pattern. The visible red membrane is the actual UIPC/Fabric target. Use
`--visual_mode full_pad` to inspect all USD-authored layers.

## Finite headless validation

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7c_pad_follow_probe.py \
  --headless \
  --render_viewport \
  --no_save_camera_rgb \
  --fail_on_verdict_fail \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7c_pad_follow_check \
  --workspace_dir /tmp/openworldtactile_uipc_v5_new_7c_pad_follow_check_ws
```

## Outputs

Primary outputs:

```text
pad_pose.npy
uipc_surface_pose.npy
follow_error.npy
camera_rgb/
verdict.json
```

Detailed diagnostics:

```text
link_pose_w.npy
pad_pose_w.npy
stage_pad_pose_w.npy
uipc_surface_w.npy
surface_follow_error_mm.npy
fabric_follow_error_mm.npy
stage_pad_position_error_mm.npy
stage_pad_angle_error_deg.npy
target_opening_mm.npy
measured_opening_mm.npy
metadata.json
summary.json
```

The verdict requires the real mount link to move and both the UIPC surface and Fabric points to
match the pad pose derived from the real articulation body.
