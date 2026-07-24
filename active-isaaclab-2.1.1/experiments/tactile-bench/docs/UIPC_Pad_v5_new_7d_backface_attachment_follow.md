# V5 new 7d back-face attachment follow

## Scope

`OpenWorldTactile_v5_new_7d_backface_follow.py` validates this chain:

```text
Piper link8 live articulation pose
  -> UIPC_Pad pose
  -> back-face UIPC SoftPositionConstraint targets
  -> UIPC solve with unconstrained front/interior vertices
  -> simulation/membrane_sim_mesh render surface
```

There is no PhysX anchor, contact object, force estimator, or pressure model. Runtime code does not
rewrite the full membrane vertex array. A full write is used once before the first solve to align the
tetrahedral mesh from its USD construction frame to the live Pad frame.

## Critical ordering

The attachment must be constructed before the membrane is added to the UIPC Scene. Otherwise
`SoftPositionConstraint.apply_to()` is too late, UIPC does not register `GlobalAnimator`, and changing
`aim_positions` has no effect.

Each simulation frame uses this order:

```text
1. PhysX step and Articulation data update
2. Read link8 pose and compose Pad pose
3. Update back-face attachment aim_positions
4. Advance UIPC manually
5. Retrieve UIPC surface and update render points
6. Drive the diagnostic camera and render
```

The tetrahedral rest coordinates are recovered with the inverse of
`UipcObject.init_world_transform`. Using a Stage Pad pose here mixes USD and Fabric frames and creates
the previously observed 31.622 mm membrane offset.

## Camera

The authored camera under `link8/UIPC_Pad/sensors/camera` remains the calibration source. A camera
with the same intrinsic parameters is created at `/World/UIPC_PadCaptureCamera` and receives the live
world pose every frame. The world-root copy is necessary because a camera under the articulation
subtree is overwritten by the stale parent Stage transform in this runtime.

This camera currently observes the red UIPC simulation surface. Mapping UIPC deformation onto
`visual/membrane_camera_surface` remains a 7g task.

## GUI loop

```bash
cd "${OWT_ROOT}"

"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7d_backface_follow.py \
  --loop_motion \
  --render_viewport \
  --visual_mode uipc_surface \
  --render_every 1 \
  --render_sleep_sec 0.01 \
  --camera_save_every 10 \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7d_gui \
  --workspace_dir /tmp/openworldtactile_uipc_v5_new_7d_gui_ws
```

Do not add `--headless` when a visible Isaac Sim window is required. Close the window or press
`Ctrl+C` to stop the loop.

## Finite validation

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7d_backface_follow.py \
  --headless \
  --render_viewport \
  --fail_on_verdict_fail \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7d_check \
  --workspace_dir /tmp/openworldtactile_uipc_v5_new_7d_check_ws
```

The tested short-cycle run produced:

```text
link motion                         18.876016 mm
back constrained coverage           1.0
front constrained fraction          0.0
max back target error               0.073464 mm
max hold front Pad-local residual   0.001606 mm
max render/UIPC surface error       0.0 mm
max camera position error           0.0000074 mm
mean back/front rigid-follow ratio  1.000010 / 1.000017
```

In zero gravity with no contact, both constrained and free vertices should translate with the Pad, so
their world displacement ratios should be approximately 1. Deformability is evaluated using the
Pad-local residual after removing rigid motion. A controlled contact in 7e is required to demonstrate
a nonzero, physically meaningful free-surface deformation.

## Outputs

Primary files:

```text
pad_pose.npy
uipc_surface_pose.npy
follow_error.npy
camera_rgb/
verdict.json
```

Additional diagnostics include attachment coverage, front/back vertex indices, Pad-local residuals,
rigid-follow ratios, camera pose errors, render surface errors, phase history, metadata, summary, and
`surface_triangles.npy`. The saved triangles use the same compact UIPC surface indexing as
`uipc_surface_w.npy`.

## Next stage

7e should add one static rigid indenter with UIPC contact enabled while preserving the 7d boundary and
step ordering. Validate indentation depth versus front-surface Pad-local displacement before adding a
moving grasp, visual texture mapping, or force estimation.
