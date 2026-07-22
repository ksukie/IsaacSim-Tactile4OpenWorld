# V5 new 7e static indenter contact deformation

## Scope

`OpenWorldTactile_v5_new_7e_indenter_deformation.py` extends 7d with one controlled UIPC
indenter:

```text
Piper link8 live pose
  -> back-face UIPC attachment targets
  -> animated ABD indenter target transform
  -> UIPC contact solve
  -> free front-surface Pad-local deformation
  -> unloading and recovery
```

The membrane attachment, rest frame, camera frame, and per-frame ordering from 7d are preserved.
The runtime does not rewrite all membrane or tool vertices. There is no force, pressure, or Fxyz
estimator in this stage.

## Indenter

The default indenter is a `2 mm x 2 mm` flat punch with a `4 mm` length. It uses a fixed six-tet box
decomposition and applies a small contact area. A cylinder remains available with `--tool_shape
cylinder` for later geometry sweeps.

The tool is an ABD object driven by `SoftTransformConstraint`. Direct FEM vertex writes are not used:
they teleport both current and previous positions and can bypass continuous contact. All-vertex FEM
position constraints were also rejected because the contact barrier prevented the requested tool
motion.

The default membrane discretization is a deterministic `1 x 10 x 12` structured grid with 286
vertices and 720 tetrahedra. `--membrane_mesh_mode wildmesh` remains available for compatibility, but
it is not used for 7e acceptance because repeated remeshing changes the contact trajectory.

The default finite-validation sequence is deliberately limited to:

```text
0 mm -> 0.2 mm -> 0.5 mm -> 0.2 mm -> 0 mm -> 0.2 mm retreat gap
```

The accepted operating interval is `0-0.5 mm`. A quasi-static `1.0 mm` attempt did not pass tool and
back-boundary tracking, so `1.0` and `2.0 mm` are outside the current validated range.

## Critical ordering

Each frame uses:

```text
1. Step PhysX and update the Piper articulation
2. Read link8 and compose the live Pad pose
3. Update back-face attachment aim positions
4. Update the ABD indenter aim transform
5. Advance and retrieve UIPC
6. Compute contact/deformation diagnostics
7. Update UIPC render surfaces and camera
```

At least one `--gripper_settle_steps` step is required before reading link8. A zero-step setup reads a
stale articulation frame and reintroduces the old approximately `31.622 mm` rest-frame offset, so the
script rejects zero.

## GUI loop

Run without `--headless`:

```bash
cd "${OWT_ROOT}"

"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7e_indenter_deformation.py \
  --loop_indentation \
  --render_viewport \
  --visual_mode uipc_surface \
  --indentation_levels_mm 0,0.2 \
  --pre_contact_frames 2 \
  --ramp_frames 8 \
  --level_hold_frames 8 \
  --recovery_frames 10 \
  --render_every 1 \
  --render_sleep_sec 0.01 \
  --log_every 5 \
  --no_save_camera_rgb \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7e_gui \
  --workspace_dir /tmp/openworldtactile_uipc_v5_new_7e_gui_ws
```

The simulation repeats loading, unloading, retreat, and recovery until the Isaac Sim window is closed
or `Ctrl+C` is pressed.

## Finite validation

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7e_indenter_deformation.py \
  --headless \
  --fail_on_verdict_fail \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7e_check \
  --workspace_dir /tmp/openworldtactile_uipc_v5_new_7e_check_ws
```

A quasi-static `0,0.2,0.5 mm` numerical validation produced:

```text
maximum commanded indentation       0.500000 mm
maximum actual indentation          0.498394 mm
settled hold tool target error      0.003023 mm
max back attachment error           0.005418 mm
peak normal compression             0.503471 mm
peak front displacement norm        1.483931 mm
max detected penetration            0.000000 mm
final contact vertex count          0
final recovery displacement         0.000026 mm
render/UIPC surface error           0.000000 mm
```

All verdict checks passed. A separate camera-enabled `0.2 mm` run also passed with a camera position
error of `0.0000079 mm`. The current contact mask contains membrane vertices in the tool footprint
whose signed gap is within `1.5 * d_hat`; it is a geometric contact-region diagnostic, not a force or
pressure field.

## Outputs

Primary files:

```text
surface_deformation.npy             # T x N x 3, Pad-local displacement
contact_vertex_mask.npy             # T x N geometric contact mask
indentation_mm.npy                  # commanded indentation
indentation_deformation_curve.npy   # command, actual, compression, deformation, contact count, penetration
final_deformation.npy               # final Pad-local residual
verdict.json
camera_rgb/
```

Additional files include Pad/link/tool poses, UIPC surface positions, back-target error, signed gap,
penetration, front compression, phase history, rest vertices, `surface_triangles.npy`, metadata, and
summary. The saved triangles use the same compact UIPC surface indexing as the surface history.

## Next stage

Use `OpenWorldTactile_v5_new_7f_deformation_field_probe.py` to extract and validate Pad-local displacement,
normal compression, shear displacement, spatial decay, and recovery. Constitutive Fxyz estimation
remains deferred until those raw deformation fields are accepted.
