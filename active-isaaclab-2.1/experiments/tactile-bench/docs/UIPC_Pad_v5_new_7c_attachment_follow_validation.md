# V5 new 7c attachment follow validation

## Goal

`OpenWorldTactile_v5_new_7c_attachment_follow_validation.py` validates only UIPC membrane follow.

The default mode is the hard follow diagnostic:

```text
commanded gripper opening
  -> controlled pad pose along the configured local follow axis
  -> rewrite all UIPC membrane tet vertices in world coordinates
  -> UIPC membrane surface visibly follows the moving pad
```

This proves the UIPC membrane no longer stays world-static after UIPC takes over the mesh.

The old USD-stage pose sources are still available for comparison:

```text
--pad_pose_source stage_pad_root
--pad_pose_source stage_pose_driver
--pad_pose_source articulation_body
```

The minimal USD probe shows that a direct referenced pad can move with the stage evaluation while
UIPC is not active. Once 7c gives the membrane mesh to UIPC/Fabric, that USD parenting is no longer a
reliable runtime source for membrane vertices. The default `gripper_opening` source therefore uses a
deterministic opening-to-pad-pose mapping for the follow validation.

The optional physical attachment mode is:

```text
controlled pad pose
  -> kinematic back-face anchor
  -> UIPC back-face attachment vertices
  -> UIPC membrane surface follows the moving pad
  -> front surface remains outside the attachment set
```

This stage does not validate contact, force, fxyz, or pressure. `direct_kinematic` is not the final soft-contact model; it is a visibility and coordinate-follow diagnostic.

The metadata is intentionally fixed to:

```json
{
  "force_source": "none",
  "pressure_source": "none",
  "contact_test": false,
  "goal": "attachment/follow validation only"
}
```

## Run

Headless:

```bash
cd "${OWT_ROOT}"

./run.sh -p experiments/tactile-bench/OpenWorldTactile_v5_new_7c_attachment_follow_validation.py \
  --headless \
  --uipc_follow_mode direct_kinematic \
  --pad_pose_source gripper_opening \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7c_attachment_follow_validation
```

With viewport:

```bash
./run.sh -p experiments/tactile-bench/OpenWorldTactile_v5_new_7c_attachment_follow_validation.py \
  --loop_motion \
  --render_every 1 \
  --render_sleep_sec 0.01 \
  --uipc_follow_mode direct_kinematic \
  --pad_pose_source gripper_opening \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7c_attachment_follow_validation
```

Non-headless runs enable viewport rendering automatically. `--loop_motion` repeats the close/open
sequence until the Isaac Sim window is closed and retains only the latest motion cycle in memory.

To test only the physical back-face attachment, add:

```bash
--uipc_follow_mode back_face_attachment
```

Current note: `back_face_attachment` is retained as a soft-attachment experiment. In the current
runtime it can leave UIPC vertices behind even when the kinematic anchor follows. The default
`direct_kinematic` bridge is the passing 7c fix for visual/coordinate follow; it is not the final
soft-contact model.

## Outputs

Required 7c outputs:

```text
pad_pose_w.npy
uipc_membrane_surface_w.npy
anchor_vertex_error_mm.npy
surface_follow_error_mm.npy
free_surface_deformation_mm.npy
attachment_vertex_count
verdict.json
```

`attachment_vertex_count` is stored in `verdict.json`, `metadata.json`, and `status.json`.

Additional debug outputs:

```text
attachment_vertex_indices.npy
front_surface_indices.npy
back_surface_indices.npy
rest_tet_vertices_local.npy
rest_surface_vertices_local.npy
anchor_pose_error_mm.npy
anchor_pose_angle_error_deg.npy
pad_motion_mm.npy
pad_angle_motion_deg.npy
gripper_opening_target_mm.npy
gripper_opening_measured_mm.npy
phase_history.json
metadata.json
status.json
```

## Verdict

Read:

```bash
cat /tmp/openworldtactile_uipc_v5_new_7c_attachment_follow_validation/verdict.json
```

The main pass field is:

```json
{
  "attachment_follow_passed": true
}
```

The verdict checks:

```text
pad pose moved enough to test follow
attachment_vertex_count > 0
anchor pose follows pad pose
attached back-face vertices follow the anchor
membrane surface follows the moving pad instead of staying world-static
front tet vertices are not in the attachment set
free/front surface deformation residuals are recorded
```

Passing 7c in default mode means only:

```text
UIPC membrane follows the commanded gripper-opening pad pose and is no longer frozen in world space.
```

Passing 7c with `--uipc_follow_mode back_face_attachment` means the same follow is achieved through the back-face UIPC attachment instead of rigidly rewriting every membrane vertex.

It does not mean:

```text
UIPC contact is validated.
UIPC force is calibrated.
pressure or fxyz is valid.
```
