# V5 new 7f deformation contract probe

## Scope

`OpenWorldTactile_v5_new_7f_deformation_contract_probe.py` is the only standardization boundary between
the UIPC membrane solve and a future 7g estimator:

```text
simulation/membrane_sim_mesh
  -> UIPC surface history in world coordinates
  -> inverse live Pad pose for every frame
  -> Pad-local current surface
  -> u = X(t) - X(rest)
  -> frozen deformation contract
```

The probe does not load `contact_vertex_mask.npy`, penetration, native gradients, proxy force, or
pressure. It does not produce Fxyz. The expected center for the normal-indentation test comes from
the authored tool placement, not detected contact geometry.

## Frozen coordinate contract

```text
frame:                 Pad local
+X:                    outward normal
+Y/+Z:                 tangent axes
displacement:          u = Xt - X0
normal compression:    max(-u_x, 0)
shear displacement:    [u_y, u_z]
NPY length unit:       meter
vertex area unit:      meter^2
```

`vertex_area.npy` is barycentric rest area from front-surface triangles. It is zero outside
`front_surface_mask.npy`. Both producer scripts now save `surface_triangles.npy`, so area and vertex
correspondence use the exact compact UIPC surface topology rather than inferred triangulation.

## Four-case acceptance suite

The contract verdict requires all four cases:

1. The settled tail of the primary 7e `pre_contact` phase has near-zero deformation.
2. A passing 7d no-contact link-motion run moves in world coordinates but has near-zero settled
   Pad-local residual.
3. The primary 7e normal press has a positive central compression peak, small center shear, and a
   fixed back surface. Peak location is compared with the authored indenter center.
4. Five distinct 7e directories have identical rest/topology/mask hashes, less than 5% peak error,
   and less than 5% normalized full-field RMS error.

The rigid-motion check uses settled `hold_*` frames for acceptance. All-frame residual is reported
separately because acceleration can create real inertial deformation even when the coordinate
transform is correct.

## Generate producer records

Run 7d once:

```bash
"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7d_backface_follow.py \
  --headless \
  --no_save_camera_rgb \
  --fail_on_verdict_fail \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7f_rigid \
  --workspace_dir /tmp/openworldtactile_uipc_v5_new_7f_rigid_ws
```

Run 7e five independent times. Each run needs its own output and UIPC workspace directory:

```bash
for repeat in 0 1 2 3 4; do
  "${PYTHON_BIN:-python}" \
    experiments/tactile-bench/OpenWorldTactile_v5_new_7e_indenter_deformation.py \
    --headless \
    --no_save_camera_rgb \
    --membrane_cells_y 22 \
    --membrane_cells_z 26 \
    --indentation_levels_mm 0,0.2,0.5 \
    --fail_on_verdict_fail \
    --output_dir "/tmp/openworldtactile_uipc_v5_new_7f_normal_${repeat}" \
    --workspace_dir "/tmp/openworldtactile_uipc_v5_new_7f_normal_${repeat}_ws"
done
```

The explicit `22 x 26` front-grid resolution is part of the currently validated 7f producer setup
for the default `2 mm x 2 mm` flat punch. The older `10 x 12` grid has approximately `2.08 x 2.10
mm` spacing, so the punch is supported by only one center vertex. In a real smoke run that produced a
center shear/normal ratio of `1.634`. With `22 x 26`, nine vertices support the same punch and the
ratio fell to `0.0515` without changing USD, tool size, material parameters, or coordinate handling.

## Build and validate the contract

```bash
python \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7f_deformation_contract_probe.py \
  --rigid_input_dir /tmp/openworldtactile_uipc_v5_new_7f_rigid \
  --normal_input_dir /tmp/openworldtactile_uipc_v5_new_7f_normal_0 \
  --repeat_input_dir /tmp/openworldtactile_uipc_v5_new_7f_normal_1 \
  --repeat_input_dir /tmp/openworldtactile_uipc_v5_new_7f_normal_2 \
  --repeat_input_dir /tmp/openworldtactile_uipc_v5_new_7f_normal_3 \
  --repeat_input_dir /tmp/openworldtactile_uipc_v5_new_7f_normal_4 \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7f_contract \
  --fail_on_verdict_fail
```

## Primary contract outputs

The primary arrays are one selected maximum-loading hold snapshot, not time histories:

```text
rest_surface_pad_local.npy           [N,3]
current_surface_pad_local.npy        [N,3]
surface_displacement_pad_local.npy   [N,3]
normal_compression.npy               [N]
shear_displacement.npy               [N,2]
vertex_area.npy                      [N]
front_surface_mask.npy               [N]
back_surface_mask.npy                [N]
surface_triangles.npy                [F,3]
front_surface_triangles.npy          [F_front,3]
vertex_id.npy                        [N]
```

Time histories and repeat snapshots live under `diagnostics/` and are not 7g inputs. JSON outputs
include `metadata.json`, `manifest.json`, `contact_center.json`, `radial_profile.json`, and
`verdict.json`.

The only allowed 7g inputs are:

```text
surface_displacement_pad_local.npy
vertex_area.npy
front_surface_mask.npy
```

## Verified UIPC result

A five-run independent `0.2 mm` refined-grid suite produced:

```text
Pad-local reconstruction error          0.00000734 mm
no-contact maximum residual             0.004031 mm
world surface motion                    18.8843 mm
settled Pad-local rigid residual         0.000289 mm
peak normal compression                 0.204608 mm
center shear / normal                   0.05154
back-surface residual                   0.002124 mm
five-run peak relative error            0.000311  (0.0311%)
five-run maximum full-field NRMSE        0.01922   (1.92%)
```

All frozen checks passed. This validates the contract and the refined `0.2 mm` diagnostic condition;
it is not a force result and does not by itself claim that every future indentation or object geometry
has converged.

## Offline regression test

The regression test uses synthetic surfaces and checks rigid translation/rotation removal, exact
output shapes, rest-area quadrature, five-run repeatability, and absence of force output:

```bash
python -m unittest \
  experiments/tactile-bench/test_v5_new_7f_deformation_contract_probe.py \
  -v
```
