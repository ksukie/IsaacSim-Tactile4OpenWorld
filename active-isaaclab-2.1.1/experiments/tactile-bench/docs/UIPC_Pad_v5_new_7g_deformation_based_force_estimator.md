# V5 new 7g deformation-based force estimator

## Scope

`OpenWorldTactile_v5_new_7g_deformation_force_estimator.py` implements the frozen 7g-a model:

```text
7f Pad-local displacement contract
  -> rest-area weighted normal/shear deformation volumes
  -> static diagonal gain F = KQ
  -> Pad-local signed vector score
  -> tactile-frame signed channel score
```

It is a reduced-order deformation-based estimator. It is not FEM stress reconstruction, native UIPC
force replacement, pressure reconstruction, or Newton calibration. Damping is disabled.

## Inputs

The estimator calculation reads only:

```text
surface_displacement_pad_local.npy   [N,3] or [T,N,3]
vertex_area.npy                      [N]
front_surface_mask.npy               [N]
```

`rest_surface` and `current_surface` remain 7f responsibilities. 7g does not create a second rest
frame or repeat world-to-Pad conversion. An optional commanded-indentation vector may be supplied for
validation, but it never enters the force calculation.

The loader rejects negative areas, nonzero area outside the front mask, missing front area, nonfinite
values, and mismatched vertex counts.

## Static estimator

For every front vertex:

```text
d_n = max(-u_x, 0)
s   = clip((d_n - d0) / (d1 - d0), 0, 1)
w   = 3*s^2 - 2*s^3
```

The default fixed thresholds are:

```text
d0 = 0.01 mm
d1 = 0.05 mm
```

Integrated deformation features are:

```text
Qn_raw = sum_i A_i * d_n_i
Qt_raw = sum_i A_i * w_i * [u_y_i, u_z_i]

Qn = max(Qn_raw - Qn0, 0)
Qt = Qt_raw - Qt0
```

All Q values have units `m^3`. Baselines may come from a separate no-contact deformation field or
the first B frames of a sequence.

The static score is:

```text
Fn    = Kn * Qn
Ft_y  = Kt_y * Qt_y
Ft_z  = Kt_z * Qt_z
```

Default gains are `1e9 relative_tactile_unit / m^3`. They are scale choices, not calibrated material
parameters and not Newton conversion.

## Frozen directions

Pad-local uses `+X` as the outward normal. The output is object-on-sensor, so:

```text
force_pad_local = [-Fn, Ft_y, Ft_z]
```

The right-handed tactile frame is:

```text
tactile +X = pad +Y
tactile +Y = pad -Z
tactile +Z = pad -X
```

Therefore:

```text
tactile_force_channels = [force_pad_y, -force_pad_z, -force_pad_x]
```

Compression is positive in tactile channel 2. Tangential components retain sign; no absolute-value
aggregation is used.

## Outputs

```text
normal_compression.npy                       [T,N]
shear_displacement.npy                       [T,N,2]
contact_activation_weight.npy                [T,N]
vertex_deformation_volume_contribution.npy   [T,N,3]
raw_normal_deformation_volume.npy             [T]
raw_shear_deformation_volume.npy              [T,2]
normal_deformation_volume.npy                 [T]
shear_deformation_volume.npy                  [T,2]
force_pad_local.npy                           [T,3]
tactile_force_channels.npy                    [T,3]
metadata.json
manifest.json
verdict.json
```

Vertex contributions are raw pre-baseline deformation-volume contributions in
`[normal, pad-Y shear, pad-Z shear]` order. They are not FEM nodal forces.

## Canonical snapshot

```bash
python \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7g_deformation_force_estimator.py \
  --contract_dir /tmp/openworldtactile_uipc_v5_new_7f_contract_verified \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7g_snapshot \
  --fail_on_verdict_fail
```

## Press-release validation

The following command uses the standardized 7f deformation history. The command history is a
diagnostic reference only:

```bash
python \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7g_deformation_force_estimator.py \
  --contract_dir /tmp/openworldtactile_uipc_v5_new_7f_contract_verified \
  --displacement_path \
    /tmp/openworldtactile_uipc_v5_new_7f_contract_verified/diagnostics/primary_surface_displacement_history_pad_local.npy \
  --baseline_frame_count 2 \
  --commanded_indentation_path \
    /tmp/openworldtactile_uipc_v5_new_7f_7e_refined_smoke/commanded_indentation_mm.npy \
  --normal_only_validation \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7g_verified \
  --fail_on_verdict_fail
```

Because 7g-a is static, loading acceptance uses settled hold responses. Per-frame dynamic ramp Pearson
correlation remains a diagnostic and is not used as a linearity requirement.

## Verified result

The real refined-grid `0.2 mm` 7f history produced:

```text
settled indentation levels               -0.2, 0.0, 0.2 mm
settled normal score                     0.0151, 0.0528, 1.6048 TU
settled loading rank correlation         1.0
settled loading response monotonic       true
peak normal score                        1.6147 TU
final released normal score              0.0 TU
release / peak                           0.0
normal-run net tangent / normal          0.1000
```

All 7g-a checks passed. This validates direction, area integration, relative response, and release;
it does not calibrate Newton units and does not replace the future lateral UIPC contact experiment.

## Regression test

```bash
python -m unittest \
  experiments/tactile-bench/test_v5_new_7g_deformation_force_estimator.py \
  -v
```

The tests cover signed tactile mapping, tangential sign reversal, no-contact shear suppression,
coarse/refined mesh invariance, baseline removal, normal loading increase, and release to zero.
