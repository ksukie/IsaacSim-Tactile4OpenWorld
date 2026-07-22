# V5 new 7f deformation field probe

> Legacy single-run diagnostic: this script is retained for prior 7e records. The formal frozen 7f
> boundary is now `OpenWorldTactile_v5_new_7f_deformation_contract_probe.py`; see
> `UIPC_Pad_v5_new_7f_deformation_contract_probe.md`.

## Scope

`OpenWorldTactile_v5_new_7f_deformation_field_probe.py` is an offline extraction and validation stage for
a completed, passing 7e run:

```text
7e UIPC surface history
  -> Pad-local displacement u = X(t) - X(rest)
  -> normal compression max(-u_x, 0)
  -> shear displacement [u_y, u_z]
  -> deformation-region mask
  -> localization and recovery checks
```

It does not run another physics solver and does not estimate force, pressure, stiffness, damping, or
Fxyz. Keeping extraction separate prevents changes to the validated 7e contact simulation.

## Coordinate contract

```text
frame:        UIPC_Pad local
+X:           outward contact normal
compression:  max(-u_x, 0)
shear:        [u_y, u_z]
NPY units:    meters
JSON units:   millimeters
```

`contact_deformation_mask.npy` is a displacement-threshold region on the free front surface. It is
not a pressure mask. `geometric_contact_mask.npy` preserves the separate 7e signed-gap contact mask.

## Run

First create a finite 7e record:

```bash
cd "${OWT_ROOT}"

"${PYTHON_BIN:-python}" \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7e_indenter_deformation.py \
  --headless \
  --indentation_levels_mm 0,0.2,0.5 \
  --fail_on_verdict_fail \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7e_check \
  --workspace_dir /tmp/openworldtactile_uipc_v5_new_7e_check_ws
```

Then extract the deformation field:

```bash
python \
  experiments/tactile-bench/OpenWorldTactile_v5_new_7f_deformation_field_probe.py \
  --input_dir /tmp/openworldtactile_uipc_v5_new_7e_check \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7f_deformation_field \
  --fail_on_verdict_fail
```

## Acceptance

The probe checks:

```text
source 7e verdict passed
all displacement values are finite
normal deformation is nonzero
normal peak lies at the geometric contact region
far-field normal displacement is small relative to the peak
the deformation mask does not cover the full front surface
the attached back surface remains fixed
normal hold response increases with indentation
the front surface recovers after unloading
```

The tested 0.5 mm record produced:

```text
actual indentation                    0.498376 mm
peak normal compression               0.499565 mm
peak-to-contact distance              0.000000 mm
center mean normal compression        0.221306 mm
far-field P95 normal compression      0.000930 mm
far-field / peak ratio                0.001862
deformed front fraction               0.300699
back displacement                     0.005418 mm
final recovery                        0.000026 mm
```

The contact-point shear/normal ratio is also recorded as a diagnostic. It is deliberately not an
acceptance criterion and must not be interpreted as tangential force before a constitutive model and
normal-only symmetry audit are completed.

## Outputs

```text
surface_displacement_pad_local.npy    # T x N x 3
vertex_displacement.npy               # alias of the full displacement field
normal_displacement.npy               # signed u_x, T x N
normal_compression.npy                # max(-u_x, 0), T x N
shear_displacement.npy                # [u_y, u_z], T x N x 2
shear_magnitude.npy                   # T x N
displacement_magnitude.npy            # T x N
contact_deformation_mask.npy          # thresholded free-surface deformation
geometric_contact_mask.npy            # copied 7e contact-region diagnostic
radial_profile_peak_hold.npy          # radius, normal mean/max, shear mean, count
verdict.json
summary.json
```

## Next stage

Before adding a constitutive force estimator, audit the unexpectedly large tangential displacement in
the pure normal indentation case. Only after the displacement components are physically accepted
should a separate stage introduce normal/shear stiffness, damping, area weights, units, and Fxyz.
