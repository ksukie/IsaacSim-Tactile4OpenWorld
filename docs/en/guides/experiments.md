<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/guides/experiments.md">简体中文</a>
</p>

# Running experiments

The mainline contains production candidates, validation probes, and historical research stages in the same source tree. Use the documented entry points below instead of selecting a script only by its version-shaped filename.

All commands assume:

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab
cd /absolute/path/to/IsaacSim-Tactile4OpenWorld/active-isaaclab-2.1.1
mkdir -p outputs
```

## Recommended entry points

| Entry point | Use it for | Prerequisites | Output |
|---|---|---|---|
| `OpenWorldTactile_v1.py` | first UIPC membrane/contact run and force-field generation | complete UIPC install | `fxyz.npy`, metadata, image/video previews |
| `OpenWorldTactile_v6_2_grasp.py` | current closed-loop Piper grasp research workflow | UIPC plus a passing V5.7f deformation contract | detailed PhysX/UIPC coupling arrays and validation metadata |
| registered `openworldtactile_tasks` environments | reinforcement learning | matching Isaac Lab training script and agent package | framework-specific logs/checkpoints |
| `tools/data/*.py` | inspect generated HDF5 episodes | `h5py`, OpenCV, NumPy | decoded image directories |

V2 through V6.1b and the many `v5_new_*` scripts document how the current pipeline was developed. Treat them as research probes unless a guide explicitly asks you to run one.

## V1 fixed-membrane bench

V1 is the best first simulation because it is self-contained inside the mainline and does not need a previously generated contract.

### Full default run

```bash
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v1.py \
  --headless \
  --shape dots \
  --indent_depth_mm 0.8 \
  --rub_distance_mm 3.0 \
  --output_dir "$PWD/outputs/v1-dots" \
  --workspace_dir "$PWD/outputs/v1-dots-workspace"
```

Available indenter shapes are `sphere`, `cylinder`, `dots`, `cross_lines`, `wave1`, and `random`. Use a new `--workspace_dir` for independent UIPC runs; do not point concurrent processes at one workspace.

### Headless and interactive modes

- Add `--headless` for a non-windowed run.
- Omit `--headless` and add `--render_viewport` to see the motion.
- Use `--render_every N` to render every Nth simulation step.
- Use `--no_save` for visual inspection without output files.
- `--loop_forever` repeats the trajectory and automatically disables saving.

The saved tactile tensor has shape `[T, 300, 300, 3]`. Its channels are local tangential Y, local tangential Z, and local normal X, exposed as `fx`, `fy`, and `fz`. The unit is `sim_constitutive_force`, not Newton. See [Data and outputs](data-and-outputs.md#v1-output-contract).

## V6.2 closed-loop Piper grasp

V6.2 couples a PhysX-controlled Piper and free cylinder to one Pad-local UIPC membrane. It is an advanced reproducibility workflow, not a one-command first demo.

### Required deformation contract

V6.2 refuses to start unless `--contract_dir` contains a passing V5.7f contract. Building that contract requires one V5.7d no-contact motion run and five independent V5.7e indentation runs.

Set an output root:

```bash
export OWT_CONTRACT_ROOT="$PWD/outputs/v62-contract"
mkdir -p "$OWT_CONTRACT_ROOT"
```

Generate the rigid-motion record:

```bash
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v5_new_7d_backface_follow.py \
  --headless \
  --no_save_camera_rgb \
  --fail_on_verdict_fail \
  --output_dir "$OWT_CONTRACT_ROOT/rigid" \
  --workspace_dir "$OWT_CONTRACT_ROOT/rigid-workspace"
```

Generate five independent normal-indentation records:

```bash
for repeat in 0 1 2 3 4; do
  ./run.sh --python experiments/tactile-bench/OpenWorldTactile_v5_new_7e_indenter_deformation.py \
    --headless \
    --no_save_camera_rgb \
    --membrane_cells_y 22 \
    --membrane_cells_z 26 \
    --indentation_levels_mm 0,0.2,0.5 \
    --fail_on_verdict_fail \
    --output_dir "$OWT_CONTRACT_ROOT/normal-$repeat" \
    --workspace_dir "$OWT_CONTRACT_ROOT/normal-$repeat-workspace"
done
```

Build and validate the frozen contract:

```bash
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v5_new_7f_deformation_contract_probe.py \
  --rigid_input_dir "$OWT_CONTRACT_ROOT/rigid" \
  --normal_input_dir "$OWT_CONTRACT_ROOT/normal-0" \
  --repeat_input_dir "$OWT_CONTRACT_ROOT/normal-1" \
  --repeat_input_dir "$OWT_CONTRACT_ROOT/normal-2" \
  --repeat_input_dir "$OWT_CONTRACT_ROOT/normal-3" \
  --repeat_input_dir "$OWT_CONTRACT_ROOT/normal-4" \
  --output_dir "$OWT_CONTRACT_ROOT/verified" \
  --fail_on_verdict_fail
```

Confirm that `verified/verdict.json` reports `deformation_contract_passed: true`. A V6.2-compatible contract directory must contain at least `vertex_area.npy`, `front_surface_mask.npy`, `rest_surface_pad_local.npy`, and the passing `verdict.json`; V6.2 checks all four before launching the scene.

### Run V6.2

```bash
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v6_2_grasp.py \
  --headless \
  --contract_dir "$OWT_CONTRACT_ROOT/verified" \
  --workspace_dir "$PWD/outputs/v62-grasp-workspace" \
  --output_dir "$PWD/outputs/v62-grasp" \
  --sim_hz 60 \
  --uipc_substeps_per_record 8 \
  --uipc_feedback_relaxation 1.0 \
  --uipc_feedback_force_limit_n 0.25 \
  --slow_frame_threshold_sec 0.5 \
  --log_every 10
```

Keep `--uipc_substeps_per_record` at 8 or higher; the script rejects smaller values. A full default motion contains hundreds of recorded frames. For an initialization-only diagnostic, add `--max_formal_frames 20`, but do not interpret that truncated result as a successful grasp.

For an interactive run, omit `--headless` and add `--render_viewport`. Do not use `--loop_forever` for a reproducible dataset run.

### Render and validate V6.2 output

```bash
./run.sh --python experiments/tactile-bench/render_tactile_field_offline.py \
  --input_dir "$PWD/outputs/v62-grasp" \
  --video_fps 15

./run.sh --python experiments/tactile-bench/validate_v6_2_once.py \
  --input_dir "$PWD/outputs/v62-grasp" \
  --penetration_tolerance_mm 0.15 \
  --precontact_deformation_tolerance_mm 0.25 \
  --minimum_object_lift_mm 20 \
  --fail_on_failure
```

The validator checks structural contracts, coupling recursion, contact response, penetration tolerance, lift, release, and offline field consistency. See [Data and outputs](data-and-outputs.md#v62-output-groups) for the maintained output inventory and interpretation rules; the producing script remains authoritative for additional diagnostic arrays.

## Other experiment groups

| Directory | Contents | Stability |
|---|---|---|
| `experiments/simulation-prototypes/` | Taxim, FOTS, FEM, force-component, and sensor integration probes | exploratory; inspect dependencies before running |
| `experiments/manipulation/` | pick-up, pick-place, insertion, rubbing, and HDF5 recording scripts | scenario-specific research code |
| `experiments/benchmarks/` | rigid/UIPC/tactile performance comparisons | benchmark harnesses; results depend on hardware and settings |
| `experiments/tactile-bench/` | V1–V6.2 UIPC tactile evolution and offline validators | V1 and V6.2 are documented entry points; other files are milestones/probes |

Use the [generated entry-point inventory](../../internal/ENTRYPOINT_MATRIX.md) to locate a script and inspect its static imports. “Directly runnable” in that inventory means only that a main guard and parser were found; it is not a runtime guarantee.

## Reproducible run checklist

For results you intend to compare or publish, record:

- repository revision and experiment filename;
- Isaac Sim, Isaac Lab, Python, CUDA toolkit, driver, GPU, and libuipc versions;
- the complete command and all non-default arguments;
- random seeds, input assets, contract hashes, and calibration files;
- output metadata and validator verdicts;
- whether the run was headless, rendered, truncated, or resumed.

Never label `sim_constitutive_force` or TU-valued fields as Newton unless a documented calibration was applied.
