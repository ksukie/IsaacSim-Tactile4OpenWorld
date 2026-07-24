<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/getting-started/quick-start.md">简体中文</a>
</p>

# Quick start

This walkthrough verifies the installed mainline and produces a small tactile force-field result. Complete [Installation](installation.md), including `./run.sh --install all`, first.

All commands below start in the mainline directory:

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab
cd /absolute/path/to/IsaacSim-Tactile4OpenWorld/active-isaaclab-2.1.1
```

## 1. Verify the selected Python environment

```bash
./run.sh --python -c "import sys; print(sys.executable)"
./run.sh --python -c "import isaaclab, openworldtactile, openworldtactile_assets, openworldtactile_uipc, uipc; print('imports: OK')"
```

The first command should print the Python interpreter from the active Isaac Lab/Isaac Sim environment. If it prints an unexpected interpreter, stop and fix the environment before running simulations.

## 2. Run lightweight algorithm checks

These tests do not launch the simulator:

```bash
./run.sh --python -m unittest discover \
  -s experiments/tactile-bench \
  -p "test_membrane_local_frame.py" -v

./run.sh --python -m unittest discover \
  -s experiments/tactile-bench \
  -p "test_tu_tactile_field.py" -v

./run.sh --python -m unittest discover \
  -s experiments/tactile-bench \
  -p "test_v5_new_7g_deformation_force_estimator.py" -v
```

They check coordinate transforms, tactile-field construction, and the offline deformation estimator. They do not prove that Isaac Sim or UIPC can step on the GPU.

## 3. Run the V1 tactile smoke test

The following reduced-resolution case is intended for setup validation. It presses and releases a sphere against one UIPC membrane and saves a `300 x 300 x 3` tactile field:

```bash
mkdir -p outputs

./run.sh --python experiments/tactile-bench/OpenWorldTactile_v1.py \
  --headless \
  --shape sphere \
  --indent_depth_mm 0.6 \
  --rub_distance_mm 0.0 \
  --front_segments_y 24 \
  --front_segments_z 30 \
  --thickness_segments 3 \
  --tet_edge_length_r 0.05 \
  --warmup_steps 5 \
  --approach_steps 20 \
  --indent_steps 40 \
  --rub_steps 0 \
  --release_steps 20 \
  --output_dir "$PWD/outputs/v1-smoke" \
  --workspace_dir "$PWD/outputs/v1-smoke-workspace"
```

The first Isaac Sim launch may spend time building shader and extension caches. The UIPC solve can also be slow on its first run.

## 4. Check the result

Expected files:

```text
outputs/v1-smoke/
├── fxyz.npy
├── metadata.json
├── preview_force.png
├── preview_sequence.mp4
└── preview_frames/
```

Validate the array and metadata:

```bash
./run.sh --python -c "import json, numpy as np; p='outputs/v1-smoke'; a=np.load(f'{p}/fxyz.npy'); m=json.load(open(f'{p}/metadata.json')); assert a.ndim == 4 and a.shape[1:] == (300, 300, 3); assert np.isfinite(a).all(); print('shape=', a.shape, 'units=', m['force_units'], 'max_abs=', float(np.abs(a).max()))"
```

The expected unit label is `sim_constitutive_force`. These values are not calibrated Newtons. A nonzero maximum shows that the run produced a response; it does not by itself establish physical accuracy.

If `error.json` appears, or any expected file is missing, use [Troubleshooting](../help/troubleshooting.md).

## 5. Optional: watch the simulation

Run without `--headless` and add viewport rendering. `--no_save` avoids writing output during visual inspection:

```bash
./run.sh --python experiments/tactile-bench/OpenWorldTactile_v1.py \
  --shape sphere \
  --indent_depth_mm 0.6 \
  --rub_distance_mm 0.0 \
  --front_segments_y 24 \
  --front_segments_z 30 \
  --thickness_segments 3 \
  --tet_edge_length_r 0.05 \
  --render_viewport \
  --render_every 5 \
  --loop_forever \
  --no_save
```

Close the Isaac Sim window or press `Ctrl+C` in the terminal to stop the loop.

## Where to go next

| Goal | Guide |
|---|---|
| Use the full-resolution V1 settings | [Experiments: V1](../guides/experiments.md#v1-fixed-membrane-bench) |
| Run the advanced V6.2 Piper grasp | [Experiments: V6.2](../guides/experiments.md#v62-closed-loop-piper-grasp) |
| Train a registered task | [Tasks and training](../guides/tasks-and-training.md) |
| Inspect NumPy, video, or HDF5 results | [Data and outputs](../guides/data-and-outputs.md) |
| Understand force axes and package flow | [Architecture](../reference/architecture.md) |
