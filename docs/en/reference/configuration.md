<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/reference/configuration.md">简体中文</a>
</p>

# Configuration reference

The project uses environment variables for external locations and command-line arguments for run-specific settings. It does not have one global project configuration file.

## Environment variables

| Variable | Route | Purpose | Required? |
|---|---|---|---|
| `ISAACLAB_PATH` | mainline | absolute root of external Isaac Lab 2.1.1 | yes for `run.sh` |
| `VCPKG_ROOT` | UIPC build | vcpkg checkout; also used by bundled CMake logic | normally |
| `CMAKE_TOOLCHAIN_FILE` | UIPC build | path to `vcpkg.cmake` | yes unless an accepted default is discoverable |
| `CMAKE_CUDA_ARCHITECTURES` | UIPC build | optional CMake CUDA architecture list | no; set only when known |
| `OWT_ASSET_ROOT` | archive | external legacy tactile asset root | only for affected scripts |
| `OWT_SDK_ROOT` | archive/selected prototypes | externally obtained camera SDK root | only for SDK-dependent scripts |
| `PYTHON_BIN` | selected historical scripts | override subprocess Python | optional |

`run.sh` sets `OWT_PATH` internally to the mainline directory. Users normally should not set it.

Example shell session:

```bash
export ISAACLAB_PATH=/opt/IsaacLab-2.1.1
export VCPKG_ROOT=/opt/vcpkg
export CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
```

Use absolute paths. Quote variables when a path may contain spaces.

## `run.sh` commands

Run from `active-isaaclab-2.1.1/`:

| Command | Behavior |
|---|---|
| `./run.sh --install` | editable-install core, assets, and tasks packages |
| `./run.sh --install all` | install the above and compile/install UIPC |
| `./run.sh --python <args>` | invoke the selected Isaac Sim/active-environment Python |
| `./run.sh --sim <args>` | launch Isaac Sim with the project package folder as an extension folder |
| `./run.sh --test <pytest-args>` | run the configured core, task, and tactile-bench pytest roots |
| `./run.sh --format` | run repository pre-commit hooks; may install `pre-commit` |
| `./run.sh --vscode` | use Isaac Lab's VS Code settings helper when present |
| `./run.sh --conda [NAME]` | create a Python 3.10 Conda environment using the wrapper's legacy helper |
| `./run.sh --help` | print wrapper help; the current script exits with status 1 after printing |

Prefer the upstream Isaac Lab environment instructions over `--conda` for a new installation. The wrapper chooses Python in this order: an active Conda environment, the Isaac Lab `_isaac_sim/python.sh`, then a system Python containing `isaacsim-rl`.

When the target script contains Isaac Lab's `AppLauncher` argument registration and no rendering mode was supplied, `run.sh --python` appends `--rendering_mode performance`.

## Common Isaac launcher flags

Exact flags come from the pinned Isaac Lab `AppLauncher`; use the target script's `--help`. Common options used by this project include:

| Flag | Use |
|---|---|
| `--headless` | run without a visible application window |
| `--enable_cameras` | enable camera sensors, including in headless task runs |
| `--device cuda:0` | choose an Isaac simulation device where supported |
| `--rendering_mode performance` | reduce rendering cost for non-visual runs |

An experiment may add its own `--render_viewport`; it is distinct from the AppLauncher headless flag.

## Common tactile-bench arguments

| Argument | Meaning | Guidance |
|---|---|---|
| `--output_dir PATH` | durable run output | always set explicitly; use a new path per run |
| `--workspace_dir PATH` | UIPC scratch/workspace | keep separate from output and unique per process |
| `--render_viewport` | render the online scene | omit for headless throughput |
| `--render_every N` | render every Nth step | larger N reduces display work |
| `--no_save` | disable saved outputs in supported scripts | use only for observation/debugging |
| `--loop_forever` | repeat until stopped | not suitable for reproducible finite datasets |
| `--log_every N` | log cadence in supported scripts | larger N reduces terminal volume |
| `--fail_on_verdict_fail` / `--fail_on_failure` | return a failure when acceptance fails | use in reproducibility/CI runs |

Argument names are per-script. Check `--help` before reusing a flag from another stage.

## V1 controls

Important categories:

- geometry: `--shape`, `--tool_radius_mm`, membrane dimensions;
- motion: `--indent_depth_mm`, `--rub_distance_mm`, step counts;
- discretization: `--front_segments_y`, `--front_segments_z`, `--thickness_segments`, `--tet_edge_length_r`;
- constitutive estimate: normal/shear stiffness and damping, friction;
- output: tactile width/height, save and preview cadence.

Reducing mesh resolution is useful for a setup smoke test but changes the numerical experiment. Do not compare low-resolution and default results as if only runtime changed.

## V6.2 controls

Required external path:

```text
--contract_dir <passing V5.7f contract>
```

Important coupling controls include:

- `--uipc_substeps_per_record` (minimum 8);
- `--uipc_feedback_relaxation`;
- `--uipc_feedback_force_limit_n`;
- gripper drive stiffness, damping, effort, velocity, and target opening;
- object dimensions, mass, friction, and initial pose;
- solver timeout and slow-frame reporting thresholds.

Changing these values changes the physical/coupling scenario. Store the full command and generated metadata.

## Configuration precedence

For most experiment scripts:

1. explicit command-line values override parser defaults;
2. environment variables select external tools/locations;
3. module-level asset/config defaults fill remaining values;
4. generated metadata records the effective subset implemented by that script.

There is no guarantee that two historical scripts use identical names, defaults, or precedence. Do not copy a configuration dictionary between versions without checking source and output metadata.
