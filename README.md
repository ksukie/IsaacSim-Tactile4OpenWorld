<p align="right">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <img src="docs/media/openworldtactile-logo-lockup-v6.png" alt="OpenWorldTactile logo">
</p>

<h1 align="center">IsaacSim-Tactile4OpenWorld</h1>

<p align="center">
  Contact, deformation, force-field, and visuotactile simulation for Isaac Sim and Isaac Lab.
</p>

<p align="center">
  <a href="docs/en/getting-started/installation.md">Install</a> ·
  <a href="docs/en/getting-started/quick-start.md">Quick start</a> ·
  <a href="docs/README.md">Documentation</a> ·
  <a href="CITATIONS.md">Cite</a>
</p>

> [!IMPORTANT]
> This is a research codebase, not a standalone Isaac Lab distribution. The current route requires an external Isaac Lab 2.1.1 / Isaac Sim 4.5 environment. UIPC experiments also compile the bundled libuipc source. See [compatibility and validation status](docs/en/reference/compatibility.md) before installing.

## What this project provides

OpenWorldTactile brings tactile sensing and contact research into a shared Isaac Lab workflow:

- deformable membrane and rigid/deformable contact simulation with UIPC/libuipc;
- GelSight Mini assets and Taxim, FOTS, FEM, RGB, marker-motion, and force-field approaches;
- AgileX Piper, Franka, and Allegro integrations;
- grasping, lifting, insertion, rolling, and reinforcement-learning environments;
- NumPy, image, video, JSON, and HDF5 inspection utilities.

“Open world” means that experiments can be extended across robots, sensors, objects, and contact conditions. It does not mean the project provides a general-purpose world model or zero-shot control in arbitrary scenes.

<p align="center">
  <img src="docs/media/openworldtactile-hero.png" alt="OpenWorldTactile contact and tactile-signal pipeline">
</p>

## Choose a starting point

| Goal | Start here |
|---|---|
| Install the supported mainline | [Installation](docs/en/getting-started/installation.md) |
| Verify the setup and run one tactile simulation | [Quick start](docs/en/getting-started/quick-start.md) |
| Run the V1 bench or advanced V6.2 grasp workflow | [Experiments](docs/en/guides/experiments.md) |
| Train or inspect registered Isaac Lab tasks | [Tasks and training](docs/en/guides/tasks-and-training.md) |
| Read and export generated results | [Data and outputs](docs/en/guides/data-and-outputs.md) |
| Integrate project packages into another scene | [Custom integration](docs/en/guides/custom-integration.md) |
| Diagnose installation or runtime failures | [Troubleshooting](docs/en/help/troubleshooting.md) |

## First run

After installing the external Isaac Lab environment and the required build tools:

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab
cd active-isaaclab-2.1.1

./run.sh --install all
./run.sh --python -m unittest discover \
  -s experiments/tactile-bench \
  -p "test_membrane_local_frame.py" -v

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

A successful V1 run writes `fxyz.npy`, `metadata.json`, and visual previews under the selected output directory. This smoke test requires the UIPC installation but does not require the V6.2 frozen deformation contract. Follow the [complete quick start](docs/en/getting-started/quick-start.md) for verification steps and expected results.

## Repository layout

```text
active-isaaclab-2.1.1/   Current packages, tasks, assets, and experiments
archive-isaaclab-2.3.2/  Historical GelSight/SDK integration; not the default route
docs/                    Mirrored English and Simplified Chinese user guides
tools/repository/        Maintainer-only static audit and inventory tools
```

The OpenWorldTactile experiment labels (`V1` through `V6.2`) are research milestones, not Isaac Lab versions. See the [experiment lineage](docs/en/reference/experiment-lineage.md) before choosing a historical script.

## Project status and support

The source tree has repository-level static validation, but the public release does not claim that every GPU, driver, UIPC build, task, archived script, or hardware integration has been rerun. When reporting a problem, include your OS, GPU and driver, Isaac Sim and Isaac Lab versions, install command, full traceback, and the smallest reproducing command. Start with the [FAQ](docs/en/help/faq.md) and [troubleshooting guide](docs/en/help/troubleshooting.md).

## Citation, license, and contribution

Use [`CITATION.cff`](CITATION.cff) for project metadata and [`CITATIONS.md`](CITATIONS.md) for method-specific research attribution.

Original project contributions are available under [BSD-3-Clause](LICENSE), but this is a multi-license repository. libuipc bindings, TetGen, GelSight-derived assets, and other bundled components have additional terms. Read [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistribution.

Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md). Report vulnerabilities according to [`SECURITY.md`](SECURITY.md).
