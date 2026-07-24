<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/guides/tasks-and-training.md">简体中文</a>
</p>

# Tasks and training

`openworldtactile_tasks` provides Gymnasium registrations and agent configuration files. The training implementations themselves come from the external Isaac Lab 2.1.1 checkout.

## Prerequisites

1. Complete the [full installation](../getting-started/installation.md), including UIPC.
2. Install the RL library required by the selected task through Isaac Lab. For example:

   ```bash
   "$ISAACLAB_PATH/isaaclab.sh" -i skrl
   "$ISAACLAB_PATH/isaaclab.sh" -i rl_games
   "$ISAACLAB_PATH/isaaclab.sh" -i rsl_rl
   ```

3. Start with one or a small number of environments. Camera and UIPC observations have significantly higher memory and compute costs than privileged-state tasks.

## List registered environments

Import the project package before reading the Gym registry:

```bash
./run.sh --python -c "import gymnasium as gym; import openworldtactile_tasks; print('\n'.join(sorted(x for x in gym.registry if x.startswith('OpenWorldTactile-'))))"
```

The exact output is authoritative for the installed revision. The current source registers:

| Task ID | Observation/workflow | Agent configs |
|---|---|---|
| `OpenWorldTactile-Ball-Rolling-Tactile-Depth-v1` | tactile depth/camera ball rolling | skrl |
| `OpenWorldTactile-Ball-Rolling-Tactile-RGB-v0` | tactile RGB ball rolling | skrl |
| `OpenWorldTactile-Ball-Rolling-Taxim-Fots-v0` | Taxim plus FOTS ball rolling | skrl |
| `OpenWorldTactile-Ball-Rolling-Tactile-RGB-Uipc-v0` | UIPC tactile RGB ball rolling | skrl; registered only when optional imports succeed |
| `OpenWorldTactile-Ball-Rolling-Privileged-v0` | privileged-state ball rolling | rl_games, rsl_rl, skrl |
| `OpenWorldTactile-Ball-Rolling-Privileged-Reset-with-IK-solver_v0` | privileged ball rolling with IK reset | rl_games, rsl_rl, skrl |
| `OpenWorldTactile-Ball-Rolling-Privileged-Without-Reaching_v0` | privileged ball rolling without reaching phase | rl_games, rsl_rl, skrl |
| `OpenWorldTactile-Factory-PegInsert-Direct-v0` | direct peg insertion | rl_games |
| `OpenWorldTactile-Factory-GearMesh-Direct-v0` | direct gear meshing | rl_games |
| `OpenWorldTactile-Factory-NutThread-Direct-v0` | direct nut threading | rl_games |
| `OpenWorldTactile-Pole-Balancing-Base-v0` | camera/tactile pole balancing | skrl |
| `OpenWorldTactile-Repose-Cube-Allegro-v0` | Allegro in-hand cube reorientation | rl_games, rsl_rl, skrl |

## Connect an Isaac Lab launcher

Unmodified upstream Isaac Lab scripts import `isaaclab_tasks`, but do not automatically import this external task package. Use an external-project launcher or a project-local copy of the upstream script and add the following line after Isaac Sim has been launched, next to the upstream task import/extension-template placeholder:

```python
import isaaclab_tasks  # upstream registrations
import openworldtactile_tasks  # OpenWorldTactile registrations
```

Do not import task modules before `AppLauncher` in scripts that follow Isaac Sim's “launch first” import rule. The Isaac Lab external-project template contains a placeholder at the correct location.

The examples below use `path/to/owt_*.py` to mean a launcher prepared this way. This repository currently supplies task definitions and configs, not duplicated copies of every upstream RL launcher.

## Smoke-test an environment

Create a project-aware copy of Isaac Lab's `scripts/environments/random_agent.py`, add the import above, then run a small case:

```bash
./run.sh --python path/to/owt_random_agent.py \
  --task OpenWorldTactile-Ball-Rolling-Privileged-v0 \
  --num_envs 1 \
  --headless
```

For a camera task, enable cameras:

```bash
./run.sh --python path/to/owt_random_agent.py \
  --task OpenWorldTactile-Ball-Rolling-Tactile-Depth-v1 \
  --num_envs 1 \
  --enable_cameras \
  --headless
```

Confirm that the environment can reset and step before starting a long training run.

## Train with skrl

Prepare a project-aware copy of Isaac Lab's `scripts/reinforcement_learning/skrl/train.py`, then run:

```bash
./run.sh --python path/to/owt_skrl_train.py \
  --task OpenWorldTactile-Ball-Rolling-Tactile-Depth-v1 \
  --num_envs 8 \
  --enable_cameras \
  --headless
```

Use `--max_iterations N` for a short integration test before a full run. Logs are written by the upstream trainer under its `logs/skrl/` convention relative to the working directory.

## Train with rl_games

Prepare a project-aware copy of `scripts/reinforcement_learning/rl_games/train.py`:

```bash
./run.sh --python path/to/owt_rl_games_train.py \
  --task OpenWorldTactile-Factory-PegInsert-Direct-v0 \
  --num_envs 8 \
  --enable_cameras \
  --headless
```

## Train with rsl_rl

Use a task that registers an `rsl_rl_cfg_entry_point`, such as the privileged ball-rolling task:

```bash
./run.sh --python path/to/owt_rsl_rl_train.py \
  --task OpenWorldTactile-Ball-Rolling-Privileged-v0 \
  --num_envs 32 \
  --headless
```

## Play a checkpoint

Prepare the corresponding project-aware `play.py` from the same RL backend and pass the exact task and checkpoint:

```bash
./run.sh --python path/to/owt_skrl_play.py \
  --task OpenWorldTactile-Ball-Rolling-Tactile-Depth-v1 \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --num_envs 1 \
  --enable_cameras
```

Use the same environment ID, algorithm family, observation configuration, and package revision used during training.

## Scaling and reproducibility

- Increase `--num_envs` only after measuring GPU memory and step time.
- Camera tasks require `--enable_cameras`, including headless runs.
- The UIPC task variant is designed for a much smaller environment count than rigid/privileged tasks.
- Preserve the trainer's environment and agent configuration dumps with checkpoints.
- Record seed, package revision, task ID, backend, command, GPU/driver, Isaac versions, and any Hydra overrides.
- Validate a checkpoint in a separate play run; a decreasing training loss alone is not a task-success result.

Task implementations are research environments. Their registration is evidence that a configuration exists, not that every backend/task combination has been rerun in this release.
