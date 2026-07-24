<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/reference/repository-layout.md">简体中文</a>
</p>

# Repository layout

## Top level

```text
IsaacSim-Tactile4OpenWorld/
├── README.md / README.zh-CN.md
├── active-isaaclab-2.1.1/
├── archive-isaaclab-2.3.2/
├── docs/
├── tools/repository/
├── CONTRIBUTING*.md, SECURITY*.md, CITATIONS*.md
├── LICENSE, NOTICE, THIRD_PARTY_NOTICES*.md
└── CITATION.cff
```

| Location | Audience | Contents |
|---|---|---|
| root README | all users | project overview and shortest path to the docs |
| `docs/en/`, `docs/zh-CN/` | users | maintained, mirrored user guides |
| `active-isaaclab-2.1.1/` | users/researchers | current installable packages, experiments, assets, and data tools |
| `archive-isaaclab-2.3.2/` | migration researchers | historical route with external dependency boundaries |
| `docs/internal/`, `tools/repository/` | maintainers | generated inventories, release audit records, and static checks |

## Current mainline

```text
active-isaaclab-2.1.1/
├── run.sh
├── packages/
│   ├── core/       -> openworldtactile
│   ├── assets/     -> openworldtactile_assets + USD/data
│   ├── tasks/      -> openworldtactile_tasks
│   └── uipc/       -> openworldtactile_uipc + bundled libuipc
├── experiments/
│   ├── tactile-bench/
│   ├── manipulation/
│   ├── simulation-prototypes/
│   └── benchmarks/
├── tools/data/
└── vendor/agilex-piper/
```

### Packages

- `packages/core/openworldtactile/`: sensor classes and Taxim/FOTS/FEM approach implementations.
- `packages/assets/openworldtactile_assets/`: Python configs and `data/` containing robot/sensor USD assets.
- `packages/tasks/openworldtactile_tasks/`: environment implementations, Gym registrations, and agent configs.
- `packages/uipc/openworldtactile_uipc/`: Isaac/UIPC integration.
- `packages/uipc/libuipc/`: bundled third-party source with its own licenses and upstream documentation.

### Experiments

- `tactile-bench/`: versioned UIPC tactile research, local APIs, offline estimators, validators, and retained research notes.
- `manipulation/`: scenario-specific pick-up, pick-place, insertion, rubbing, and recording scripts.
- `simulation-prototypes/`: focused sensor/method integration checks.
- `benchmarks/`: performance comparison harnesses.

The many tactile-bench files are not separate packages. They use same-directory imports and shared assets, so copying a single script out of the directory is unsupported.

### Data tools

`tools/data/` contains standalone HDF5 image/tactile viewers and exporters. They operate on completed files and do not collect data by themselves.

### Vendor content

`vendor/agilex-piper/` is upstream integration material. Do not treat its READMEs, paths, or build instructions as OpenWorldTactile user documentation, and avoid modifying it unless the change is an intentional upstream patch with preserved attribution.

## Historical archive

```text
archive-isaaclab-2.3.2/
├── VERSION
├── packages/{contrib,assets-patches}/
├── experiments/{basics,franka,rgb-pipeline,sensors}/
├── hardware-sdk/
├── notes/
└── tools/
```

The archive is not installed by `active-isaaclab-2.1.1/run.sh`. See [Legacy route](../guides/legacy.md).

## Documentation

```text
docs/
├── README.md / README.zh-CN.md
├── en/
│   ├── getting-started/
│   ├── guides/
│   ├── reference/
│   └── help/
├── zh-CN/              # exact topic mirror of en/
├── internal/           # maintainer/generated records
└── media/
```

Every maintained user guide has an English and Simplified Chinese counterpart and a language switch at the top. Internal release records and retained experiment notes are not part of the user-guide contract.

## Where generated files belong

| File type | Recommended location |
|---|---|
| experiment arrays/JSON/previews | an explicit `active-isaaclab-2.1.1/outputs/<run>/` or external dataset directory |
| UIPC workspace | a separate per-run scratch directory |
| RL logs/checkpoints | trainer log directory outside package source |
| build products | package build directory, ignored and not committed |
| public documentation images | `docs/media/` when licensed and intentionally versioned |

Do not write generated results under `packages/**/data/`; that tree is for versioned assets.

## Finding things

From the repository root:

```bash
# Find a script or asset by name
rg --files | rg 'OpenWorldTactile_v6_2|GelSight_Mini'

# Find a task registration
rg -n 'gym.register|OpenWorldTactile-' active-isaaclab-2.1.1/packages/tasks

# Find a command-line option
rg -n 'add_argument.*output_dir' active-isaaclab-2.1.1/experiments

# Find public class definitions
rg -n '^class ' active-isaaclab-2.1.1/packages/{core,uipc}
```

For a full statically generated script list, use the [internal entry-point matrix](../../internal/ENTRYPOINT_MATRIX.md).
