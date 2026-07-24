<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/reference/compatibility.md">简体中文</a>
</p>

# Compatibility and validation status

The project currently targets one mainline combination. Other combinations may work, but are not supported by the user guide until they are reproduced and documented.

## Mainline matrix

| Layer | Target/reference | Status |
|---|---|---|
| Operating system | Ubuntu 22.04-class Linux environment | recommended from the Isaac Lab 2.1.1 / Isaac Sim 4.5 upstream combination; project runtime matrix is not yet automated |
| Isaac Sim | 4.5.0 | pinned mainline target |
| Isaac Lab | 2.1.1 | pinned mainline target |
| Python | 3.10 | required by Isaac Sim 4.5/project package metadata |
| libuipc | bundled source identifies 0.9.0-alpha | compiled locally; no portable binary is distributed |
| UIPC build toolchain | bundled guide requires CMake 3.26+, CUDA 12.4+, vcpkg, C++20 toolchain | target-machine validation required |
| GPU/driver | NVIDIA RTX-class GPU and a driver compatible with Isaac Sim and the selected CUDA toolkit | follow NVIDIA's version-specific requirements |

Official references:

- [Isaac Lab 2.1.1 installation using Isaac Sim binaries](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/setup/installation/binaries_installation.html)
- [Isaac Sim 4.5 system and driver requirements](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html)

The upstream 2.1.1 guide reports Isaac Sim 4.5 testing on Ubuntu 22.04. Older project notes mention Ubuntu 24.04 as a migration environment, but this release does not present that note as a tested public baseline.

## What “validated” means

| Level | Evidence | What it does not prove |
|---|---|---|
| Static | files exist, Python AST parses, local documentation links resolve, asset references are inventoried | imports, native loading, GPU execution, numerical correctness |
| Import | packages and `uipc` import in one Python environment | simulator launch or correct physics |
| Unit | dependency-light numeric tests pass | Isaac/PhysX/UIPC integration |
| Smoke | a short V1 or task run initializes and steps | full scenario acceptance or reproducibility |
| Scenario | full command completes and its validator passes on a recorded environment | support for other GPUs, drivers, or parameter sets |
| Hardware | named hardware/SDK combination is tested with safety controls | safety certification or general device compatibility |

Repository-level static checks are available. They must not be described as runtime validation. The public documentation provides commands for import, unit, V1 smoke, and V6.2 scenario checks, but results depend on the user's external environment.

## Version routes

| Directory | Baseline | Intended use | Support status |
|---|---|---|---|
| `active-isaaclab-2.1.1/` | Isaac Lab 2.1.1 | current packages, UIPC, tasks, and experiments | maintained mainline |
| `archive-isaaclab-2.3.2/` | Isaac Lab 2.3.2 | historical GelSight/SDK reproducibility and migration | archive; no routine security/runtime support |

Do not mix both routes in one Python environment. OpenWorldTactile `V1`–`V6.2` labels are experiment stages inside the 2.1.1 route and have no ordering relationship with the Isaac Lab `2.3.2` archive label.

## Known external boundaries

The repository does not provide:

- Isaac Sim or a complete Isaac Lab checkout;
- a prebuilt `uipc` native module;
- NVIDIA drivers, CUDA toolkit, vcpkg, compiler, or OS packages;
- an authorized copy of the historical OpenWorldTactile camera SDK;
- every Nucleus/Factory/GelSight asset referenced by archived experiments;
- a universal pretrained policy set;
- hardware-safety certification.

The mainline includes project USD assets, bundled libuipc source, and selected third-party source/assets under their own licenses. See [Third-party notices](../../../THIRD_PARTY_NOTICES.md).

## Platform notes

### Linux

Linux/Bash is the documented path. The wrapper, `/tmp` defaults, shell loops, and bundled libuipc Linux instructions assume a Unix-like environment.

### Windows

Isaac Sim and Isaac Lab provide upstream Windows workflows, and libuipc contains upstream Windows build notes. This repository has not integrated those pieces into a tested PowerShell wrapper. Native Windows users must adapt installation and commands and should report their complete toolchain if contributing support.

### Containers and remote/headless systems

Isaac Sim supports upstream container/headless deployments, but this repository does not ship a project container image. Account for GPU passthrough, EULA handling, writable output/UIPC workspace mounts, shader caches, and any network access needed for assets and vcpkg dependencies.

## Changing a pinned version

Treat an Isaac Sim, Isaac Lab, Python, CUDA, or libuipc change as a migration:

1. create a separate environment;
2. rebuild UIPC from source;
3. run import and dependency-light tests;
4. run the V1 smoke test;
5. regenerate and validate the V5.7f contract before V6.2;
6. run scenario validators and record the exact matrix;
7. update both language versions of compatibility and installation docs.

Do not silently broaden the supported matrix based on a successful import or one truncated run.
