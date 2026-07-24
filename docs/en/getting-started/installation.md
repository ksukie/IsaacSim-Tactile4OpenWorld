<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/getting-started/installation.md">简体中文</a>
</p>

# Installation

This guide installs the current `active-isaaclab-2.1.1/` route on Linux. The repository extends an existing Isaac Lab installation; it does not install Isaac Sim or Isaac Lab for you.

## 1. Check the environment

Use this combination for the first setup:

| Component | Project target | Notes |
|---|---|---|
| Operating system | Linux with Bash | The wrapper and documented UIPC build path are Linux-oriented. |
| Isaac Sim | 4.5.0 | Install and verify it before this project. |
| Isaac Lab | 2.1.1 | Keep it outside this repository. |
| Python | 3.10 | Use the interpreter supplied by Isaac Sim or the activated Isaac Lab environment. |
| GPU and driver | Compatible NVIDIA RTX GPU and driver | Follow the Isaac Sim 4.5 requirements; do not infer compatibility from a successful Python import. |
| UIPC build tools | CMake 3.26+, CUDA toolkit, C++20 compiler, vcpkg | Required by the bundled libuipc build. |

The upstream Isaac Lab 2.1.1 guide documents an Isaac Sim 4.5 setup tested on Ubuntu 22.04. The repository also contains historical Ubuntu 24.04 migration notes, but that combination is not the public compatibility baseline. Check the official [Isaac Sim 4.5 requirements](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html) and [Isaac Lab 2.1.1 binary installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/setup/installation/binaries_installation.html) before selecting drivers or system packages.

> [!NOTE]
> Isaac Sim 4.5 supports native Windows upstream, but this repository does not currently provide a tested PowerShell installation path for its Bash wrapper and bundled UIPC build. Use Linux for the documented workflow.

## 2. Install Isaac Sim and Isaac Lab

Follow the upstream Isaac Lab guide to:

1. install and launch Isaac Sim 4.5 successfully;
2. clone Isaac Lab and check out `v2.1.1`;
3. connect Isaac Lab to the Isaac Sim installation as described upstream;
4. create or activate the Isaac Lab Python environment.

At the end, choose the absolute Isaac Lab root and verify it. For a binary installation, `_isaac_sim/python.sh` should normally exist:

```bash
export ISAACLAB_PATH=/absolute/path/to/IsaacLab

test -f "$ISAACLAB_PATH/isaaclab.sh"
test -f "$ISAACLAB_PATH/_isaac_sim/python.sh"
```

If Isaac Sim was installed from Python packages, `_isaac_sim/python.sh` may not exist; activate that environment before using `run.sh`. The wrapper detects an active Conda environment or an installed `isaacsim-rl` package.

## 3. Prepare the UIPC toolchain

The primary tactile bench compiles the bundled libuipc source. Its local build guide declares CMake 3.26+, CUDA 12.4+, vcpkg, and a suitable compiler as prerequisites. Read the bundled [libuipc Linux build notes](../../../active-isaaclab-2.1.1/packages/uipc/libuipc/docs/build_install/linux.md) for the upstream details.

Make vcpkg visible to CMake:

```bash
export VCPKG_ROOT=/absolute/path/to/vcpkg
export CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

cmake --version
nvcc --version
test -f "$CMAKE_TOOLCHAIN_FILE"
```

`CMAKE_CUDA_ARCHITECTURES` is optional. Set it only when you know the architecture value for the target GPU:

```bash
export CMAKE_CUDA_ARCHITECTURES=<your-cmake-architecture-number>
```

The UIPC build may download C++ dependencies through vcpkg and can take substantially longer than the pure-Python packages.

## 4. Get the project

```bash
git clone <repository-url> IsaacSim-Tactile4OpenWorld
cd IsaacSim-Tactile4OpenWorld/active-isaaclab-2.1.1
```

Replace `<repository-url>` with the published Git URL. Keep the repository and Isaac Lab in separate directories.

## 5. Install project packages

For the complete current route, including UIPC:

```bash
./run.sh --install all
```

This installs four editable packages into the Isaac Lab/Isaac Sim Python environment:

| Package | Purpose |
|---|---|
| `openworldtactile` | sensor interfaces and tactile simulation approaches |
| `openworldtactile_assets` | robot, sensor, and USD asset configurations |
| `openworldtactile_tasks` | registered Isaac Lab environments and agent configs |
| `openworldtactile_uipc` plus `uipc` | Isaac Lab integration and compiled libuipc bindings |

`./run.sh --install` without `all` skips the UIPC build. Use that only when working on code that does not import UIPC; it is insufficient for the tactile-bench quick start and UIPC task variants.

## 6. Verify the installation

```bash
./run.sh --python -c "import openworldtactile; import openworldtactile_assets; import openworldtactile_uipc; import uipc; print('OpenWorldTactile imports: OK')"
```

Then run one dependency-light unit test:

```bash
./run.sh --python -m unittest discover \
  -s experiments/tactile-bench \
  -p "test_membrane_local_frame.py" -v
```

A passing import confirms that Python can locate the packages and compiled binding. It does not validate GPU contact simulation. Continue with the [quick start](quick-start.md) for a simulation-level check.

## Install outcomes

| Result | Meaning | Next action |
|---|---|---|
| All imports pass | Editable packages and UIPC binding are discoverable | Run the quick start |
| `uipc` cannot be imported | The native build did not install into the selected Python environment | Check vcpkg/CMake/CUDA and the active interpreter |
| `isaaclab` cannot be imported | `ISAACLAB_PATH` or the active environment is wrong | Recheck the upstream Isaac Lab installation |
| Build stops while resolving vcpkg packages | The toolchain or network dependency step failed | See [troubleshooting](../help/troubleshooting.md#uipc-build-failures) |

Do not copy a compiled UIPC module from another machine unless its OS, Python ABI, CUDA toolchain, GPU target, and dependent libraries are known to match.

## Next step

Run [Quick start](quick-start.md). For version limits and validation claims, see [Compatibility](../reference/compatibility.md).
