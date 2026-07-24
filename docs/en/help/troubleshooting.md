<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/help/troubleshooting.md">简体中文</a>
</p>

# Troubleshooting

Start with the smallest failing layer. Do not debug a full V6.2 run until imports, lightweight tests, and V1 have been checked.

## Collect basic diagnostics

From `active-isaaclab-2.1.1/`:

```bash
printf 'ISAACLAB_PATH=%s\n' "$ISAACLAB_PATH"
nvidia-smi
nvcc --version
cmake --version
git --version

./run.sh --python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.platform())"
./run.sh --python -c "import isaaclab, openworldtactile, openworldtactile_assets, openworldtactile_uipc, uipc; print('uipc=', uipc.__file__)"
```

Save the full command and traceback. For native crashes, include the last simulator/UIPC log lines and whether the same command fails headless.

## `ISAACLAB_PATH` or Python is wrong

Symptoms include:

- `Unable to find the Isaac Sim directory`;
- `Unable to find any Python executable`;
- `ModuleNotFoundError: isaaclab`;
- `run.sh` prints an unexpected Python path.

Check:

```bash
test -d "$ISAACLAB_PATH"
test -f "$ISAACLAB_PATH/isaaclab.sh"
ls -l "$ISAACLAB_PATH/_isaac_sim/python.sh"
./run.sh --python -c "import sys; print(sys.executable)"
```

For an Isaac Sim pip installation, activate the environment containing `isaacsim-rl` before running the wrapper. For a binary installation, follow the Isaac Lab guide to create the `_isaac_sim` connection. Do not point `ISAACLAB_PATH` at this OpenWorldTactile repository.

## Project package import fails

Reinstall into the interpreter printed by `run.sh`:

```bash
./run.sh --install all
./run.sh --python -m pip show \
  openworldtactile openworldtactile-assets openworldtactile-tasks openworldtactile-uipc
```

Package display names may be normalized with hyphens by pip. If only one package is missing, inspect the corresponding install error rather than adding source directories manually to `PYTHONPATH`.

If `openworldtactile_tasks` fails while importing an optional UIPC task, verify `uipc` first. The UIPC task registration is conditional; other task registrations may still appear.

## UIPC build failures

### CMake cannot find vcpkg

```bash
export VCPKG_ROOT=/absolute/path/to/vcpkg
export CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
test -f "$CMAKE_TOOLCHAIN_FILE"
```

The bundled CMake logic requires a valid toolchain. A path to the vcpkg directory itself is not the same as a path to `vcpkg.cmake`.

### CMake is too old

The bundled libuipc `CMakeLists.txt` requires CMake 3.26 or later:

```bash
cmake --version
```

Make sure the `cmake` found in the same shell is the version used by the build.

### CUDA compiler or architecture errors

Check both the system toolkit and driver:

```bash
which nvcc
nvcc --version
nvidia-smi
```

If `CMAKE_CUDA_ARCHITECTURES` was set, confirm it is a valid CMake architecture number for the target GPU. Unset an incorrect override and let CMake use its native selection, or set the correct value. The Isaac Sim runtime CUDA libraries and the system toolkit used to compile libuipc are different layers; both must be compatible with the driver.

### `uipc` builds but cannot import

```bash
./run.sh --python -c "import sys; print(sys.executable); import uipc; print(uipc.__file__)"
```

Common causes are building with a different Python interpreter, an incompatible C++ runtime, missing shared libraries, or reusing a binary from another machine. Rebuild using `./run.sh --install all` from the intended environment. Avoid masking ABI failures with arbitrary `LD_LIBRARY_PATH` changes.

### Build is slow or runs out of memory

libuipc compiles CUDA/C++ dependencies and can be resource intensive. Close other GPU/compiler workloads and ensure adequate disk/RAM. The package setup currently chooses a parallel build internally; there is no documented wrapper flag to change that value. Capture the failing compiler line when reporting the issue.

## Isaac Sim does not open or hangs at first launch

- Run a standalone Isaac Sim example from the external installation first.
- Confirm the GPU and driver satisfy the Isaac Sim 4.5 requirements.
- Expect the first launch to build shader/extension caches.
- On a server without display access, add `--headless`.
- Camera tasks need `--enable_cameras` even when headless.
- Check free disk space for Isaac/Omniverse caches.

If a minimal upstream Isaac Sim/Isaac Lab example also fails, solve that environment issue before debugging OpenWorldTactile.

## V1 fails or produces unexpected output

### Output directory is empty

Check that `--no_save` and `--loop_forever` were not used. `--loop_forever` intentionally disables saving. Use an explicit writable `--output_dir` and a separate writable `--workspace_dir`.

### `fxyz.npy` is missing

Read the terminal tail and look for an exception. V1 does not use `error.json` as a universal failure contract, so absence of files after a crash is possible. Rerun the reduced quick-start case before the default-resolution case.

### Force field is zero or contains non-finite values

```bash
./run.sh --python -c "import numpy as np; a=np.load('outputs/v1-smoke/fxyz.npy', allow_pickle=False); print(a.shape, np.isfinite(a).all(), float(np.abs(a).max()))"
```

Verify that the trajectory includes indentation, the output contains recorded frames, and metadata matches the command. Non-finite values are a failed run. Preserve the workspace and command for diagnosis; do not use the data.

### Preview video is missing but arrays exist

OpenCV may not have a working MP4 codec. `fxyz.npy`, `metadata.json`, and PNG frames remain the numeric/source outputs. Use the PNG previews or install a codec compatible with the selected OpenCV build.

## V6.2 contract and runtime failures

### “Frozen 7f contract is incomplete”

`--contract_dir` must contain at least the files checked by V6.2, including `vertex_area.npy`, `front_surface_mask.npy`, `rest_surface_pad_local.npy`, and `verdict.json`. Follow the complete [contract workflow](../guides/experiments.md#required-deformation-contract); do not create empty placeholder files.

### “Frozen 7f deformation contract did not pass”

Open `verdict.json` and inspect the failed criterion. Regenerate the V5.7d/7e source runs with the documented 22 × 26 grid and separate workspaces. Do not edit the verdict to bypass validation.

### UIPC substep value is rejected

V6.2 requires `--uipc_substeps_per_record >= 8`. The minimum is part of the contact/coupling design, not a performance suggestion.

### `uipc_timeout.json` appears

The native UIPC substep exceeded `--uipc_substep_timeout_sec`. Inspect the JSON stage/frame/substep, `uipc_substep_time_sec.npy`, and terminal `[V62_SLOW_FRAME]` messages. A timeout is a failed/incomplete run. Test default parameters and a fresh workspace before changing solver/physics values.

### `error.json` appears

Read its traceback and `termination_reason`. Preserve partial arrays for diagnostics, but do not pass them to the acceptance workflow as a successful complete run.

### Validation fails after a completed run

The validator reports the failed physical or structural criterion. Check whether the run was truncated with `--max_formal_frames`, used changed coupling/drive/object parameters, or omitted offline rendering outputs required by an optional check. A completed process is not automatically a passing scenario.

## Task ID is not found

Check registration in isolation:

```bash
./run.sh --python -c "import gymnasium as gym; import openworldtactile_tasks; print([x for x in gym.registry if x.startswith('OpenWorldTactile-')])"
```

If the ID appears here but not in an Isaac Lab train/random/play script, that launcher did not import `openworldtactile_tasks`. Add the import at the external-project placeholder after app launch as described in [Tasks and training](../guides/tasks-and-training.md#connect-an-isaac-lab-launcher).

If only the UIPC RGB task is missing, its guarded import failed. Import its module directly to expose the underlying traceback only after the Isaac app has been launched.

## Camera task errors or out-of-memory failures

- Add `--enable_cameras`.
- Start with `--num_envs 1`.
- Confirm the selected task actually has the requested camera/tactile outputs.
- Reduce environment count before changing image/sensor configuration.
- Record GPU memory and command when reporting an OOM.

Reducing a sensor's resolution changes the observation space and invalidates existing checkpoints unless the model/config is updated accordingly.

## HDF5 viewer/exporter errors

### Group or dataset not found

Inspect the file first:

```bash
./run.sh --python -c "import h5py; f=h5py.File('path/to/file.hdf5'); f.visititems(lambda n,o: print(n, getattr(o,'shape','group'))); f.close()"
```

Then pass the actual `--image-group`, `--streams`, or `--datasets`. Historical HDF5 files do not share one schema.

### JPEG decode fails

The file may use padded compressed rows without the expected `compress_len`, contain a truncated frame, or store a non-JPEG uint8 dataset. Inspect dtype/shape and recorder metadata before modifying the exporter.

### OpenCV window cannot open

Use `--export-dir ... --no-show` on headless systems.

## Legacy SDK or asset errors

The archived route intentionally excludes the vendor camera SDK and some external assets. Set `OWT_SDK_ROOT`/`OWT_ASSET_ROOT` only to legally obtained, compatible copies. A missing SDK cannot be fixed by downloading an unverified DLL/SO into the public repository.

## Report a reproducible issue

Include:

- repository revision and route (`active` or `archive`);
- OS, GPU, driver, CUDA toolkit, Python, Isaac Sim, Isaac Lab, and libuipc versions;
- complete command and environment variables with secrets removed;
- smallest failing example and whether upstream Isaac examples work;
- full traceback/log tail and generated `error.json`, timeout, metadata, or verdict files;
- expected versus actual behavior.

Do not attach proprietary SDK binaries, credentials, or large private datasets to a public issue.
