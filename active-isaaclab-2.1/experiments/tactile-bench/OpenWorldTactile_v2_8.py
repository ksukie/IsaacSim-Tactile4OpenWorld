#!/usr/bin/env python3
"""V2.8 dataset generator.

This script does not change the force model. It runs the frozen V2.7 save
pipeline for a list of probe geometries, then writes a dataset manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from build_v2_8_manifest import write_manifest


DEFAULT_SHAPES = ("edged_box", "hollow_frame", "cross_ridge", "bar_ridge", "dot_array")
SUPPORTED_SHAPES = set(DEFAULT_SHAPES)
DEFAULT_PYTHON_BIN = sys.executable


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a float, got {raw!r}") from exc


def _default_dataset_dir() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"/tmp/openworldtactile_dataset_v2_8_{stamp}"


def _parse_shapes(raw: str) -> list[str]:
    tokens = raw.replace(",", " ").split()
    shapes = [token.strip() for token in tokens if token.strip()]
    if not shapes:
        raise SystemExit("At least one shape is required.")
    unsupported = [shape for shape in shapes if shape not in SUPPORTED_SHAPES]
    if unsupported:
        supported = ", ".join(DEFAULT_SHAPES)
        raise SystemExit(f"Unsupported shape(s): {', '.join(unsupported)}. Supported: {supported}")
    return shapes


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _script_dir().parents[2]


def _clean_extra_args(extra_args: list[str]) -> list[str]:
    if extra_args and extra_args[0] == "--":
        return extra_args[1:]
    return extra_args


def _write_config(path: Path, config: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, allow_nan=False)


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _sample_outputs_ready(sample_dir: Path) -> bool:
    required = ("fxyz.npy", "mechanics_summary.json", "metadata.json")
    return all((sample_dir / name).exists() for name in required)


def _run_sample_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    sample_dir: Path,
    close_grace_sec: float,
) -> None:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    saw_complete = False

    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if "[INFO] OpenWorldTactileBench V2.7 complete:" not in line:
                continue

            saw_complete = True
            deadline = time.monotonic() + max(float(close_grace_sec), 0.0)
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.25)

            if process.poll() is None and _sample_outputs_ready(sample_dir):
                print(
                    "[WARN] V2.7 finished writing outputs but Isaac shutdown is still blocking; "
                    "terminating this sample process and continuing.",
                    flush=True,
                )
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
                return

        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    if saw_complete and not _sample_outputs_ready(sample_dir):
        raise RuntimeError(f"V2.7 printed complete, but required outputs are missing in {sample_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a V2.8 tactile dataset by running the frozen V2.7 save pipeline."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path(os.environ.get("DATASET_DIR", _default_dataset_dir())),
        help="Dataset output root. Defaults to a timestamped /tmp/openworldtactile_dataset_v2_8_* directory.",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default=os.environ.get("SHAPES", " ".join(DEFAULT_SHAPES)),
        help="Space- or comma-separated shape list.",
    )
    parser.add_argument("--indent_depth_mm", type=float, default=_env_float("INDENT_DEPTH_MM", 0.2))
    parser.add_argument("--membrane_thickness_mm", type=float, default=_env_float("MEMBRANE_THICKNESS_MM", 0.5))
    parser.add_argument("--fixed_fz_max", type=float, default=_env_float("DISPLAY_TACTILE_FIXED_FZ_MAX", 1.5e-6))
    parser.add_argument(
        "--fixed_shear_max",
        type=float,
        default=_env_float("DISPLAY_TACTILE_FIXED_SHEAR_MAX", 4.0e-8),
    )
    parser.add_argument("--python_bin", type=str, default=os.environ.get("PYTHON_BIN", DEFAULT_PYTHON_BIN))
    parser.add_argument(
        "--close_grace_sec",
        type=float,
        default=_env_float("CLOSE_GRACE_SEC", 20.0),
        help="Seconds to wait after V2.7 reports complete before terminating a stuck Isaac shutdown.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=os.environ.get("SKIP_EXISTING", "0") == "1",
        help="Skip sample directories that already contain fxyz.npy.",
    )
    parser.add_argument(
        "--allow_existing",
        action="store_true",
        default=os.environ.get("ALLOW_EXISTING", "0") == "1",
        help="Allow writing into sample directories that already contain fxyz.npy.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running simulation.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are passed to run_v2_7_physical_save.sh.",
    )
    args = parser.parse_args()

    if args.indent_depth_mm < 0.0:
        parser.error("--indent_depth_mm must be >= 0.")
    if args.membrane_thickness_mm <= 0.0:
        parser.error("--membrane_thickness_mm must be > 0.")
    if args.fixed_fz_max < 0.0 or args.fixed_shear_max < 0.0:
        parser.error("--fixed_fz_max and --fixed_shear_max must be >= 0.")
    if args.close_grace_sec < 0.0:
        parser.error("--close_grace_sec must be >= 0.")
    if args.skip_existing and args.allow_existing:
        parser.error("--skip_existing and --allow_existing cannot be used together.")

    shapes = _parse_shapes(args.shapes)
    extra_args = _clean_extra_args(list(args.extra_args))
    dataset_dir = args.dataset_dir.expanduser().resolve()
    save_runner = _script_dir() / "run_v2_7_physical_save.sh"
    if not save_runner.exists():
        raise FileNotFoundError(f"Missing V2.7 save runner: {save_runner}")

    config = {
        "version": "V2.8",
        "force_logic": "unchanged_v2_7_surface_deformation_constitutive_reference",
        "dataset_dir": str(dataset_dir),
        "shapes": shapes,
        "indent_depth_mm": float(args.indent_depth_mm),
        "membrane_thickness_mm": float(args.membrane_thickness_mm),
        "display_tactile_fixed_fz_max": float(args.fixed_fz_max),
        "display_tactile_fixed_shear_max": float(args.fixed_shear_max),
        "close_grace_sec": float(args.close_grace_sec),
        "python_bin": str(args.python_bin),
        "extra_args": extra_args,
    }

    if args.dry_run:
        print("[INFO] V2.8 dry run", flush=True)
    else:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_config(dataset_dir / "dataset_config.json", config)

    for shape in shapes:
        sample_dir = dataset_dir / shape
        fxyz_path = sample_dir / "fxyz.npy"
        if fxyz_path.exists() and args.skip_existing:
            print(f"[INFO] Skipping existing sample: {shape} ({sample_dir})", flush=True)
            continue
        if fxyz_path.exists() and not args.allow_existing:
            raise SystemExit(
                f"Sample already exists: {sample_dir}. Use --skip_existing or --allow_existing, "
                "or choose a new --dataset_dir."
            )

        cmd = [str(save_runner), shape, *extra_args]
        env = os.environ.copy()
        env.update(
            {
                "PYTHON_BIN": str(args.python_bin),
                "OUTPUT_DIR": str(sample_dir),
                "INDENT_DEPTH_MM": _format_float(args.indent_depth_mm),
                "MEMBRANE_THICKNESS_MM": _format_float(args.membrane_thickness_mm),
                "DISPLAY_TACTILE_FIXED_FZ_MAX": _format_float(args.fixed_fz_max),
                "DISPLAY_TACTILE_FIXED_SHEAR_MAX": _format_float(args.fixed_shear_max),
            }
        )

        print(f"[INFO] V2.8 sample start: shape={shape}, output={sample_dir}", flush=True)
        if args.dry_run:
            env_preview = (
                f"OUTPUT_DIR={sample_dir} INDENT_DEPTH_MM={args.indent_depth_mm} "
                f"MEMBRANE_THICKNESS_MM={args.membrane_thickness_mm} "
                f"DISPLAY_TACTILE_FIXED_FZ_MAX={args.fixed_fz_max} "
                f"DISPLAY_TACTILE_FIXED_SHEAR_MAX={args.fixed_shear_max}"
            )
            print(f"       {env_preview} {' '.join(cmd)}", flush=True)
            continue

        _run_sample_command(
            cmd,
            cwd=_repo_root(),
            env=env,
            sample_dir=sample_dir,
            close_grace_sec=float(args.close_grace_sec),
        )
        print(f"[INFO] V2.8 sample done: shape={shape}", flush=True)

    if args.dry_run:
        print("[INFO] Dry run complete. No files were written.", flush=True)
        return

    manifest_path = write_manifest(dataset_dir)
    print(f"[INFO] V2.8 dataset complete: {dataset_dir}", flush=True)
    print(f"[INFO] V2.8 manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] V2.8 sample command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        raise
