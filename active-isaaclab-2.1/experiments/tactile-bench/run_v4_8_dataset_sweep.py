from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_V4_8_SCRIPT = SCRIPT_DIR / "OpenWorldTactile_v4_8_pad_visual_contact.py"
DEFAULT_OUTPUT_ROOT = Path("/tmp/openworldtactile_dataset_v4_8")

MANIFEST_FIELDS = [
    "sample_id",
    "status",
    "return_code",
    "texture_type",
    "texture_height_mm",
    "texture_pitch_mm",
    "indent_depth_mm",
    "rub_axis",
    "friction_mu",
    "output_dir",
    "local_fxyz_path",
    "pressure_video_path",
    "pressure_fx_video_path",
    "pressure_fy_video_path",
    "pressure_fz_video_path",
    "object_texture_video_path",
    "metadata_path",
    "passed",
    "fz_max",
    "fx_range",
    "fy_range",
    "active_pressure_ratio",
    "elapsed_sec",
    "command",
]


def _csv_strings(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _csv_floats(text: str) -> list[float]:
    return [float(item) for item in _csv_strings(text)]


def _format_float(value: float) -> str:
    return f"{float(value):g}"


def _height_tag(value_mm: float) -> str:
    return f"h{int(round(float(value_mm) * 100.0)):03d}"


def _one_decimal_tag(prefix: str, value_mm: float) -> str:
    return f"{prefix}{int(round(float(value_mm) * 10.0)):02d}"


def _mu_tag(value: float) -> str:
    return f"mu{int(round(float(value) * 10.0)):02d}"


def _sample_name(index: int, cfg: dict[str, object], *, include_mu: bool) -> str:
    parts = [
        f"sample_{index:04d}",
        str(cfg["texture_type"]),
        _height_tag(float(cfg["texture_height_mm"])),
        _one_decimal_tag("p", float(cfg["texture_pitch_mm"])),
        _one_decimal_tag("d", float(cfg["indent_depth_mm"])),
        str(cfg["rub_axis"]),
    ]
    if include_mu:
        parts.append(_mu_tag(float(cfg["friction_mu"])))
    return "_".join(parts)


def _iter_configs(args: argparse.Namespace) -> Iterable[dict[str, object]]:
    textures = _csv_strings(args.textures)
    heights = _csv_floats(args.texture_heights_mm)
    pitches = _csv_floats(args.texture_pitches_mm)
    depths = _csv_floats(args.indent_depths_mm)
    axes = _csv_strings(args.rub_axes)
    friction_mus = _csv_floats(args.friction_mus)

    for texture_type, height_mm, pitch_mm, depth_mm, rub_axis, friction_mu in itertools.product(
        textures, heights, pitches, depths, axes, friction_mus
    ):
        if args.collapse_none_texture and texture_type == "none" and (height_mm != heights[0] or pitch_mm != pitches[0]):
            continue
        yield {
            "texture_type": texture_type,
            "texture_height_mm": float(height_mm),
            "texture_pitch_mm": float(pitch_mm),
            "indent_depth_mm": float(depth_mm),
            "rub_axis": rub_axis,
            "friction_mu": float(friction_mu),
        }


def _write_manifest(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def _safe_load_json(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _metadata_output_path(metadata: dict[str, object], key: str, sample_dir: Path) -> Path:
    output_files = metadata.get("output_files", {})
    if isinstance(output_files, dict) and output_files.get(key):
        return Path(str(output_files[key]))
    if key == "object_texture_video":
        if isinstance(output_files, dict) and output_files.get("object_texture_gray_sequence"):
            return Path(str(output_files["object_texture_gray_sequence"]))
        if isinstance(output_files, dict) and output_files.get("object_texture_rgb_sequence"):
            return Path(str(output_files["object_texture_rgb_sequence"]))
    if key == "pressure_video":
        if isinstance(output_files, dict) and output_files.get("pressure_fz_gray_sequence"):
            return Path(str(output_files["pressure_fz_gray_sequence"]))
        if isinstance(output_files, dict) and output_files.get("pressure_fxyz_rgb_sequence"):
            return Path(str(output_files["pressure_fxyz_rgb_sequence"]))
    defaults = {
        "local_fxyz": "local_fxyz.npy",
        "pressure_video": "pressure_fz_gray_sequence.mp4",
        "pressure_fx_gray_sequence": "pressure_fx_gray_sequence.mp4",
        "pressure_fy_gray_sequence": "pressure_fy_gray_sequence.mp4",
        "pressure_fz_gray_sequence": "pressure_fz_gray_sequence.mp4",
        "pressure_fxyz_rgb_sequence": "pressure_fxyz_rgb_sequence.mp4",
        "object_texture_video": "object_texture_gray_sequence.mp4",
        "object_texture_gray_sequence": "object_texture_gray_sequence.mp4",
        "object_texture_rgb_sequence": "object_texture_rgb_sequence.mp4",
        "pressure_mask": "pressure_mask.npy",
    }
    return sample_dir / defaults[key]


def _array_metrics(sample_dir: Path, metadata: dict[str, object] | None) -> dict[str, object]:
    if metadata is None:
        metadata = {}

    local_fxyz_path = _metadata_output_path(metadata, "local_fxyz", sample_dir)
    pressure_path = _metadata_output_path(metadata, "pressure_mask", sample_dir)
    metrics: dict[str, object] = {
        "local_fxyz_path": str(local_fxyz_path),
        "pressure_video_path": str(_metadata_output_path(metadata, "pressure_video", sample_dir)),
        "pressure_fx_video_path": str(_metadata_output_path(metadata, "pressure_fx_gray_sequence", sample_dir)),
        "pressure_fy_video_path": str(_metadata_output_path(metadata, "pressure_fy_gray_sequence", sample_dir)),
        "pressure_fz_video_path": str(_metadata_output_path(metadata, "pressure_fz_gray_sequence", sample_dir)),
        "object_texture_video_path": str(_metadata_output_path(metadata, "object_texture_video", sample_dir)),
        "fz_max": "",
        "fx_range": "",
        "fy_range": "",
        "active_pressure_ratio": "",
    }

    if local_fxyz_path.exists():
        try:
            local_fxyz = np.load(local_fxyz_path)
            metrics["fz_max"] = float(np.max(local_fxyz[..., 2]))
            metrics["fx_range"] = float(np.max(local_fxyz[..., 0]) - np.min(local_fxyz[..., 0]))
            metrics["fy_range"] = float(np.max(local_fxyz[..., 1]) - np.min(local_fxyz[..., 1]))
        except Exception as exc:
            metrics["fz_max"] = f"error:{exc}"
    else:
        force_ranges = metadata.get("force_ranges", {})
        if isinstance(force_ranges, dict):
            fz_min_max = force_ranges.get("fz_min_max")
            fx_min_max = force_ranges.get("fx_min_max")
            fy_min_max = force_ranges.get("fy_min_max")
            if isinstance(fz_min_max, list) and len(fz_min_max) == 2:
                metrics["fz_max"] = fz_min_max[1]
            if isinstance(fx_min_max, list) and len(fx_min_max) == 2:
                metrics["fx_range"] = float(fx_min_max[1]) - float(fx_min_max[0])
            if isinstance(fy_min_max, list) and len(fy_min_max) == 2:
                metrics["fy_range"] = float(fy_min_max[1]) - float(fy_min_max[0])

    if pressure_path.exists():
        try:
            pressure_mask = np.load(pressure_path)
            metrics["active_pressure_ratio"] = float(np.mean(pressure_mask.astype(np.float32)))
        except Exception as exc:
            metrics["active_pressure_ratio"] = f"error:{exc}"

    return metrics


def _build_command(
    args: argparse.Namespace,
    sample_dir: Path,
    cfg: dict[str, object],
    passthrough: list[str],
) -> list[str]:
    command = [
        str(Path(args.python).expanduser()),
        str(Path(args.v4_8_script).expanduser()),
        "--output_dir",
        str(sample_dir),
        "--workspace_dir",
        str(sample_dir / "uipc_workspace"),
        "--membrane_source",
        "asset_runtime_volume",
        "--object_texture_type",
        str(cfg["texture_type"]),
        "--object_texture_height_mm",
        _format_float(float(cfg["texture_height_mm"])),
        "--object_texture_pitch_mm",
        _format_float(float(cfg["texture_pitch_mm"])),
        "--indent_depth_mm",
        _format_float(float(cfg["indent_depth_mm"])),
        "--rub_axis",
        str(cfg["rub_axis"]),
        "--friction_mu",
        _format_float(float(cfg["friction_mu"])),
    ]
    if args.headless:
        command.append("--headless")
    if args.strict_child_sanity:
        command.append("--strict_sanity")
    command.extend(passthrough)
    return command


def _run_sample(command: list[str], sample_dir: Path, timeout_sec: float) -> tuple[int, str, float]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    started = time.perf_counter()
    log_path = sample_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=None if timeout_sec <= 0.0 else timeout_sec,
                check=False,
            )
            return_code = int(proc.returncode)
            status = "ok" if return_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            return_code = 124
            status = "timeout"
            log.write(f"\n[TIMEOUT] command exceeded {timeout_sec:.1f} sec\n")
    return return_code, status, time.perf_counter() - started


def _row_from_sample(
    sample_id: str,
    cfg: dict[str, object],
    sample_dir: Path,
    command: list[str],
    *,
    return_code: int,
    status: str,
    elapsed_sec: float,
) -> dict[str, object]:
    metadata_path = sample_dir / "metadata.json"
    metadata = _safe_load_json(metadata_path)
    metrics = _array_metrics(sample_dir, metadata)
    passed = ""
    if metadata is not None:
        passed = bool(metadata.get("passed", False))

    row = {
        "sample_id": sample_id,
        "status": status,
        "return_code": return_code,
        "texture_type": cfg["texture_type"],
        "texture_height_mm": cfg["texture_height_mm"],
        "texture_pitch_mm": cfg["texture_pitch_mm"],
        "indent_depth_mm": cfg["indent_depth_mm"],
        "rub_axis": cfg["rub_axis"],
        "friction_mu": cfg["friction_mu"],
        "output_dir": str(sample_dir),
        "metadata_path": str(metadata_path),
        "passed": passed,
        "elapsed_sec": round(float(elapsed_sec), 3),
        "command": " ".join(command),
    }
    row.update(metrics)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch dataset sweep for OpenWorldTactile UIPC v4.8. This script only orchestrates subprocess "
            "runs of OpenWorldTactile_v4_8_pad_visual_contact.py and writes dataset_manifest.csv."
        )
    )
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--manifest_name", type=str, default="dataset_manifest.csv")
    parser.add_argument("--v4_8_script", type=str, default=str(DEFAULT_V4_8_SCRIPT))
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--textures", type=str, default="none,stripes,grid,weave,bumps,grooves")
    parser.add_argument("--texture_heights_mm", type=str, default="0.05,0.12,0.18")
    parser.add_argument("--texture_pitches_mm", type=str, default="0.8,1.2,1.6")
    parser.add_argument("--indent_depths_mm", type=str, default="0.4,0.8,1.2")
    parser.add_argument("--rub_axes", type=str, default="y,z")
    parser.add_argument("--friction_mus", type=str, default="0.8")
    parser.add_argument("--collapse_none_texture", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=1)
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--no_headless", dest="headless", action="store_false")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_failure", action="store_true")
    parser.add_argument("--strict_child_sanity", action="store_true")
    parser.add_argument("--timeout_sec", type=float, default=0.0)
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / args.manifest_name
    configs = list(_iter_configs(args))
    if int(args.start_index) < 1:
        raise ValueError("--start_index must be >= 1.")
    if args.max_samples > 0:
        configs = configs[: int(args.max_samples)]
    include_mu = len(_csv_floats(args.friction_mus)) > 1

    planned_samples: list[tuple[str, dict[str, object], Path, list[str]]] = []
    for offset, cfg in enumerate(configs):
        index = int(args.start_index) + offset
        sample_id = _sample_name(index, cfg, include_mu=include_mu)
        sample_dir = output_root / sample_id
        command = _build_command(args, sample_dir, cfg, passthrough)
        planned_samples.append((sample_id, cfg, sample_dir, command))

    rows: list[dict[str, object]] = []
    planned_commands_path = output_root / "planned_commands.txt"
    planned_commands_path.write_text(
        "".join(" ".join(command) + "\n" for _, _, _, command in planned_samples),
        encoding="utf-8",
    )

    print(f"[INFO] v4_8 dataset sweep planned samples: {len(planned_samples)}", flush=True)
    for sample_id, cfg, sample_dir, command in planned_samples:
        if args.dry_run:
            row = _row_from_sample(
                sample_id,
                cfg,
                sample_dir,
                command,
                return_code=0,
                status="dry_run",
                elapsed_sec=0.0,
            )
            rows.append(row)
            _write_manifest(manifest_path, rows)
            print(f"[DRY] {sample_id}: {' '.join(command)}", flush=True)
            continue

        metadata_path = sample_dir / "metadata.json"
        if args.resume and metadata_path.exists():
            row = _row_from_sample(
                sample_id,
                cfg,
                sample_dir,
                command,
                return_code=0,
                status="skipped_existing",
                elapsed_sec=0.0,
            )
            rows.append(row)
            _write_manifest(manifest_path, rows)
            print(f"[SKIP] {sample_id}: metadata exists", flush=True)
            continue

        print(f"[RUN] {sample_id}", flush=True)
        return_code, status, elapsed_sec = _run_sample(command, sample_dir, float(args.timeout_sec))
        row = _row_from_sample(
            sample_id,
            cfg,
            sample_dir,
            command,
            return_code=return_code,
            status=status,
            elapsed_sec=elapsed_sec,
        )
        rows.append(row)
        _write_manifest(manifest_path, rows)
        print(
            "[DONE] "
            f"{sample_id} status={status} passed={row['passed']} "
            f"fz_max={row['fz_max']} active_pressure_ratio={row['active_pressure_ratio']} "
            f"elapsed={elapsed_sec:.1f}s",
            flush=True,
        )
        if status != "ok" and args.stop_on_failure:
            print(f"[STOP] First failure at {sample_id}. See {sample_dir / 'run.log'}", flush=True)
            return return_code if return_code != 0 else 1

    print(f"[INFO] manifest: {manifest_path}", flush=True)
    print(f"[INFO] planned commands: {planned_commands_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
