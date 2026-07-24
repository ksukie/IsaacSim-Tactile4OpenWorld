#!/usr/bin/env python3
"""Build a V2.8 dataset manifest from saved V2.7 sample directories."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMPORTANT_OUTPUT_KEYS = (
    "fxyz",
    "metadata",
    "mechanics_summary",
    "preview_force",
    "preview_fxyz_channels",
    "preview_sequence",
    "fxyz_channels_video",
    "fxyz_camera_sequence",
    "preview_fxyz_camera",
    "compression_map",
    "shear_map",
    "shear_confidence",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _maybe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _existing_files(sample_dir: Path, metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output_files = metadata.get("output_files", {})
    if not isinstance(output_files, dict):
        output_files = {}

    fallback_names = {
        "fxyz": "fxyz.npy",
        "metadata": "metadata.json",
        "mechanics_summary": "mechanics_summary.json",
        "preview_force": "preview_force.png",
        "preview_fxyz_channels": "preview_fxyz_channels.png",
        "preview_sequence": "preview_sequence.mp4",
        "fxyz_channels_video": "fxyz_channels.mp4",
        "fxyz_camera_sequence": "fxyz_camera_sequence.mp4",
        "preview_fxyz_camera": "preview_fxyz_camera.png",
        "compression_map": "compression_map.npy",
        "shear_map": "shear_map.npy",
        "shear_confidence": "shear_confidence.npy",
    }

    files: dict[str, dict[str, Any]] = {}
    for key in IMPORTANT_OUTPUT_KEYS:
        raw_path = output_files.get(key)
        path = Path(str(raw_path)).expanduser() if raw_path else sample_dir / fallback_names[key]
        files[key] = {
            "path": str(path),
            "exists": bool(path.exists()),
        }
    return files


def _sample_dirs(dataset_dir: Path) -> list[Path]:
    if (dataset_dir / "metadata.json").exists():
        return [dataset_dir]
    return sorted(path for path in dataset_dir.iterdir() if path.is_dir())


def summarize_sample(sample_dir: Path) -> dict[str, Any]:
    metadata_path = sample_dir / "metadata.json"
    mechanics_path = sample_dir / "mechanics_summary.json"

    sample: dict[str, Any] = {
        "sample_dir": str(sample_dir),
        "shape": sample_dir.name,
        "status": "ok",
    }

    if not metadata_path.exists():
        sample["status"] = "missing_metadata"
        sample["files"] = _existing_files(sample_dir, {})
        return sample
    if not mechanics_path.exists():
        sample["status"] = "missing_mechanics_summary"

    metadata = _read_json(metadata_path)
    mechanics = _maybe_read_json(mechanics_path)
    peak = mechanics.get("peak", {}) if isinstance(mechanics.get("peak", {}), dict) else {}
    peak_sum = peak.get("by_sum_fz", {}) if isinstance(peak.get("by_sum_fz", {}), dict) else {}
    peak_max = peak.get("by_max_fz", {}) if isinstance(peak.get("by_max_fz", {}), dict) else {}
    peak_compression = (
        peak.get("by_max_compression", {})
        if isinstance(peak.get("by_max_compression", {}), dict)
        else {}
    )

    membrane = metadata.get("membrane", {}) if isinstance(metadata.get("membrane", {}), dict) else {}
    trajectory = metadata.get("trajectory", {}) if isinstance(metadata.get("trajectory", {}), dict) else {}
    force_model = metadata.get("force_model", {}) if isinstance(metadata.get("force_model", {}), dict) else {}
    physical_model = (
        metadata.get("physical_sensor_model", {})
        if isinstance(metadata.get("physical_sensor_model", {}), dict)
        else {}
    )

    sample.update(
        {
            "shape": str(metadata.get("shape", sample_dir.name)),
            "force_core_version": str(metadata.get("script_version", "")),
            "force_source": str(metadata.get("force_source", "")),
            "force_units": str(metadata.get("force_units", "")),
            "native_uipc_contact_force_used": bool(metadata.get("native_uipc_contact_force_used", False)),
            "fxyz_shape": metadata.get("fxyz_shape", mechanics.get("fxyz_shape")),
            "channel_order": metadata.get("channel_order"),
            "display_color_scale": metadata.get("display_color_scale", {}),
            "membrane": {
                "thickness_m": membrane.get("thickness_m"),
                "width_m": membrane.get("width_m"),
                "length_m": membrane.get("length_m"),
                "front_segments_y": membrane.get("front_segments_y"),
                "front_segments_z": membrane.get("front_segments_z"),
            },
            "trajectory": {
                "indent_depth_m": trajectory.get("indent_depth_m"),
                "rub_distance_m": trajectory.get("rub_distance_m"),
                "total_steps_per_cycle": trajectory.get("total_steps_per_cycle"),
                "finite_total_steps": trajectory.get("finite_total_steps"),
                "save_every": trajectory.get("save_every"),
            },
            "physical_sensor_model": {
                "gel_youngs_modulus_mpa": physical_model.get("gel_youngs_modulus_mpa"),
                "gel_poisson_rate": physical_model.get("gel_poisson_rate"),
                "backing_attachment_strength_ratio": physical_model.get("backing_attachment_strength_ratio"),
                "backing_attachment_radius_m": physical_model.get("backing_attachment_radius_m"),
                "contact_regularization_d_hat_m": physical_model.get("contact_regularization_d_hat_m"),
            },
            "force_model": {
                "normal_stiffness": force_model.get("normal_stiffness"),
                "normal_damping": force_model.get("normal_damping"),
                "shear_stiffness": force_model.get("shear_stiffness"),
                "shear_damping": force_model.get("shear_damping"),
                "friction_mu": force_model.get("friction_mu"),
                "surface_reference_max_conservation_error": force_model.get(
                    "surface_reference_max_conservation_error"
                ),
            },
            "mechanics_peak": {
                "by_sum_fz": peak_sum,
                "by_max_fz": peak_max,
                "by_max_compression": peak_compression,
            },
            "peak_step": peak_sum.get("step"),
            "peak_sum_fz": peak_sum.get("sum_fz"),
            "peak_max_fz": peak_sum.get("max_fz"),
            "peak_max_compression_m": peak_sum.get("max_compression_m"),
            "active_area_fraction_at_peak": peak_sum.get("active_area_fraction"),
            "center_sum_fz_ratio_at_peak": peak_sum.get("center_sum_fz_ratio"),
            "force_centroid_px_at_peak": peak_sum.get("force_centroid_px"),
            "mechanics_ranges": mechanics.get("ranges", {}),
            "max_conservation_error": mechanics.get("max_conservation_error"),
            "files": _existing_files(sample_dir, metadata),
        }
    )
    return sample


def build_manifest(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")

    config = _maybe_read_json(dataset_dir / "dataset_config.json")
    samples = [summarize_sample(path) for path in _sample_dirs(dataset_dir)]
    ok_samples = [sample for sample in samples if sample.get("status") == "ok"]

    return {
        "version": "V2.8",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "force_logic": "unchanged_v2_7_surface_deformation_constitutive_reference",
        "run_config": config,
        "sample_count": len(samples),
        "ok_sample_count": len(ok_samples),
        "shapes": [str(sample.get("shape", "")) for sample in samples],
        "samples": samples,
    }


def write_manifest(dataset_dir: Path, manifest_path: Path | None = None) -> Path:
    dataset_dir = dataset_dir.expanduser().resolve()
    manifest = build_manifest(dataset_dir)
    output_path = manifest_path.expanduser().resolve() if manifest_path else dataset_dir / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a V2.8 dataset manifest from saved sample directories.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    output_path = write_manifest(args.dataset_dir, args.manifest)
    manifest = _read_json(output_path)
    print(
        "[INFO] V2.8 manifest written: "
        f"{output_path} samples={manifest.get('sample_count')} ok={manifest.get('ok_sample_count')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
