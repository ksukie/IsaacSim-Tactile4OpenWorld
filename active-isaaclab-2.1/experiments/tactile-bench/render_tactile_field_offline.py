from __future__ import annotations

"""Render the V6.2 membrane history as a conservative 81x65 tactile field.

This is deliberately an offline program.  It reconstructs the frozen 7g
vertex contributions from the saved membrane displacement, applies the exact
V6.2 contact gate, reconstructs a continuous piecewise-linear field on the
front-surface triangles, and then writes NumPy fields plus MP4 videos.  It
never starts Isaac Sim or UIPC.
"""

import argparse
import json
import math
import os
import uuid
from pathlib import Path

import numpy as np

import OpenWorldTactile_v5_new_7g_deformation_force_estimator as frozen_7g
import tu_tactile_field


VERSION = "v6.2_offline_tactile_field_v2_triangle"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create conservative 81x65 Fx/Fy/Fz tactile fields and MP4 videos "
            "from a V6.2 simple-grasp output directory."
        )
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--output_dir",
        default="",
        help="Defaults to INPUT_DIR/offline_tactile_field.",
    )
    parser.add_argument("--height", type=int, default=81)
    parser.add_argument("--width", type=int, default=65)
    parser.add_argument(
        "--reconstruction",
        choices=("triangle", "gaussian"),
        default="triangle",
        help="Triangle is the continuous default; gaussian is the legacy comparison mode.",
    )
    parser.add_argument("--sigma_cells", type=float, default=1.25)
    parser.add_argument("--truncate_sigma", type=float, default=4.0)
    parser.add_argument("--video_fps", type=float, default=15.0)
    parser.add_argument(
        "--video_normalization",
        choices=("fixed_global", "diagnostic_per_frame_percentile"),
        default="fixed_global",
        help=(
            "fixed_global preserves cross-frame magnitude comparison; the diagnostic "
            "mode exposes weak connected regions with a per-frame robust range."
        ),
    )
    parser.add_argument("--video_percentile", type=float, default=99.5)
    parser.add_argument("--video_gamma", type=float, default=0.45)
    return parser


def _load_array(directory: Path, filename: str) -> np.ndarray:
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required V6.2 array is missing: {path}")
    return np.load(path, allow_pickle=False)


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required V6.2 metadata is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _estimator_config(metadata: dict[str, object]) -> frozen_7g.EstimatorConfig:
    estimator = metadata.get("estimator")
    if not isinstance(estimator, dict):
        raise ValueError("metadata.json has no estimator configuration")
    return frozen_7g.EstimatorConfig(
        normal_gain_tu_per_m3=float(estimator["normal_gain_tu_per_m3"]),
        tangent_y_gain_tu_per_m3=float(estimator["tangent_y_gain_tu_per_m3"]),
        tangent_z_gain_tu_per_m3=float(estimator["tangent_z_gain_tu_per_m3"]),
        activation_start_m=float(estimator["activation_start_m"]),
        activation_full_m=float(estimator["activation_full_m"]),
    )


def _gate_vertex_history(
    vertex_history: np.ndarray, contact_active: np.ndarray
) -> np.ndarray:
    vertex = np.asarray(vertex_history, dtype=np.float64)
    active = np.asarray(contact_active, dtype=bool).reshape(-1)
    if vertex.ndim != 3 or vertex.shape[2] != 3:
        raise ValueError(f"Vertex history must be [T,N,3], got {vertex.shape}")
    if active.shape != (vertex.shape[0],):
        raise ValueError(
            f"contact_active must be [{vertex.shape[0]}], got {active.shape}"
        )
    gated = vertex.copy()
    gated[~active] = 0.0
    return gated


def _maximum_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    first = np.asarray(actual, dtype=np.float64)
    second = np.asarray(expected, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"Cannot compare shapes {first.shape} and {second.shape}")
    return float(np.max(np.abs(first - second), initial=0.0))


def _load_front_triangle_topology(
    input_dir: Path,
    metadata: dict[str, object],
    front_surface_mask: np.ndarray,
    front_surface_yz_m: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Load self-contained topology, then contract topology, then structured fallback."""

    for filename in ("front_surface_triangles.npy", "surface_triangles.npy"):
        path = input_dir / filename
        if path.is_file():
            triangles = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)
            return (
                tu_tactile_field.front_surface_triangles(
                    triangles, front_surface_mask
                ),
                f"input_directory/{filename}",
            )

    mapping = metadata.get("contract_vertex_mapping")
    contract_dir_value = metadata.get("contract_dir")
    permutation_count = (
        int(mapping.get("permuted_vertex_count", -1))
        if isinstance(mapping, dict)
        else -1
    )
    if isinstance(contract_dir_value, str) and permutation_count == 0:
        contract_path = (
            Path(contract_dir_value).expanduser().resolve()
            / "front_surface_triangles.npy"
        )
        if contract_path.is_file():
            triangles = np.asarray(
                np.load(contract_path, allow_pickle=False), dtype=np.int64
            )
            return (
                tu_tactile_field.front_surface_triangles(
                    triangles, front_surface_mask
                ),
                f"contract/{contract_path}",
            )

    return (
        tu_tactile_field.structured_front_surface_triangles(front_surface_yz_m),
        "structured_yz_fallback",
    )


def build_offline_fields(
    input_dir: Path,
    output_dir: Path,
    *,
    height: int,
    width: int,
    sigma_cells: float,
    truncate_sigma: float,
    video_fps: float,
    reconstruction: str = "triangle",
    video_normalization: str = "fixed_global",
    video_percentile: float = 99.5,
    video_gamma: float = 0.45,
) -> dict[str, object]:
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_json(input_dir / "metadata.json")
    displacement = np.asarray(
        _load_array(input_dir, "surface_displacement_pad_local.npy"),
        dtype=np.float64,
    )
    rest_surface = np.asarray(
        _load_array(input_dir, "rest_surface_vertices_pad_local.npy"),
        dtype=np.float64,
    )
    vertex_area = np.asarray(
        _load_array(input_dir, "vertex_area.npy"), dtype=np.float64
    ).reshape(-1)
    front_mask = np.asarray(
        _load_array(input_dir, "front_surface_mask.npy"), dtype=bool
    ).reshape(-1)
    contact_active = np.asarray(
        _load_array(input_dir, "contact_active.npy"), dtype=bool
    ).reshape(-1)
    saved_force_pad = np.asarray(
        _load_array(input_dir, "force_pad_local.npy"), dtype=np.float64
    )
    saved_tactile = np.asarray(
        _load_array(input_dir, "tactile_force_channels.npy"), dtype=np.float64
    )
    motion_stage = np.asarray(_load_array(input_dir, "motion_stage.npy"), dtype=str)

    if displacement.ndim != 3 or displacement.shape[2] != 3:
        raise ValueError(
            "surface_displacement_pad_local.npy must have shape [T,N,3]"
        )
    frame_count, vertex_count, _ = displacement.shape
    if frame_count == 0:
        raise ValueError("The V6.2 dataset has no completed frames")
    if rest_surface.shape != (vertex_count, 3):
        raise ValueError("Rest surface does not match displacement vertex count")
    if vertex_area.shape != (vertex_count,) or front_mask.shape != (vertex_count,):
        raise ValueError("7f vertex area/front mask does not match displacement")
    for name, value, shape in (
        ("contact_active", contact_active, (frame_count,)),
        ("force_pad_local", saved_force_pad, (frame_count, 3)),
        ("tactile_force_channels", saved_tactile, (frame_count, 3)),
        ("motion_stage", motion_stage, (frame_count,)),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}; expected {shape}")

    config = _estimator_config(metadata)
    result = frozen_7g.estimate_deformation_force(
        displacement, vertex_area, front_mask, config
    )
    gated_vertex_volume = _gate_vertex_history(
        result.vertex_deformation_volume_contribution_m3, contact_active
    )
    pad_vertex, tactile_vertex = tu_tactile_field.tactile_vertex_contributions(
        gated_vertex_volume,
        normal_gain_tu_per_m3=config.normal_gain_tu_per_m3,
        tangent_y_gain_tu_per_m3=config.tangent_y_gain_tu_per_m3,
        tangent_z_gain_tu_per_m3=config.tangent_z_gain_tu_per_m3,
    )
    reconstructed_pad = np.sum(pad_vertex, axis=1, dtype=np.float64)
    reconstructed_tactile = np.sum(tactile_vertex, axis=1, dtype=np.float64)

    front_indices = np.flatnonzero(front_mask)
    front_yz = rest_surface[front_indices, 1:3]
    reconstruction_name = str(reconstruction).strip().lower()
    reconstruction_metrics: dict[str, object]
    extra_arrays: dict[str, np.ndarray] = {}
    if reconstruction_name == "triangle":
        front_triangles, topology_source = _load_front_triangle_topology(
            input_dir, metadata, front_mask, front_yz
        )
        plan = tu_tactile_field.build_triangle_raster_plan(
            front_yz,
            front_triangles,
            height=int(height),
            width=int(width),
        )
        tactile_field, conservation_metrics = (
            tu_tactile_field.reconstruct_triangle_force_field(
                tactile_vertex[:, front_indices],
                vertex_area[front_indices],
                plan,
            )
        )
        reconstruction_metrics = {
            "method": "triangle_barycentric_vertex_density_cell_area",
            "topology_source": topology_source,
            "front_triangle_count": int(front_triangles.shape[0]),
            "covered_grid_cell_count": int(np.count_nonzero(plan.covered_mask)),
            "grid_cell_count": int(height) * int(width),
            "projected_surface_area_m2": float(plan.projected_surface_area_m2),
            "quadrature_area_m2": float(np.sum(plan.cell_area_m2)),
            "signed_conservation": "positive_and_negative_vertex_contributions_scaled_separately",
            **conservation_metrics,
        }
        extra_arrays = {
            "front_surface_triangles_front_local.npy": front_triangles,
            "field_cell_area_m2.npy": plan.cell_area_m2,
            "field_covered_mask.npy": plan.covered_mask,
        }
    elif reconstruction_name == "gaussian":
        plan = tu_tactile_field.build_gaussian_splat_plan(
            front_yz,
            height=int(height),
            width=int(width),
            sigma_cells=float(sigma_cells),
            truncate_sigma=float(truncate_sigma),
        )
        tactile_field = tu_tactile_field.splat_vertex_values(
            tactile_vertex[:, front_indices], plan
        )
        reconstruction_metrics = {
            "method": "legacy_independent_vertex_gaussian_splat",
            "sigma_cells": float(sigma_cells),
            "sigma_m": float(plan.sigma_m),
            "truncate_sigma": float(truncate_sigma),
        }
    else:
        raise ValueError(f"Unsupported reconstruction method: {reconstruction}")
    shear_magnitude = np.linalg.norm(tactile_field[..., :2], axis=3)
    field_total = np.sum(tactile_field, axis=(1, 2), dtype=np.float64)

    cycles = np.zeros(frame_count, dtype=np.int64)
    video = tu_tactile_field.render_tactile_videos(
        output_dir=output_dir,
        tactile_field=tactile_field,
        shear_magnitude=shear_magnitude,
        phases=motion_stage,
        cycles=cycles,
        fps=float(video_fps),
        normalization=str(video_normalization),
        diagnostic_percentile=float(video_percentile),
        diagnostic_gamma=float(video_gamma),
    )

    arrays = {
        # Keep the canonical vector field in float64 so summing its cells remains
        # a meaningful conservation check.  Per-channel display arrays stay compact.
        "tactile_force_field.npy": tactile_field,
        "tactile_fx_field.npy": tactile_field[..., 0].astype(np.float32),
        "tactile_fy_field.npy": tactile_field[..., 1].astype(np.float32),
        "tactile_fz_field.npy": tactile_field[..., 2].astype(np.float32),
        "tactile_shear_magnitude_field.npy": shear_magnitude.astype(np.float32),
        "tactile_vertex_force_channels.npy": tactile_vertex,
        "tactile_vertex_force_density_tu_per_m2.npy": (
            tactile_vertex[:, front_indices]
            / vertex_area[front_indices][None, :, None]
        ),
        "field_grid_y_m.npy": plan.grid_y_m,
        "field_grid_z_m.npy": plan.grid_z_m,
        "field_total_force_channels.npy": field_total,
        **extra_arrays,
    }
    for filename, value in arrays.items():
        _atomic_save(output_dir / filename, value)

    observed = {
        "version": VERSION,
        "source_directory": str(input_dir),
        "frame_count": frame_count,
        "vertex_count": vertex_count,
        "front_surface_vertex_count": int(front_indices.size),
        "active_frame_count": int(np.count_nonzero(contact_active)),
        "grid_shape": [int(height), int(width)],
        "reconstruction": reconstruction_metrics,
        "maximum_pad_reconstruction_error_tu": _maximum_abs_error(
            reconstructed_pad, saved_force_pad
        ),
        "maximum_tactile_reconstruction_error_tu": _maximum_abs_error(
            reconstructed_tactile, saved_tactile
        ),
        "maximum_field_conservation_error_tu": _maximum_abs_error(
            field_total, reconstructed_tactile
        ),
        "maximum_inactive_field_value_tu": float(
            np.max(np.abs(tactile_field[~contact_active]), initial=0.0)
        ),
        "video": video,
    }
    _atomic_json(output_dir / "metadata.json", observed)
    return observed


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if int(args.height) < 2 or int(args.width) < 2:
        parser.error("--height and --width must both be at least 2")
    for name in ("sigma_cells", "truncate_sigma", "video_fps", "video_gamma"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name} must be finite and > 0")
    if not math.isfinite(float(args.video_percentile)) or not (
        0.0 < float(args.video_percentile) <= 100.0
    ):
        parser.error("--video_percentile must be finite and in (0, 100]")
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else input_dir / "offline_tactile_field"
    )
    observed = build_offline_fields(
        input_dir,
        output_dir,
        height=int(args.height),
        width=int(args.width),
        reconstruction=str(args.reconstruction),
        sigma_cells=float(args.sigma_cells),
        truncate_sigma=float(args.truncate_sigma),
        video_fps=float(args.video_fps),
        video_normalization=str(args.video_normalization),
        video_percentile=float(args.video_percentile),
        video_gamma=float(args.video_gamma),
    )
    print(
        "[V62_OFFLINE_FIELD] "
        f"frames={observed['frame_count']} active={observed['active_frame_count']} "
        f"method={observed['reconstruction']['method']} "
        f"field_error={observed['maximum_field_conservation_error_tu']:.3e} "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
