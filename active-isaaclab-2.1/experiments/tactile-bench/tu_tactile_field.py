from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


FIELD_HEIGHT = 81
FIELD_WIDTH = 65
EPS = 1.0e-12
V9_VERSION = "v5_new_9_tu_tactile_field_rendering_v1"
CORE_VIDEO_NAMES = (
    "tactile_fx_signed_sequence.mp4",
    "tactile_fy_signed_sequence.mp4",
    "tactile_fz_sequence.mp4",
    "tactile_shear_magnitude_sequence.mp4",
    "tactile_fxyz_composite_sequence.mp4",
)
FROZEN_SOURCE_NAMES = (
    "OpenWorldTactile_v5_new_7f_deformation_contract_probe.py",
    "OpenWorldTactile_v5_new_7g_deformation_force_estimator.py",
    "OpenWorldTactile_v5_new_8_grasp_integration.py",
)


@dataclass(frozen=True)
class GaussianSplatPlan:
    height: int
    width: int
    grid_y_m: np.ndarray
    grid_z_m: np.ndarray
    sigma_m: float
    truncate_sigma: float
    cell_indices: tuple[np.ndarray, ...]
    cell_weights: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class TriangleRasterPlan:
    """Barycentric sampling plan for a triangulated membrane surface."""

    height: int
    width: int
    grid_y_m: np.ndarray
    grid_z_m: np.ndarray
    surface_triangles: np.ndarray
    cell_indices: np.ndarray
    cell_triangle_vertices: np.ndarray
    cell_barycentric_weights: np.ndarray
    cell_area_m2: np.ndarray
    covered_mask: np.ndarray
    projected_surface_area_m2: float


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def _load_array(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required {name} is missing: {path}")
    return np.load(path, allow_pickle=False)


def _load_json(path: Path, name: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required {name} is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_source_paths(bench_dir: Path | None = None) -> tuple[Path, ...]:
    root = Path(bench_dir) if bench_dir is not None else Path(__file__).resolve().parent
    return tuple(root / name for name in FROZEN_SOURCE_NAMES)


def capture_frozen_hashes(paths: Sequence[Path] | None = None) -> dict[str, str]:
    selected = tuple(paths) if paths is not None else frozen_source_paths()
    missing = [str(path) for path in selected if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen source files are missing: {missing}")
    return {Path(path).name: _sha256(Path(path)) for path in selected}


def tactile_vertex_contributions(
    vertex_deformation_volume_m3: np.ndarray,
    *,
    normal_gain_tu_per_m3: float,
    tangent_y_gain_tu_per_m3: float,
    tangent_z_gain_tu_per_m3: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen 7g gains before the vertex reduction.

    Input channels are [A*d_n, A*w*u_y, A*w*u_z]. Returned channels are
    pad-local [-f_n, f_y, f_z] and tactile [f_y, -f_z, f_n].
    """

    volume = np.asarray(vertex_deformation_volume_m3, dtype=np.float64)
    if volume.ndim != 3 or volume.shape[2] != 3:
        raise ValueError(
            f"vertex deformation volume must be [T,N,3], got {volume.shape}."
        )
    if not np.all(np.isfinite(volume)):
        raise ValueError("vertex deformation volume contains NaN or Inf.")
    fn = float(normal_gain_tu_per_m3) * volume[..., 0]
    fy = float(tangent_y_gain_tu_per_m3) * volume[..., 1]
    fz = float(tangent_z_gain_tu_per_m3) * volume[..., 2]
    pad_local = np.stack((-fn, fy, fz), axis=2)
    tactile = np.stack((fy, -fz, fn), axis=2)
    return pad_local, tactile


def build_gaussian_splat_plan(
    front_surface_yz_m: np.ndarray,
    *,
    height: int = FIELD_HEIGHT,
    width: int = FIELD_WIDTH,
    sigma_cells: float = 1.25,
    truncate_sigma: float = 4.0,
) -> GaussianSplatPlan:
    yz = np.asarray(front_surface_yz_m, dtype=np.float64)
    if yz.ndim != 2 or yz.shape[1] != 2 or yz.shape[0] == 0:
        raise ValueError(f"front_surface_yz must be nonempty [N,2], got {yz.shape}.")
    if not np.all(np.isfinite(yz)):
        raise ValueError("front_surface_yz contains NaN or Inf.")
    if int(height) < 2 or int(width) < 2:
        raise ValueError("Gaussian splat grid dimensions must both be at least two.")
    if not math.isfinite(float(sigma_cells)) or float(sigma_cells) <= 0.0:
        raise ValueError("sigma_cells must be finite and > 0.")
    if not math.isfinite(float(truncate_sigma)) or float(truncate_sigma) <= 0.0:
        raise ValueError("truncate_sigma must be finite and > 0.")

    y_min, z_min = np.min(yz, axis=0)
    y_max, z_max = np.max(yz, axis=0)
    if y_max - y_min <= EPS or z_max - z_min <= EPS:
        raise ValueError("front-surface Y and Z spans must both be nonzero.")
    grid_y = np.linspace(y_min, y_max, int(width), dtype=np.float64)
    # Image row zero is +Z so the rendered tactile field is not vertically mirrored.
    grid_z = np.linspace(z_max, z_min, int(height), dtype=np.float64)
    dy = abs(float(grid_y[1] - grid_y[0]))
    dz = abs(float(grid_z[1] - grid_z[0]))
    sigma_m = float(sigma_cells) * math.sqrt(dy * dz)
    radius_m = float(truncate_sigma) * sigma_m

    indices: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for vertex_y, vertex_z in yz:
        cols = np.flatnonzero(np.abs(grid_y - vertex_y) <= radius_m)
        rows = np.flatnonzero(np.abs(grid_z - vertex_z) <= radius_m)
        if rows.size == 0:
            rows = np.asarray(
                [int(np.argmin(np.abs(grid_z - vertex_z)))], dtype=np.int64
            )
        if cols.size == 0:
            cols = np.asarray(
                [int(np.argmin(np.abs(grid_y - vertex_y)))], dtype=np.int64
            )
        row_mesh, col_mesh = np.meshgrid(rows, cols, indexing="ij")
        row_flat = row_mesh.reshape(-1)
        col_flat = col_mesh.reshape(-1)
        distance_squared = (grid_y[col_flat] - vertex_y) ** 2 + (
            grid_z[row_flat] - vertex_z
        ) ** 2
        gaussian = np.exp(-distance_squared / (2.0 * sigma_m * sigma_m))
        gaussian_sum = float(np.sum(gaussian, dtype=np.float64))
        if not math.isfinite(gaussian_sum) or gaussian_sum <= 0.0:
            raise ValueError("Gaussian splat normalization produced an empty kernel.")
        indices.append((row_flat * int(width) + col_flat).astype(np.int64))
        weights.append((gaussian / gaussian_sum).astype(np.float64))

    return GaussianSplatPlan(
        height=int(height),
        width=int(width),
        grid_y_m=grid_y,
        grid_z_m=grid_z,
        sigma_m=sigma_m,
        truncate_sigma=float(truncate_sigma),
        cell_indices=tuple(indices),
        cell_weights=tuple(weights),
    )


def splat_vertex_values(values: np.ndarray, plan: GaussianSplatPlan) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    squeeze_channel = source.ndim == 2
    if squeeze_channel:
        source = source[..., None]
    if source.ndim != 3:
        raise ValueError(f"Splat input must be [T,N] or [T,N,C], got {source.shape}.")
    if source.shape[1] != len(plan.cell_indices):
        raise ValueError(
            f"Splat vertex count {source.shape[1]} does not match plan {len(plan.cell_indices)}."
        )
    if not np.all(np.isfinite(source)):
        raise ValueError("Splat input contains NaN or Inf.")
    field = np.zeros(
        (source.shape[0], plan.height * plan.width, source.shape[2]), dtype=np.float64
    )
    for vertex_index, (cell_index, cell_weight) in enumerate(
        zip(plan.cell_indices, plan.cell_weights, strict=True)
    ):
        field[:, cell_index, :] += (
            source[:, vertex_index, None, :] * cell_weight[None, :, None]
        )
    reshaped = field.reshape(source.shape[0], plan.height, plan.width, source.shape[2])
    return reshaped[..., 0] if squeeze_channel else reshaped


def front_surface_triangles(
    surface_triangles: np.ndarray, front_surface_mask: np.ndarray
) -> np.ndarray:
    """Filter global surface triangles and remap them to front-vertex indices."""

    triangles = np.asarray(surface_triangles, dtype=np.int64)
    mask = np.asarray(front_surface_mask, dtype=bool).reshape(-1)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"surface_triangles must have shape [M,3], got {triangles.shape}.")
    if triangles.size == 0:
        raise ValueError("surface_triangles must not be empty.")
    if int(np.min(triangles)) < 0 or int(np.max(triangles)) >= mask.size:
        raise ValueError("surface_triangles contains an out-of-range vertex index.")
    selected = triangles[np.all(mask[triangles], axis=1)]
    if selected.size == 0:
        raise ValueError("No triangle has all three vertices on the front surface.")
    global_to_front = np.full(mask.size, -1, dtype=np.int64)
    global_to_front[np.flatnonzero(mask)] = np.arange(np.count_nonzero(mask))
    return global_to_front[selected]


def structured_front_surface_triangles(
    front_surface_yz_m: np.ndarray,
) -> np.ndarray:
    """Reconstruct rectangular structured topology as a last-resort fallback."""

    yz = np.asarray(front_surface_yz_m, dtype=np.float64)
    if yz.ndim != 2 or yz.shape[1] != 2 or yz.shape[0] < 4:
        raise ValueError(f"front_surface_yz must have shape [N,2], got {yz.shape}.")
    if not np.all(np.isfinite(yz)):
        raise ValueError("front_surface_yz contains NaN or Inf.")
    span = np.ptp(yz, axis=0)
    decimals = max(6, int(-math.floor(math.log10(max(float(np.max(span)), EPS)))) + 7)
    rounded_y = np.round(yz[:, 0], decimals=decimals)
    rounded_z = np.round(yz[:, 1], decimals=decimals)
    unique_y = np.unique(rounded_y)
    unique_z = np.unique(rounded_z)
    if unique_y.size * unique_z.size != yz.shape[0]:
        raise ValueError(
            "Front vertices are not a complete rectangular YZ grid; provide saved surface triangles."
        )
    vertex_grid = np.full((unique_y.size, unique_z.size), -1, dtype=np.int64)
    for vertex_index, (value_y, value_z) in enumerate(zip(rounded_y, rounded_z, strict=True)):
        y_index = int(np.searchsorted(unique_y, value_y))
        z_index = int(np.searchsorted(unique_z, value_z))
        if vertex_grid[y_index, z_index] >= 0:
            raise ValueError("Structured front grid contains a duplicate YZ coordinate.")
        vertex_grid[y_index, z_index] = vertex_index
    if np.any(vertex_grid < 0):
        raise ValueError("Structured front grid has a missing YZ coordinate.")
    triangles: list[tuple[int, int, int]] = []
    for y_index in range(unique_y.size - 1):
        for z_index in range(unique_z.size - 1):
            lower_left = int(vertex_grid[y_index, z_index])
            lower_right = int(vertex_grid[y_index + 1, z_index])
            upper_right = int(vertex_grid[y_index + 1, z_index + 1])
            upper_left = int(vertex_grid[y_index, z_index + 1])
            triangles.append((lower_left, lower_right, upper_right))
            triangles.append((lower_left, upper_right, upper_left))
    return np.asarray(triangles, dtype=np.int64)


def build_triangle_raster_plan(
    front_surface_yz_m: np.ndarray,
    surface_triangles: np.ndarray,
    *,
    height: int = FIELD_HEIGHT,
    width: int = FIELD_WIDTH,
) -> TriangleRasterPlan:
    """Build a piecewise-linear barycentric sampling plan on an image grid."""

    yz = np.asarray(front_surface_yz_m, dtype=np.float64)
    triangles = np.asarray(surface_triangles, dtype=np.int64)
    if yz.ndim != 2 or yz.shape[1] != 2 or yz.shape[0] < 3:
        raise ValueError(f"front_surface_yz must have shape [N,2], got {yz.shape}.")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
        raise ValueError(f"surface_triangles must be nonempty [M,3], got {triangles.shape}.")
    if not np.all(np.isfinite(yz)):
        raise ValueError("front_surface_yz contains NaN or Inf.")
    if int(np.min(triangles)) < 0 or int(np.max(triangles)) >= yz.shape[0]:
        raise ValueError("surface_triangles contains an out-of-range front vertex index.")
    if int(height) < 2 or int(width) < 2:
        raise ValueError("Triangle raster grid dimensions must both be at least two.")

    y_min, z_min = np.min(yz, axis=0)
    y_max, z_max = np.max(yz, axis=0)
    if y_max - y_min <= EPS or z_max - z_min <= EPS:
        raise ValueError("Front-surface Y and Z spans must both be nonzero.")
    grid_y = np.linspace(y_min, y_max, int(width), dtype=np.float64)
    grid_z = np.linspace(z_max, z_min, int(height), dtype=np.float64)
    dy = abs(float(grid_y[1] - grid_y[0]))
    dz = abs(float(grid_z[1] - grid_z[0]))
    y_quadrature = np.full(int(width), dy, dtype=np.float64)
    z_quadrature = np.full(int(height), dz, dtype=np.float64)
    y_quadrature[[0, -1]] *= 0.5
    z_quadrature[[0, -1]] *= 0.5
    full_cell_area = np.outer(z_quadrature, y_quadrature)

    sample_triangle = np.full(int(height) * int(width), -1, dtype=np.int64)
    sample_barycentric = np.zeros((sample_triangle.size, 3), dtype=np.float64)
    projected_area = 0.0
    inside_tolerance = 1.0e-9
    for triangle_index, triangle in enumerate(triangles):
        triangle_yz = yz[triangle]
        y0, z0 = triangle_yz[0]
        y1, z1 = triangle_yz[1]
        y2, z2 = triangle_yz[2]
        denominator = (z1 - z2) * (y0 - y2) + (y2 - y1) * (z0 - z2)
        if abs(float(denominator)) <= EPS:
            raise ValueError(f"Projected surface triangle {triangle_index} is degenerate.")
        projected_area += 0.5 * abs(float(denominator))
        columns = np.flatnonzero(
            (grid_y >= np.min(triangle_yz[:, 0]) - inside_tolerance)
            & (grid_y <= np.max(triangle_yz[:, 0]) + inside_tolerance)
        )
        rows = np.flatnonzero(
            (grid_z >= np.min(triangle_yz[:, 1]) - inside_tolerance)
            & (grid_z <= np.max(triangle_yz[:, 1]) + inside_tolerance)
        )
        if rows.size == 0 or columns.size == 0:
            continue
        row_mesh, column_mesh = np.meshgrid(rows, columns, indexing="ij")
        flat_indices = (row_mesh.reshape(-1) * int(width) + column_mesh.reshape(-1))
        sample_y = grid_y[column_mesh.reshape(-1)]
        sample_z = grid_z[row_mesh.reshape(-1)]
        weight0 = ((z1 - z2) * (sample_y - y2) + (y2 - y1) * (sample_z - z2)) / denominator
        weight1 = ((z2 - z0) * (sample_y - y2) + (y0 - y2) * (sample_z - z2)) / denominator
        weight2 = 1.0 - weight0 - weight1
        weights = np.column_stack((weight0, weight1, weight2))
        inside = np.all(weights >= -inside_tolerance, axis=1) & np.all(
            weights <= 1.0 + inside_tolerance, axis=1
        )
        available = sample_triangle[flat_indices] < 0
        selected = inside & available
        if np.any(selected):
            selected_indices = flat_indices[selected]
            sample_triangle[selected_indices] = triangle_index
            clipped = np.clip(weights[selected], 0.0, 1.0)
            sample_barycentric[selected_indices] = clipped / np.sum(
                clipped, axis=1, keepdims=True
            )

    covered_flat = sample_triangle >= 0
    if not np.any(covered_flat):
        raise ValueError("No output grid sample lies inside the projected membrane triangles.")
    covered_mask = covered_flat.reshape(int(height), int(width))
    cell_area = full_cell_area * covered_mask
    cell_indices = np.flatnonzero(covered_flat)
    chosen_triangles = triangles[sample_triangle[cell_indices]]
    return TriangleRasterPlan(
        height=int(height),
        width=int(width),
        grid_y_m=grid_y,
        grid_z_m=grid_z,
        surface_triangles=triangles.copy(),
        cell_indices=cell_indices,
        cell_triangle_vertices=chosen_triangles,
        cell_barycentric_weights=sample_barycentric[cell_indices],
        cell_area_m2=cell_area,
        covered_mask=covered_mask,
        projected_surface_area_m2=float(projected_area),
    )


def reconstruct_triangle_force_field(
    vertex_force_channels: np.ndarray,
    vertex_area_m2: np.ndarray,
    plan: TriangleRasterPlan,
) -> tuple[np.ndarray, dict[str, float]]:
    """Interpolate vertex force density and integrate it into conservative cells.

    Positive and negative nodal contributions are reconstructed independently.
    This retains signed shear lobes and remains well-conditioned when their net
    Fx or Fy is close to zero.
    """

    force = np.asarray(vertex_force_channels, dtype=np.float64)
    squeeze_channel = force.ndim == 2
    if squeeze_channel:
        force = force[..., None]
    area = np.asarray(vertex_area_m2, dtype=np.float64).reshape(-1)
    if force.ndim != 3:
        raise ValueError(f"vertex_force_channels must be [T,N] or [T,N,C], got {force.shape}.")
    if force.shape[1] != area.size:
        raise ValueError("vertex force and vertex area counts do not match.")
    if plan.surface_triangles.size and int(np.max(plan.surface_triangles)) >= area.size:
        raise ValueError("Triangle plan references a vertex outside vertex_area_m2.")
    if not np.all(np.isfinite(force)) or not np.all(np.isfinite(area)):
        raise ValueError("Triangle reconstruction input contains NaN or Inf.")
    if np.any(area <= 0.0):
        raise ValueError("Every reconstructed front vertex must have positive area.")

    def integrate(nonnegative_force: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        density = nonnegative_force / area[None, :, None]
        triangle_density = density[:, plan.cell_triangle_vertices, :]
        sampled_density = np.einsum(
            "tpkc,pk->tpc",
            triangle_density,
            plan.cell_barycentric_weights,
            optimize=True,
        )
        raw_cells = sampled_density * plan.cell_area_m2.reshape(-1)[
            plan.cell_indices
        ][None, :, None]
        raw_total = np.sum(raw_cells, axis=1, dtype=np.float64)
        target_total = np.sum(nonnegative_force, axis=1, dtype=np.float64)
        scale = np.ones_like(target_total)
        active = target_total > EPS
        if np.any(active & (raw_total <= EPS)):
            raise RuntimeError("Triangle rasterization lost a nonzero force channel.")
        scale[active] = target_total[active] / raw_total[active]
        raw_cells *= scale[:, None, :]
        return raw_cells, raw_total, scale

    positive_cells, positive_raw_total, positive_scale = integrate(np.clip(force, 0.0, None))
    negative_cells, negative_raw_total, negative_scale = integrate(np.clip(-force, 0.0, None))
    covered_cells = positive_cells - negative_cells
    field = np.zeros(
        (force.shape[0], plan.height * plan.width, force.shape[2]), dtype=np.float64
    )
    field[:, plan.cell_indices, :] = covered_cells
    field = field.reshape(force.shape[0], plan.height, plan.width, force.shape[2])
    target = np.sum(force, axis=1, dtype=np.float64)
    raw_signed = positive_raw_total - negative_raw_total
    metrics = {
        "maximum_preconservation_error_tu": float(
            np.max(np.abs(raw_signed - target), initial=0.0)
        ),
        "minimum_positive_scale": float(np.min(positive_scale, initial=1.0)),
        "maximum_positive_scale": float(np.max(positive_scale, initial=1.0)),
        "minimum_negative_scale": float(np.min(negative_scale, initial=1.0)),
        "maximum_negative_scale": float(np.max(negative_scale, initial=1.0)),
    }
    result = field[..., 0] if squeeze_channel else field
    return result, metrics


def _coefficient_of_variation(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return float("inf")
    return float(np.std(array) / max(abs(float(np.mean(array))), EPS))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=np.float64)
    begin = 0
    while begin < array.size:
        end = begin + 1
        while end < array.size and array[order[end]] == array[order[begin]]:
            end += 1
        ranks[order[begin:end]] = 0.5 * float(begin + end - 1)
        begin = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    if first_rank.size < 2 or np.std(first_rank) <= EPS or np.std(second_rank) <= EPS:
        return 0.0
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _field_centroid(field: np.ndarray) -> list[float] | None:
    weights = np.clip(np.asarray(field, dtype=np.float64), 0.0, None)
    total = float(np.sum(weights, dtype=np.float64))
    if total <= EPS:
        return None
    rows, cols = np.indices(weights.shape, dtype=np.float64)
    return [
        float(np.sum(rows * weights, dtype=np.float64) / total),
        float(np.sum(cols * weights, dtype=np.float64) / total),
    ]


def _maximum_centroid_distance(centroids: Sequence[list[float] | None]) -> float:
    finite = [
        np.asarray(value, dtype=np.float64) for value in centroids if value is not None
    ]
    if len(finite) < 2:
        return 0.0
    return float(
        max(
            np.linalg.norm(finite[first] - finite[second])
            for first in range(len(finite))
            for second in range(first + 1, len(finite))
        )
    )


def _field_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= EPS:
        return 1.0 if np.allclose(first, second, rtol=0.0, atol=1.0e-12) else 0.0
    return float(np.dot(a, b) / denominator)


def _maximum_abs_relative_error(
    actual: np.ndarray, expected: np.ndarray, *, near_zero_tu: float
) -> dict[str, object]:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    error = np.abs(actual_array - expected_array)
    nonzero = np.abs(expected_array) >= float(near_zero_tu)
    near_zero = ~nonzero
    relative = np.zeros_like(error)
    relative[nonzero] = error[nonzero] / np.abs(expected_array[nonzero])
    return {
        "maximum_absolute_error_tu": float(np.max(error, initial=0.0)),
        "maximum_nonzero_relative_error": float(np.max(relative, initial=0.0)),
        "maximum_near_zero_absolute_error_tu": float(
            np.max(error[near_zero], initial=0.0)
        ),
        "nonzero_value_count": int(np.count_nonzero(nonzero)),
        "near_zero_value_count": int(np.count_nonzero(near_zero)),
    }


def calculate_spatial_metrics(
    *,
    tactile_vertices: np.ndarray,
    tactile_total: np.ndarray,
    tactile_field: np.ndarray,
    activation_field: np.ndarray,
    phases: np.ndarray,
    cycles: np.ndarray,
    measured_opening_mm: np.ndarray,
    expected_frame_count: int,
    expected_cycle_count: int,
    no_contact_tolerance_tu: float,
    active_epsilon: float,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    vertex_total = np.sum(tactile_vertices, axis=1, dtype=np.float64)
    field_total = np.sum(tactile_field, axis=(1, 2), dtype=np.float64)
    vertex_error = _maximum_abs_relative_error(
        vertex_total, tactile_total, near_zero_tu=1.0e-6
    )
    field_error = _maximum_abs_relative_error(
        field_total, vertex_total, near_zero_tu=1.0e-6
    )
    active_area = np.count_nonzero(
        activation_field > float(active_epsilon), axis=(1, 2)
    )
    fz_field = tactile_field[..., 2]
    centroids = [_field_centroid(frame) for frame in fz_field]

    no_contact_indices = np.flatnonzero(phases == "pre_contact")
    no_contact_active_max = int(np.max(active_area[no_contact_indices], initial=0))
    no_contact_max_total = float(
        np.max(np.abs(field_total[no_contact_indices]), initial=0.0)
    )
    close_indices = np.flatnonzero(phases == "close")
    closure = float(np.max(measured_opening_mm)) - measured_opening_mm[close_indices]
    close_fz_spearman = _spearman(closure, field_total[close_indices, 2])
    close_area_spearman = _spearman(closure, active_area[close_indices])
    close_third = max(1, close_indices.size // 3)
    close_area_start = float(np.mean(active_area[close_indices[:close_third]]))
    close_area_end = float(np.mean(active_area[close_indices[-close_third:]]))

    hold_metrics: list[dict[str, object]] = []
    hold_mean_fields: list[np.ndarray] = []
    release_ratios: list[np.ndarray] = []
    for cycle in range(int(expected_cycle_count)):
        cycle_indices = np.flatnonzero(cycles == cycle)
        hold_indices = np.flatnonzero((cycles == cycle) & (phases == "hold"))
        recovery_indices = np.flatnonzero((cycles == cycle) & (phases == "recovery"))
        if (
            cycle_indices.size == 0
            or hold_indices.size == 0
            or recovery_indices.size == 0
        ):
            raise ValueError(f"Cycle {cycle} lacks a hold or recovery segment.")
        hold_totals = field_total[hold_indices, 2]
        hold_peaks = np.max(fz_field[hold_indices], axis=(1, 2))
        hold_areas = active_area[hold_indices]
        hold_centroids = [centroids[index] for index in hold_indices]
        hold_mean_field = np.mean(fz_field[hold_indices], axis=0)
        hold_mean_fields.append(hold_mean_field)
        cycle_peak = np.max(np.abs(field_total[cycle_indices]), axis=0)
        tail_count = min(3, recovery_indices.size)
        release_tail = np.max(
            np.abs(field_total[recovery_indices[-tail_count:]]), axis=0
        )
        release_ratio = np.asarray(
            [
                (
                    float(release_tail[axis] / cycle_peak[axis])
                    if cycle_peak[axis] > float(no_contact_tolerance_tu)
                    else (
                        0.0
                        if release_tail[axis] < float(no_contact_tolerance_tu)
                        else float("inf")
                    )
                )
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        release_ratios.append(release_ratio)
        hold_metrics.append(
            {
                "cycle": cycle,
                "field_total_fz_cv": _coefficient_of_variation(hold_totals),
                "field_peak_fz_cv": _coefficient_of_variation(hold_peaks),
                "field_centroid_maximum_drift_cells": _maximum_centroid_distance(
                    hold_centroids
                ),
                "field_active_area_cv": _coefficient_of_variation(hold_areas),
                "hold_mean_fz_total_tu": float(np.sum(hold_mean_field)),
                "hold_mean_fz_peak_tu_per_cell": float(np.max(hold_mean_field)),
                "hold_mean_fz_centroid_row_col": _field_centroid(hold_mean_field),
                "release_axis_to_peak_ratio": release_ratio.tolist(),
            }
        )

    hold_mean_fields_array = np.asarray(hold_mean_fields, dtype=np.float64)
    repeat_totals = np.sum(hold_mean_fields_array, axis=(1, 2), dtype=np.float64)
    repeat_peaks = np.max(hold_mean_fields_array, axis=(1, 2))
    repeat_centroids = [_field_centroid(field) for field in hold_mean_fields_array]
    pair_correlations = [
        {
            "cycle_a": first,
            "cycle_b": second,
            "correlation": _field_correlation(
                hold_mean_fields_array[first], hold_mean_fields_array[second]
            ),
        }
        for first in range(len(hold_mean_fields_array))
        for second in range(first + 1, len(hold_mean_fields_array))
    ]
    minimum_repeat_correlation = min(
        (float(item["correlation"]) for item in pair_correlations), default=0.0
    )
    repeatability = {
        "comparison_phase": "per-cycle hold-mean Fz field",
        "peak_cv": _coefficient_of_variation(repeat_peaks),
        "total_force_cv": _coefficient_of_variation(repeat_totals),
        "centroid_maximum_offset_cells": _maximum_centroid_distance(repeat_centroids),
        "minimum_pairwise_field_correlation": minimum_repeat_correlation,
        "pairwise_field_correlations": pair_correlations,
        "per_cycle_hold_mean_peak_tu_per_cell": repeat_peaks.tolist(),
        "per_cycle_hold_mean_total_tu": repeat_totals.tolist(),
        "per_cycle_hold_mean_centroid_row_col": repeat_centroids,
    }
    maximum_hold_total_cv = max(
        float(item["field_total_fz_cv"]) for item in hold_metrics
    )
    maximum_hold_peak_cv = max(float(item["field_peak_fz_cv"]) for item in hold_metrics)
    maximum_hold_centroid_drift = max(
        float(item["field_centroid_maximum_drift_cells"]) for item in hold_metrics
    )
    maximum_hold_area_cv = max(
        float(item["field_active_area_cv"]) for item in hold_metrics
    )
    maximum_release_ratio = float(np.max(np.asarray(release_ratios)))
    final_active_cells = int(active_area[-1])

    field_metrics = {
        "frame_count": int(tactile_field.shape[0]),
        "grid_shape": [int(tactile_field.shape[1]), int(tactile_field.shape[2])],
        "cycle_count": int(np.unique(cycles).size),
        "no_contact_frame_count": int(no_contact_indices.size),
        "no_contact_maximum_active_field_cells": no_contact_active_max,
        "no_contact_maximum_abs_axis_total_tu": no_contact_max_total,
        "close_fz_spearman": close_fz_spearman,
        "close_active_area_spearman": close_area_spearman,
        "close_active_area_start_mean_cells": close_area_start,
        "close_active_area_end_mean_cells": close_area_end,
        "maximum_hold_field_total_fz_cv": maximum_hold_total_cv,
        "maximum_hold_field_peak_fz_cv": maximum_hold_peak_cv,
        "maximum_hold_centroid_drift_cells": maximum_hold_centroid_drift,
        "maximum_hold_active_area_cv": maximum_hold_area_cv,
        "maximum_release_axis_to_peak_ratio": maximum_release_ratio,
        "final_active_field_cells": final_active_cells,
        "per_cycle": hold_metrics,
    }
    conservation = {
        "vertex_sum_vs_tactile_force_channels": vertex_error,
        "field_sum_vs_vertex_sum": field_error,
        "vertex_totals_tu": vertex_total.tolist(),
        "field_totals_tu": field_total.tolist(),
        "reference_tactile_force_channels_tu": np.asarray(
            tactile_total, dtype=np.float64
        ).tolist(),
    }
    spatial_checks = {
        "expected_frame_count": int(tactile_field.shape[0])
        == int(expected_frame_count),
        "expected_cycle_count": int(np.unique(cycles).size)
        == int(expected_cycle_count),
        "vertex_force_sum_reconstructs_frozen_7g": bool(
            np.allclose(vertex_total, tactile_total, rtol=1.0e-6, atol=1.0e-8)
        ),
        "field_force_is_conserved": bool(
            float(field_error["maximum_nonzero_relative_error"]) < 0.01
            and float(field_error["maximum_near_zero_absolute_error_tu"]) < 1.0e-6
        ),
        "no_contact_has_zero_active_cells": no_contact_indices.size > 0
        and no_contact_active_max == 0,
        "no_contact_field_is_near_zero": no_contact_indices.size > 0
        and no_contact_max_total < float(no_contact_tolerance_tu),
        "close_fz_spearman_above_0_95": close_fz_spearman > 0.95,
        "close_field_area_increases_overall": close_area_end > close_area_start
        and close_area_spearman > 0.5,
        "hold_total_cv_below_2_percent": maximum_hold_total_cv < 0.02,
        "hold_centroid_drift_below_one_cell": maximum_hold_centroid_drift < 1.0,
        "repeat_peak_cv_below_2_percent": float(repeatability["peak_cv"]) < 0.02,
        "repeat_total_cv_below_2_percent": float(repeatability["total_force_cv"])
        < 0.02,
        "repeat_centroid_offset_below_two_cells": float(
            repeatability["centroid_maximum_offset_cells"]
        )
        < 2.0,
        "repeat_field_correlation_above_0_95": minimum_repeat_correlation > 0.95,
        "release_all_axes_below_2_percent_peak": maximum_release_ratio < 0.02,
        "final_active_field_cells_zero": final_active_cells == 0,
    }
    field_metrics["spatial_checks"] = spatial_checks
    return field_metrics, conservation, repeatability


def _signed_bgr(
    field: np.ndarray, maximum: float, *, gamma: float = 1.0
) -> np.ndarray:
    normalized = np.clip(
        np.asarray(field, dtype=np.float64) / max(float(maximum), EPS), -1.0, 1.0
    )
    magnitude = np.power(np.abs(normalized), float(gamma))[..., None]
    neutral = np.full((*normalized.shape, 3), 36.0, dtype=np.float64)
    target = np.zeros_like(neutral)
    target[normalized < 0.0] = (255.0, 48.0, 24.0)
    target[normalized >= 0.0] = (24.0, 48.0, 255.0)
    return np.clip(neutral * (1.0 - magnitude) + target * magnitude, 0.0, 255.0).astype(
        np.uint8
    )


def _positive_bgr(
    field: np.ndarray, maximum: float, *, gamma: float = 1.0
) -> np.ndarray:
    normalized = np.clip(
        np.asarray(field, dtype=np.float64) / max(float(maximum), EPS), 0.0, 1.0
    )
    normalized = np.power(normalized, float(gamma))
    return cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO
    )


def _nonzero_percentile_range(field: np.ndarray, percentile: float) -> float:
    magnitude = np.abs(np.asarray(field, dtype=np.float64)).reshape(-1)
    nonzero = magnitude[magnitude > EPS]
    if nonzero.size == 0:
        return EPS
    return max(float(np.percentile(nonzero, float(percentile))), EPS)


def _annotated_panel(
    color_field: np.ndarray,
    *,
    title: str,
    frame_index: int,
    phase: str,
    cycle: int,
    totals: np.ndarray,
    scale: int,
    header_height: int,
) -> np.ndarray:
    image = cv2.resize(
        color_field,
        (color_field.shape[1] * int(scale), color_field.shape[0] * int(scale)),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.zeros(
        (image.shape[0] + int(header_height), image.shape[1], 3), dtype=np.uint8
    )
    canvas[int(header_height) :] = image
    cv2.putText(
        canvas,
        f"{title} | frame {frame_index + 1} | phase {phase} | cycle {cycle + 1}",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Fx {totals[0]:+.6f}  Fy {totals[1]:+.6f}  Fz {totals[2]:+.6f}  unit = TU",
        (10, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _open_video(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {path}.")
    return writer


def _decode_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"opened": False, "decoded_frame_count": 0, "reported_frame_count": 0}
    reported = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    return {
        "opened": True,
        "decoded_frame_count": decoded,
        "reported_frame_count": reported,
    }


def render_tactile_videos(
    *,
    output_dir: Path,
    tactile_field: np.ndarray,
    shear_magnitude: np.ndarray,
    phases: np.ndarray,
    cycles: np.ndarray,
    fps: float,
    normalization: str = "fixed_global",
    diagnostic_percentile: float = 99.5,
    diagnostic_gamma: float = 0.45,
) -> dict[str, object]:
    if float(fps) <= 0.0 or not math.isfinite(float(fps)):
        raise ValueError("Video FPS must be finite and > 0.")
    normalization_mode = str(normalization).strip().lower()
    if normalization_mode not in ("fixed_global", "diagnostic_per_frame_percentile"):
        raise ValueError(
            "Video normalization must be fixed_global or "
            "diagnostic_per_frame_percentile."
        )
    if not (0.0 < float(diagnostic_percentile) <= 100.0) or not math.isfinite(
        float(diagnostic_percentile)
    ):
        raise ValueError("Diagnostic percentile must be finite and in (0, 100].")
    if float(diagnostic_gamma) <= 0.0 or not math.isfinite(float(diagnostic_gamma)):
        raise ValueError("Diagnostic gamma must be finite and > 0.")
    frame_count = int(tactile_field.shape[0])
    totals = np.sum(tactile_field, axis=(1, 2), dtype=np.float64)
    ranges = {
        "Fx": float(np.max(np.abs(tactile_field[..., 0]), initial=0.0)),
        "Fy": float(np.max(np.abs(tactile_field[..., 1]), initial=0.0)),
        "Fz": float(np.max(tactile_field[..., 2], initial=0.0)),
        "shear_magnitude": float(np.max(shear_magnitude, initial=0.0)),
    }
    render_ranges = {name: max(value, EPS) for name, value in ranges.items()}
    diagnostic_mode = normalization_mode == "diagnostic_per_frame_percentile"
    gamma = float(diagnostic_gamma) if diagnostic_mode else 1.0
    per_frame_ranges = np.zeros((frame_count, 4), dtype=np.float64)
    scale = 8
    header = 76
    panel_width = tactile_field.shape[2] * scale
    panel_height = tactile_field.shape[1] * scale + header
    individual_size = (panel_width, panel_height)
    composite_scale = 5
    composite_header = 90
    composite_panel_width = tactile_field.shape[2] * composite_scale
    composite_panel_height = tactile_field.shape[1] * composite_scale
    composite_size = (
        composite_panel_width * 2,
        composite_panel_height * 2 + composite_header,
    )
    paths = {name: output_dir / name for name in CORE_VIDEO_NAMES}
    writers = {
        CORE_VIDEO_NAMES[0]: _open_video(
            paths[CORE_VIDEO_NAMES[0]], fps, individual_size
        ),
        CORE_VIDEO_NAMES[1]: _open_video(
            paths[CORE_VIDEO_NAMES[1]], fps, individual_size
        ),
        CORE_VIDEO_NAMES[2]: _open_video(
            paths[CORE_VIDEO_NAMES[2]], fps, individual_size
        ),
        CORE_VIDEO_NAMES[3]: _open_video(
            paths[CORE_VIDEO_NAMES[3]], fps, individual_size
        ),
        CORE_VIDEO_NAMES[4]: _open_video(
            paths[CORE_VIDEO_NAMES[4]], fps, composite_size
        ),
    }
    try:
        for frame_index in range(frame_count):
            if diagnostic_mode:
                frame_ranges = (
                    _nonzero_percentile_range(
                        tactile_field[frame_index, ..., 0], diagnostic_percentile
                    ),
                    _nonzero_percentile_range(
                        tactile_field[frame_index, ..., 1], diagnostic_percentile
                    ),
                    _nonzero_percentile_range(
                        tactile_field[frame_index, ..., 2], diagnostic_percentile
                    ),
                    _nonzero_percentile_range(
                        shear_magnitude[frame_index], diagnostic_percentile
                    ),
                )
            else:
                frame_ranges = (
                    render_ranges["Fx"],
                    render_ranges["Fy"],
                    render_ranges["Fz"],
                    render_ranges["shear_magnitude"],
                )
            per_frame_ranges[frame_index] = frame_ranges
            colors = (
                _signed_bgr(
                    tactile_field[frame_index, ..., 0], frame_ranges[0], gamma=gamma
                ),
                _signed_bgr(
                    tactile_field[frame_index, ..., 1], frame_ranges[1], gamma=gamma
                ),
                _positive_bgr(
                    tactile_field[frame_index, ..., 2], frame_ranges[2], gamma=gamma
                ),
                _positive_bgr(
                    shear_magnitude[frame_index], frame_ranges[3], gamma=gamma
                ),
            )
            titles = (
                "tactile Fx signed",
                "tactile Fy signed",
                "tactile Fz",
                "shear magnitude",
            )
            for video_name, color, title in zip(
                CORE_VIDEO_NAMES[:4], colors, titles, strict=True
            ):
                writers[video_name].write(
                    _annotated_panel(
                        color,
                        title=title,
                        frame_index=frame_index,
                        phase=str(phases[frame_index]),
                        cycle=int(cycles[frame_index]),
                        totals=totals[frame_index],
                        scale=scale,
                        header_height=header,
                    )
                )

            composite = np.zeros(
                (composite_size[1], composite_size[0], 3), dtype=np.uint8
            )
            cv2.putText(
                composite,
                f"frame {frame_index + 1} | phase {phases[frame_index]} | cycle {int(cycles[frame_index]) + 1}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                composite,
                f"Fx {totals[frame_index, 0]:+.6f}  Fy {totals[frame_index, 1]:+.6f}  Fz {totals[frame_index, 2]:+.6f}  unit = TU",
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
            if diagnostic_mode:
                cv2.putText(
                    composite,
                    f"diagnostic per-frame nonzero p{float(diagnostic_percentile):g} | gamma {gamma:g}",
                    (12, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (190, 190, 190),
                    1,
                    cv2.LINE_AA,
                )
            for panel_index, (color, title) in enumerate(
                zip(colors, ("Fx", "Fy", "Fz", "shear"), strict=True)
            ):
                resized = cv2.resize(
                    color,
                    (composite_panel_width, composite_panel_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                row, col = divmod(panel_index, 2)
                y0 = composite_header + row * composite_panel_height
                x0 = col * composite_panel_width
                composite[
                    y0 : y0 + composite_panel_height, x0 : x0 + composite_panel_width
                ] = resized
                cv2.putText(
                    composite,
                    title,
                    (x0 + 8, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            writers[CORE_VIDEO_NAMES[4]].write(composite)
    finally:
        for writer in writers.values():
            writer.release()

    decode = {name: _decode_video(path) for name, path in paths.items()}
    videos_complete = all(
        bool(item["opened"])
        and int(item["decoded_frame_count"]) == frame_count
        and int(item["reported_frame_count"]) == frame_count
        for item in decode.values()
    )
    video_metrics: dict[str, object] = {
        "fps": float(fps),
        "codec": "mp4v",
        "frame_count": frame_count,
        "normalization": (
            "diagnostic_per_frame_nonzero_percentile"
            if diagnostic_mode
            else "fixed_global_range_over_complete_sequence"
        ),
        "global_reference_ranges_tu_per_cell": {
            "Fx": [-ranges["Fx"], ranges["Fx"]],
            "Fy": [-ranges["Fy"], ranges["Fy"]],
            "Fz": [0.0, ranges["Fz"]],
            "shear_magnitude": [0.0, ranges["shear_magnitude"]],
        },
        "decode_validation": decode,
        "all_videos_decode_with_expected_frame_count": videos_complete,
    }
    if diagnostic_mode:
        video_metrics["diagnostic_percentile_of_nonzero_magnitude"] = float(
            diagnostic_percentile
        )
        video_metrics["diagnostic_gamma"] = gamma
        video_metrics["per_frame_upper_range_summary_tu_per_cell"] = {
            name: {
                "minimum": float(np.min(per_frame_ranges[:, index])),
                "maximum": float(np.max(per_frame_ranges[:, index])),
            }
            for index, name in enumerate(("Fx", "Fy", "Fz", "shear_magnitude"))
        }
    else:
        video_metrics["display_ranges_tu_per_cell"] = {
            "Fx": [-ranges["Fx"], ranges["Fx"]],
            "Fy": [-ranges["Fy"], ranges["Fy"]],
            "Fz": [0.0, ranges["Fz"]],
            "shear_magnitude": [0.0, ranges["shear_magnitude"]],
        }
    return video_metrics


def _estimator_gains(output_dir: Path) -> dict[str, float]:
    metadata_path = output_dir / "frozen_7g_force" / "metadata.json"
    metadata = (
        _load_json(metadata_path, "frozen 7g metadata")
        if metadata_path.is_file()
        else {}
    )
    return {
        "normal_gain_tu_per_m3": float(metadata.get("normal_gain_tu_per_m3", 1.0e9)),
        "tangent_y_gain_tu_per_m3": float(
            metadata.get("tangent_y_gain_tu_per_m3", 1.0e9)
        ),
        "tangent_z_gain_tu_per_m3": float(
            metadata.get("tangent_z_gain_tu_per_m3", 1.0e9)
        ),
    }


def build_tactile_field_outputs(
    *,
    output_dir: Path,
    contract_dir: Path,
    source_v8_metadata: dict[str, object] | None = None,
    source_v8_verdict: dict[str, object] | None = None,
    frozen_hashes_before: dict[str, str] | None = None,
    height: int = FIELD_HEIGHT,
    width: int = FIELD_WIDTH,
    sigma_cells: float = 1.25,
    truncate_sigma: float = 4.0,
    expected_frame_count: int = 225,
    expected_cycle_count: int = 5,
    no_contact_tolerance_tu: float = 1.0e-3,
    active_epsilon: float = 1.0e-12,
    video_fps: float = 15.0,
) -> dict[str, object]:
    output_dir = Path(output_dir).expanduser().resolve()
    contract_dir = Path(contract_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_metadata_path = (
        output_dir / "source_v8_metadata.json"
        if (output_dir / "source_v8_metadata.json").is_file()
        else output_dir / "metadata.json"
    )
    source_verdict_path = (
        output_dir / "source_v8_verdict.json"
        if (output_dir / "source_v8_verdict.json").is_file()
        else output_dir / "verdict.json"
    )
    source_metadata = source_v8_metadata or _load_json(
        source_metadata_path, "v8 metadata"
    )
    source_verdict = source_v8_verdict or _load_json(source_verdict_path, "v8 verdict")
    frozen_7g_verdict = _load_json(
        output_dir / "frozen_7g_force" / "verdict.json", "frozen 7g verdict"
    )
    frozen_7g_observed = frozen_7g_verdict.get("observed", {})
    baseline_normal = (
        float(
            frozen_7g_observed.get(
                "baseline_normal_deformation_volume_m3", float("nan")
            )
        )
        if isinstance(frozen_7g_observed, dict)
        else float("nan")
    )
    baseline_shear = np.asarray(
        (
            frozen_7g_observed.get(
                "baseline_shear_deformation_volume_m3", [float("nan")]
            )
            if isinstance(frozen_7g_observed, dict)
            else [float("nan")]
        ),
        dtype=np.float64,
    )
    no_baseline_subtraction = bool(
        math.isfinite(baseline_normal)
        and baseline_normal == 0.0
        and baseline_shear.shape == (2,)
        and np.all(baseline_shear == 0.0)
    )
    _write_json(output_dir / "source_v8_metadata.json", source_metadata)
    _write_json(output_dir / "source_v8_verdict.json", source_verdict)

    deformation = np.asarray(
        _load_array(output_dir / "surface_deformation.npy", "surface deformation"),
        dtype=np.float64,
    )
    frozen_7g_dir = output_dir / "frozen_7g_force"
    vertex_volume = np.asarray(
        _load_array(
            frozen_7g_dir / "vertex_deformation_volume_contribution.npy",
            "frozen 7g vertex deformation volume",
        ),
        dtype=np.float64,
    )
    activation = np.asarray(
        _load_array(
            frozen_7g_dir / "contact_activation_weight.npy", "contact activation"
        ),
        dtype=np.float64,
    )
    tactile_total = np.asarray(
        _load_array(
            output_dir / "tactile_force_channels.npy", "tactile force channels"
        ),
        dtype=np.float64,
    )
    rest_surface = np.asarray(
        _load_array(
            output_dir / "rest_surface_vertices_pad_local.npy", "rest surface vertices"
        ),
        dtype=np.float64,
    )
    front_indices = np.asarray(
        _load_array(output_dir / "front_surface_indices.npy", "front surface indices"),
        dtype=np.int64,
    ).reshape(-1)
    vertex_area_full = np.asarray(
        _load_array(contract_dir / "vertex_area.npy", "7f vertex area"),
        dtype=np.float64,
    ).reshape(-1)
    front_mask = np.asarray(
        _load_array(contract_dir / "front_surface_mask.npy", "7f front mask"),
        dtype=bool,
    ).reshape(-1)
    if deformation.ndim != 3 or deformation.shape[2] != 3:
        raise ValueError(
            f"surface_deformation.npy must be [T,N,3], got {deformation.shape}."
        )
    full_shape = deformation.shape[:2]
    if vertex_volume.shape != (*full_shape, 3):
        raise ValueError(
            "Frozen 7g vertex contribution shape does not match deformation."
        )
    if activation.shape != full_shape or rest_surface.shape != (full_shape[1], 3):
        raise ValueError("Activation or rest-surface shape does not match deformation.")
    if tactile_total.shape != (full_shape[0], 3):
        raise ValueError("tactile_force_channels.npy must have shape [T,3].")
    if vertex_area_full.shape != (full_shape[1],) or front_mask.shape != (
        full_shape[1],
    ):
        raise ValueError(
            "Frozen 7f area/mask does not match the v8 surface vertex count."
        )
    if (
        np.any(front_indices < 0)
        or np.any(front_indices >= full_shape[1])
        or np.unique(front_indices).size != front_indices.size
        or np.any(~front_mask[front_indices])
    ):
        raise ValueError(
            "Runtime front-surface indices violate the frozen 7f front mask."
        )
    if front_indices.size != int(np.count_nonzero(front_mask)):
        raise ValueError("Runtime and frozen 7f front-surface vertex counts differ.")
    for name, value in (
        ("deformation", deformation),
        ("vertex volume", vertex_volume),
        ("activation", activation),
        ("tactile total", tactile_total),
        ("rest surface", rest_surface),
        ("vertex area", vertex_area_full),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or Inf.")

    gains = _estimator_gains(output_dir)
    pad_vertices_full, tactile_vertices_full = tactile_vertex_contributions(
        vertex_volume, **gains
    )
    tactile_vertices = tactile_vertices_full[:, front_indices, :]
    pad_vertices = pad_vertices_full[:, front_indices, :]
    displacement_vertices = deformation[:, front_indices, :]
    activation_vertices = activation[:, front_indices]
    front_yz = rest_surface[front_indices, 1:3]
    vertex_area = vertex_area_full[front_indices]

    plan = build_gaussian_splat_plan(
        front_yz,
        height=int(height),
        width=int(width),
        sigma_cells=float(sigma_cells),
        truncate_sigma=float(truncate_sigma),
    )
    tactile_field = splat_vertex_values(tactile_vertices, plan)
    activation_field = splat_vertex_values(activation_vertices, plan)
    shear_magnitude = np.linalg.norm(tactile_field[..., :2], axis=3)
    dy_mm = abs(float(plan.grid_y_m[1] - plan.grid_y_m[0])) * 1000.0
    dz_mm = abs(float(plan.grid_z_m[1] - plan.grid_z_m[0])) * 1000.0
    cell_area_mm2 = dy_mm * dz_mm
    density = tactile_field / cell_area_mm2

    phases = np.asarray(
        json.loads((output_dir / "phase_history.json").read_text()), dtype=str
    )
    cycles = np.asarray(
        _load_array(output_dir / "cycle_index.npy", "cycle index"), dtype=np.int64
    ).reshape(-1)
    opening = np.asarray(
        _load_array(output_dir / "measured_opening_mm.npy", "measured opening"),
        dtype=np.float64,
    ).reshape(-1)
    if (
        phases.shape != (full_shape[0],)
        or cycles.shape != phases.shape
        or opening.shape != phases.shape
    ):
        raise ValueError("Phase, cycle, and gripper histories must all match T.")

    arrays = {
        "force_pad_local_vertices.npy": pad_vertices,
        "tactile_force_vertices.npy": tactile_vertices,
        "tactile_fx_vertices.npy": tactile_vertices[..., 0],
        "tactile_fy_vertices.npy": tactile_vertices[..., 1],
        "tactile_fz_vertices.npy": tactile_vertices[..., 2],
        "surface_displacement_vertices.npy": displacement_vertices,
        "contact_activation_weight.npy": activation_vertices,
        "front_surface_yz.npy": front_yz,
        "vertex_area.npy": vertex_area,
        "field_grid_y_m.npy": plan.grid_y_m,
        "field_grid_z_m.npy": plan.grid_z_m,
        "tactile_force_field_tu.npy": tactile_field,
        "tactile_fx_field_tu.npy": tactile_field[..., 0],
        "tactile_fy_field_tu.npy": tactile_field[..., 1],
        "tactile_fz_field_tu.npy": tactile_field[..., 2],
        "tactile_shear_magnitude_tu.npy": shear_magnitude,
        "contact_activation_field.npy": activation_field,
        "tactile_force_density_tu_per_mm2.npy": density,
    }
    for name, value in arrays.items():
        np.save(output_dir / name, value)

    field_metrics, conservation, repeatability = calculate_spatial_metrics(
        tactile_vertices=tactile_vertices,
        tactile_total=tactile_total,
        tactile_field=tactile_field,
        activation_field=activation_field,
        phases=phases,
        cycles=cycles,
        measured_opening_mm=opening,
        expected_frame_count=int(expected_frame_count),
        expected_cycle_count=int(expected_cycle_count),
        no_contact_tolerance_tu=float(no_contact_tolerance_tu),
        active_epsilon=float(active_epsilon),
    )
    video_metrics = render_tactile_videos(
        output_dir=output_dir,
        tactile_field=tactile_field,
        shear_magnitude=shear_magnitude,
        phases=phases,
        cycles=cycles,
        fps=float(video_fps),
    )
    field_metrics["video"] = video_metrics
    _write_json(output_dir / "tactile_field_metrics.json", field_metrics)
    _write_json(output_dir / "field_conservation_metrics.json", conservation)
    _write_json(output_dir / "cycle_spatial_repeatability.json", repeatability)

    hashes_before = frozen_hashes_before or capture_frozen_hashes()
    hashes_after = capture_frozen_hashes()
    frozen_unchanged = hashes_before == hashes_after
    source_checks = source_verdict.get("checks", {})
    source_v8_passed = bool(
        source_verdict.get("v5_new_8_grasp_tactile_integration_passed", False)
    )
    checks = {
        "frozen_7f_7g_v8_hashes_unchanged": frozen_unchanged,
        "frozen_7f_contract_passed": bool(
            isinstance(source_checks, dict)
            and source_checks.get("frozen_7f_contract_passed", False)
        ),
        "frozen_7g_estimator_passed": bool(
            isinstance(source_checks, dict)
            and source_checks.get("frozen_7g_estimator_passed", False)
        ),
        "baseline_subtraction_not_used": no_baseline_subtraction,
        "source_v8_integration_passed": source_v8_passed,
        "all_exported_arrays_finite": all(
            np.all(np.isfinite(value)) for value in arrays.values()
        ),
        **field_metrics["spatial_checks"],
        "five_videos_decode_with_expected_frame_count": bool(
            video_metrics["all_videos_decode_with_expected_frame_count"]
        ),
        "video_ranges_are_fixed_over_complete_sequence": video_metrics["normalization"]
        == "fixed_global_range_over_complete_sequence",
    }
    verdict = {
        "v5_new_9_tu_tactile_field_rendering_passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "vertex_sum_max_absolute_error_tu": 1.0e-8,
            "vertex_sum_max_relative_error": 1.0e-6,
            "field_nonzero_max_relative_error": 0.01,
            "field_near_zero_max_absolute_error_tu": 1.0e-6,
            "no_contact_maximum_abs_axis_total_tu": float(no_contact_tolerance_tu),
            "close_fz_spearman_strictly_greater_than": 0.95,
            "hold_total_cv_strictly_less_than": 0.02,
            "hold_centroid_drift_cells_strictly_less_than": 1.0,
            "repeat_peak_and_total_cv_strictly_less_than": 0.02,
            "repeat_centroid_offset_cells_strictly_less_than": 2.0,
            "repeat_field_correlation_strictly_greater_than": 0.95,
            "release_to_peak_strictly_less_than": 0.02,
            "expected_frame_count": int(expected_frame_count),
            "expected_cycle_count": int(expected_cycle_count),
        },
        "observed": {
            "frozen_hashes_before": hashes_before,
            "frozen_hashes_after": hashes_after,
            "field_metrics": field_metrics,
            "conservation": conservation,
            "repeatability": repeatability,
        },
        "force_source": "uipc_membrane_surface_deformation_reduced_order",
        "spatial_field_source": "per_vertex_estimator_contribution",
        "field_mapping": "force_conserving_gaussian_splat",
        "force_unit": "TU",
        "field_unit": "TU_per_cell",
        "newton_calibrated": False,
        "damping_used": False,
    }

    metadata = dict(source_metadata)
    metadata.update(
        {
            "script_version": V9_VERSION,
            "force_source": "uipc_membrane_surface_deformation_reduced_order",
            "spatial_field_source": "per_vertex_estimator_contribution",
            "field_mapping": "force_conserving_gaussian_splat",
            "force_unit": "TU",
            "field_unit": "TU_per_cell",
            "density_unit": "TU_per_mm2",
            "newton_calibrated": False,
            "damping_used": False,
            "baseline_subtraction_used": False,
            "frozen_7f_contract_dir": str(contract_dir),
            "force_semantics": "object_on_sensor",
            "contact_geometry_role": "none",
            "native_uipc_contact_force_used": False,
            "frame_count": int(full_shape[0]),
            "cycle_count": int(np.unique(cycles).size),
            "front_surface_vertex_count": int(front_indices.size),
            "front_surface_coordinate_frame": "pad_local_YZ",
            "front_surface_coordinate_unit": "m",
            "field_shape": [int(height), int(width)],
            "field_grid_orientation": {
                "columns": "pad_local_Y_min_to_max",
                "rows": "pad_local_Z_max_to_min",
            },
            "gaussian_sigma_cells": float(sigma_cells),
            "gaussian_sigma_m": float(plan.sigma_m),
            "gaussian_truncate_sigma": float(truncate_sigma),
            "gaussian_kernel_normalization": "each_vertex_weights_sum_to_one",
            "field_cell_area_mm2": cell_area_mm2,
            "field_semantics": "each grid cell carries a TU force contribution",
            "density_semantics": "simulated TU area density; not Pa or Newton",
            "video": video_metrics,
            "frozen_source_hashes": hashes_after,
        }
    )
    metadata["outputs"] = {
        **dict(metadata.get("outputs", {})),
        "force_pad_local": str(output_dir / "force_pad_local.npy"),
        "tactile_force_channels": str(output_dir / "tactile_force_channels.npy"),
        "surface_deformation_pad_local": str(output_dir / "surface_deformation.npy"),
        "vertex_tactile_field": str(
            output_dir / "vertex_deformation_volume_contribution.npy"
        ),
        **{name.removesuffix(".npy"): str(output_dir / name) for name in arrays},
        **{
            name.removesuffix(".mp4"): str(output_dir / name)
            for name in CORE_VIDEO_NAMES
        },
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "verdict.json", verdict)
    _write_json(output_dir / "summary.json", {**metadata, "verdict": verdict})
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build v9 force-conserving TU tactile fields and fixed-range videos from a passed v8 output."
    )
    parser.add_argument(
        "--source_dir", default="/tmp/openworldtactile_uipc_v5_new_8_grasp_tactile_integration"
    )
    parser.add_argument(
        "--output_dir", default="/tmp/openworldtactile_uipc_v5_new_9_tu_tactile_field_rendering"
    )
    parser.add_argument(
        "--contract_dir", default="/tmp/openworldtactile_uipc_v5_new_7f_contract_verified"
    )
    parser.add_argument("--field_height", type=int, default=FIELD_HEIGHT)
    parser.add_argument("--field_width", type=int, default=FIELD_WIDTH)
    parser.add_argument("--gaussian_sigma_cells", type=float, default=1.25)
    parser.add_argument("--gaussian_truncate_sigma", type=float, default=4.0)
    parser.add_argument("--expected_frame_count", type=int, default=225)
    parser.add_argument("--expected_cycle_count", type=int, default=5)
    parser.add_argument("--no_contact_tolerance_tu", type=float, default=1.0e-3)
    parser.add_argument("--active_epsilon", type=float, default=1.0e-12)
    parser.add_argument("--video_fps", type=float, default=15.0)
    parser.add_argument("--fail_on_verdict_fail", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"V8 source output is missing: {source_dir}")
    if source_dir != output_dir:
        if output_dir.exists():
            raise FileExistsError(
                f"Refusing to merge with an existing output directory: {output_dir}"
            )
        shutil.copytree(source_dir, output_dir)
    hashes_before = capture_frozen_hashes()
    verdict = build_tactile_field_outputs(
        output_dir=output_dir,
        contract_dir=Path(args.contract_dir),
        frozen_hashes_before=hashes_before,
        height=int(args.field_height),
        width=int(args.field_width),
        sigma_cells=float(args.gaussian_sigma_cells),
        truncate_sigma=float(args.gaussian_truncate_sigma),
        expected_frame_count=int(args.expected_frame_count),
        expected_cycle_count=int(args.expected_cycle_count),
        no_contact_tolerance_tu=float(args.no_contact_tolerance_tu),
        active_epsilon=float(args.active_epsilon),
        video_fps=float(args.video_fps),
    )
    print(
        json.dumps(verdict, indent=2, ensure_ascii=False, allow_nan=False), flush=True
    )
    if bool(args.fail_on_verdict_fail) and not bool(
        verdict["v5_new_9_tu_tactile_field_rendering_passed"]
    ):
        raise RuntimeError(f"v9 TU tactile field rendering failed: {verdict}")


if __name__ == "__main__":
    main()
