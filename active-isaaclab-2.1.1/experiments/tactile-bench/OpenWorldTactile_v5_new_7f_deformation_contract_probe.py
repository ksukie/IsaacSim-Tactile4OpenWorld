from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


CONTRACT_VERSION = "v5_new_7f_deformation_contract_v1"
EPS = 1.0e-12


@dataclass(frozen=True)
class SurfaceRun:
    input_dir: Path
    rest_surface_l: np.ndarray
    current_surface_l: np.ndarray
    displacement_l: np.ndarray
    surface_world: np.ndarray
    pad_pose_world: np.ndarray
    surface_triangles: np.ndarray
    front_mask: np.ndarray
    back_mask: np.ndarray
    phases: tuple[str, ...]
    commanded_indentation_mm: np.ndarray | None
    actual_indentation_mm: np.ndarray | None
    metadata: dict[str, object]
    verdict: dict[str, object]
    producer_reconstruction_error_mm: float | None

    @property
    def frame_count(self) -> int:
        return int(self.current_surface_l.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.rest_surface_l.shape[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate the v5_new_7f Pad-local deformation contract. The probe consumes "
            "one no-contact rigid-follow run and five or more independent normal-indentation runs. "
            "It does not read penetration/contact masks and does not estimate force or pressure."
        )
    )
    parser.add_argument("--rigid_input_dir", required=True, help="Completed 7d no-contact link-motion run.")
    parser.add_argument("--normal_input_dir", required=True, help="Primary completed 7e normal-indentation run.")
    parser.add_argument(
        "--repeat_input_dir",
        action="append",
        default=[],
        help="Additional independent 7e run. Repeat four times for the default five-run acceptance suite.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_contact_tail_frames", type=int, default=10)
    parser.add_argument("--center_radius_mm", type=float, default=1.0)
    parser.add_argument("--radial_bin_width_mm", type=float, default=1.0)
    parser.add_argument("--accept_max_no_contact_residual_mm", type=float, default=0.05)
    parser.add_argument("--accept_min_rigid_world_motion_mm", type=float, default=1.0)
    parser.add_argument("--accept_max_rigid_local_residual_mm", type=float, default=0.05)
    parser.add_argument("--accept_max_transform_reconstruction_mm", type=float, default=1.0e-3)
    parser.add_argument("--accept_min_peak_normal_mm", type=float, default=0.10)
    parser.add_argument("--accept_max_peak_center_distance_mm", type=float, default=1.5)
    parser.add_argument("--accept_max_center_shear_normal_ratio", type=float, default=0.25)
    parser.add_argument("--accept_max_back_residual_mm", type=float, default=0.20)
    parser.add_argument("--accept_min_repeat_runs", type=int, default=5)
    parser.add_argument("--accept_max_repeat_peak_relative_error", type=float, default=0.05)
    parser.add_argument("--accept_max_repeat_field_nrmse", type=float, default=0.05)
    parser.add_argument("--fail_on_verdict_fail", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive_names = (
        "no_contact_tail_frames",
        "center_radius_mm",
        "radial_bin_width_mm",
        "accept_max_no_contact_residual_mm",
        "accept_min_rigid_world_motion_mm",
        "accept_max_rigid_local_residual_mm",
        "accept_max_transform_reconstruction_mm",
        "accept_min_peak_normal_mm",
        "accept_max_peak_center_distance_mm",
        "accept_max_center_shear_normal_ratio",
        "accept_max_back_residual_mm",
        "accept_min_repeat_runs",
        "accept_max_repeat_peak_relative_error",
        "accept_max_repeat_field_nrmse",
    )
    for name in positive_names:
        if float(getattr(args, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    if int(args.accept_min_repeat_runs) < 2:
        parser.error("--accept_min_repeat_runs must be >= 2.")
    for name in ("accept_max_repeat_peak_relative_error", "accept_max_repeat_field_nrmse"):
        if float(getattr(args, name)) > 1.0:
            parser.error(f"--{name} must be <= 1.")


def _load_npy(directory: Path, name: str, *, required: bool = True) -> np.ndarray | None:
    path = directory / name
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required producer output is missing: {path}")
        return None
    return np.load(path, allow_pickle=False)


def _load_json(directory: Path, name: str, *, required: bool = True) -> dict[str, object]:
    path = directory / name
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required producer output is missing: {path}")
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _load_phases(directory: Path) -> tuple[str, ...]:
    path = directory / "phase_history.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required producer output is missing: {path}")
    values = json.loads(path.read_text())
    if not isinstance(values, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return tuple(str(value) for value in values)


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= EPS:
        raise ValueError(f"Invalid Pad quaternion: {quat.tolist()}")
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world_to_pad_local(surface_world: np.ndarray, pad_pose_world: np.ndarray) -> np.ndarray:
    surface = np.asarray(surface_world, dtype=np.float64)
    poses = np.asarray(pad_pose_world, dtype=np.float64)
    if surface.ndim != 3 or surface.shape[2] != 3:
        raise ValueError(f"Expected surface history [T,N,3], got {surface.shape}.")
    if poses.shape != (surface.shape[0], 7):
        raise ValueError(f"Expected Pad pose history {(surface.shape[0], 7)}, got {poses.shape}.")
    result = np.empty_like(surface)
    for frame in range(surface.shape[0]):
        rotation_world_from_pad = _quat_to_matrix(poses[frame, 3:7])
        result[frame] = (surface[frame] - poses[frame, :3]) @ rotation_world_from_pad
    return result


def _indices_to_mask(indices: np.ndarray, vertex_count: int, name: str) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if np.any(values < 0) or np.any(values >= vertex_count):
        raise ValueError(f"{name} contains indices outside [0, {vertex_count}).")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} contains duplicate indices.")
    mask = np.zeros(vertex_count, dtype=bool)
    mask[values] = True
    return mask


def load_surface_run(input_dir: Path) -> SurfaceRun:
    directory = input_dir.expanduser().resolve()
    rest = np.asarray(_load_npy(directory, "rest_surface_vertices_pad_local.npy"), dtype=np.float64)
    surface_world = np.asarray(_load_npy(directory, "uipc_surface_w.npy"), dtype=np.float64)
    pad_pose_world = np.asarray(_load_npy(directory, "pad_pose.npy"), dtype=np.float64)
    triangles = np.asarray(_load_npy(directory, "surface_triangles.npy"), dtype=np.int64).reshape(-1, 3)
    front_indices = np.asarray(_load_npy(directory, "front_surface_indices.npy"), dtype=np.int64)
    back_indices = np.asarray(_load_npy(directory, "back_surface_indices.npy"), dtype=np.int64)
    phases = _load_phases(directory)
    metadata = _load_json(directory, "metadata.json")
    verdict = _load_json(directory, "verdict.json")

    if rest.ndim != 2 or rest.shape[1] != 3:
        raise ValueError(f"Expected rest surface [N,3], got {rest.shape} in {directory}.")
    if not np.all(np.isfinite(rest)) or not np.all(np.isfinite(surface_world)):
        raise ValueError(f"Non-finite surface values in {directory}.")
    if surface_world.shape[1:] != rest.shape:
        raise ValueError(f"Surface/rest shape mismatch in {directory}: {surface_world.shape} vs {rest.shape}.")
    if len(phases) != surface_world.shape[0]:
        raise ValueError(f"Phase count does not match frame count in {directory}.")
    if triangles.size == 0 or np.any(triangles < 0) or np.any(triangles >= rest.shape[0]):
        raise ValueError(f"Invalid or empty surface topology in {directory}.")

    front_mask = _indices_to_mask(front_indices, rest.shape[0], "front_surface_indices")
    back_mask = _indices_to_mask(back_indices, rest.shape[0], "back_surface_indices")
    if np.any(front_mask & back_mask):
        raise ValueError(f"Front/back masks overlap in {directory}.")

    current_l = world_to_pad_local(surface_world, pad_pose_world)
    displacement_l = current_l - rest.reshape(1, *rest.shape)
    stored = _load_npy(directory, "surface_deformation.npy", required=False)
    reconstruction_error_mm: float | None = None
    if stored is not None:
        stored = np.asarray(stored, dtype=np.float64)
        if stored.shape != displacement_l.shape:
            raise ValueError(f"Stored deformation shape mismatch in {directory}.")
        reconstruction_error_mm = float(np.max(np.abs(stored - displacement_l)) * 1000.0)

    commanded = _load_npy(directory, "commanded_indentation_mm.npy", required=False)
    actual = _load_npy(directory, "actual_indentation_mm.npy", required=False)
    if commanded is not None:
        commanded = np.asarray(commanded, dtype=np.float64).reshape(-1)
        if commanded.shape != (surface_world.shape[0],):
            raise ValueError(f"Commanded indentation history mismatch in {directory}.")
    if actual is not None:
        actual = np.asarray(actual, dtype=np.float64).reshape(-1)
        if actual.shape != (surface_world.shape[0],):
            raise ValueError(f"Actual indentation history mismatch in {directory}.")

    return SurfaceRun(
        input_dir=directory,
        rest_surface_l=rest,
        current_surface_l=current_l,
        displacement_l=displacement_l,
        surface_world=surface_world,
        pad_pose_world=pad_pose_world,
        surface_triangles=triangles,
        front_mask=front_mask,
        back_mask=back_mask,
        phases=phases,
        commanded_indentation_mm=commanded,
        actual_indentation_mm=actual,
        metadata=metadata,
        verdict=verdict,
        producer_reconstruction_error_mm=reconstruction_error_mm,
    )


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _contract_identity(run: SurfaceRun) -> dict[str, str]:
    return {
        "rest_surface_sha256": _array_hash(run.rest_surface_l.astype(np.float32)),
        "surface_triangles_sha256": _array_hash(run.surface_triangles.astype(np.int64)),
        "front_surface_mask_sha256": _array_hash(run.front_mask),
        "back_surface_mask_sha256": _array_hash(run.back_mask),
    }


def compute_front_vertex_area(run: SurfaceRun) -> tuple[np.ndarray, np.ndarray]:
    triangle_is_front = np.all(run.front_mask[run.surface_triangles], axis=1)
    front_triangles = run.surface_triangles[triangle_is_front]
    if front_triangles.size == 0:
        raise ValueError("Surface topology contains no triangles fully inside the front-surface mask.")
    points = run.rest_surface_l[front_triangles]
    face_area = 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )
    if np.any(~np.isfinite(face_area)) or np.any(face_area <= 0.0):
        raise ValueError("Front surface contains non-finite or degenerate triangles.")
    vertex_area = np.zeros(run.vertex_count, dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_area, front_triangles[:, corner], face_area / 3.0)
    if np.any(vertex_area[run.front_mask] <= 0.0):
        missing = np.flatnonzero(run.front_mask & (vertex_area <= 0.0))
        raise ValueError(f"Front vertices without rest-area support: {missing[:20].tolist()}")
    return vertex_area, front_triangles


def _phase_end_indices(phases: Sequence[str], prefix: str) -> list[int]:
    return [
        index
        for index, phase in enumerate(phases)
        if str(phase).startswith(prefix)
        and (index == len(phases) - 1 or phases[index + 1] != phase)
    ]


def select_peak_hold_frame(run: SurfaceRun) -> int:
    if run.commanded_indentation_mm is None or run.actual_indentation_mm is None:
        raise ValueError(f"Normal run lacks indentation histories: {run.input_dir}")
    candidates = _phase_end_indices(run.phases, "load_hold_")
    if not candidates:
        raise ValueError(f"Normal run has no completed load_hold phase: {run.input_dir}")
    return int(
        max(
            candidates,
            key=lambda index: (
                float(run.commanded_indentation_mm[index]),
                float(run.actual_indentation_mm[index]),
            ),
        )
    )


def _source_passed(run: SurfaceRun, key: str) -> bool:
    return bool(run.verdict.get(key, False))


def _expected_contact_center_yz(run: SurfaceRun) -> tuple[np.ndarray, str]:
    authored = run.metadata.get("tool_center_pad_local_m")
    if isinstance(authored, list) and len(authored) == 3:
        center = np.asarray(authored, dtype=np.float64)
        if np.all(np.isfinite(center)):
            return center[1:3], "authored_tool_center_pad_local"
    return np.mean(run.rest_surface_l[run.front_mask, 1:3], axis=0), "rest_front_surface_centroid"


def _radial_profile(
    radial_distance_mm: np.ndarray,
    normal_compression_mm: np.ndarray,
    shear_magnitude_mm: np.ndarray,
    bin_width_mm: float,
) -> list[dict[str, float | int]]:
    maximum = max(float(bin_width_mm), float(np.max(radial_distance_mm)))
    edges = np.arange(0.0, maximum + float(bin_width_mm), float(bin_width_mm), dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (radial_distance_mm >= lower) & (radial_distance_mm < upper)
        if not np.any(selected):
            continue
        rows.append(
            {
                "radius_center_mm": float(0.5 * (lower + upper)),
                "normal_mean_mm": float(np.mean(normal_compression_mm[selected])),
                "normal_max_mm": float(np.max(normal_compression_mm[selected])),
                "shear_mean_mm": float(np.mean(shear_magnitude_mm[selected])),
                "vertex_count": int(np.count_nonzero(selected)),
            }
        )
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _save_array(path: Path, value: np.ndarray, saved: list[Path]) -> None:
    np.save(path, value)
    saved.append(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    rigid_run = load_surface_run(Path(args.rigid_input_dir))
    primary_run = load_surface_run(Path(args.normal_input_dir))
    repeat_paths = [Path(value).expanduser().resolve() for value in args.repeat_input_dir]
    all_normal_paths = [primary_run.input_dir, *repeat_paths]
    if len(set(all_normal_paths)) != len(all_normal_paths):
        raise ValueError("Normal repeat directories must be distinct independent runs.")
    normal_runs = [primary_run, *(load_surface_run(path) for path in repeat_paths)]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    vertex_area, front_triangles = compute_front_vertex_area(primary_run)
    identity = _contract_identity(primary_run)
    peak_frame = select_peak_hold_frame(primary_run)
    displacement = primary_run.displacement_l[peak_frame]
    current_surface = primary_run.current_surface_l[peak_frame]
    normal_compression = np.clip(-displacement[:, 0], 0.0, None)
    shear_displacement = displacement[:, 1:3]
    shear_magnitude = np.linalg.norm(shear_displacement, axis=1)

    # Test 1: no-contact noise uses the settled tail of the producer's explicit pre-contact phase.
    no_contact_indices = [index for index, phase in enumerate(primary_run.phases) if phase == "pre_contact"]
    no_contact_indices = no_contact_indices[-int(args.no_contact_tail_frames) :]
    if no_contact_indices:
        no_contact_field = primary_run.displacement_l[no_contact_indices]
        no_contact_front_norm_mm = (
            np.linalg.norm(no_contact_field[:, primary_run.front_mask], axis=2) * 1000.0
        )
        max_no_contact_residual_mm = float(np.max(no_contact_front_norm_mm))
        p99_no_contact_residual_mm = float(np.percentile(no_contact_front_norm_mm, 99.0))
    else:
        no_contact_field = np.empty((0, primary_run.vertex_count, 3), dtype=np.float64)
        max_no_contact_residual_mm = float("inf")
        p99_no_contact_residual_mm = float("inf")

    # Test 2: evaluate settled no-contact holds so inertia is not confused with frame-removal error.
    rigid_hold_indices = [
        index for index, phase in enumerate(rigid_run.phases) if str(phase).startswith("hold_")
    ]
    rigid_eval_indices = rigid_hold_indices or list(range(rigid_run.frame_count))
    rigid_local_norm_mm = (
        np.linalg.norm(rigid_run.displacement_l[rigid_eval_indices][:, rigid_run.front_mask], axis=2)
        * 1000.0
    )
    max_rigid_local_residual_mm = float(np.max(rigid_local_norm_mm))
    all_frame_max_rigid_local_residual_mm = float(
        np.max(np.linalg.norm(rigid_run.displacement_l[:, rigid_run.front_mask], axis=2)) * 1000.0
    )
    rigid_world_motion_mm = float(
        np.max(np.linalg.norm(rigid_run.surface_world - rigid_run.surface_world[0:1], axis=2)) * 1000.0
    )
    pad_translation_mm = float(
        np.max(np.linalg.norm(rigid_run.pad_pose_world[:, :3] - rigid_run.pad_pose_world[0, :3], axis=1))
        * 1000.0
    )

    # Test 3: pure normal indentation is validated against the authored tool center, not a contact mask.
    front_indices = np.flatnonzero(primary_run.front_mask)
    expected_center_yz, center_source = _expected_contact_center_yz(primary_run)
    front_radial_distance_mm = (
        np.linalg.norm(
            primary_run.rest_surface_l[front_indices, 1:3] - expected_center_yz.reshape(1, 2), axis=1
        )
        * 1000.0
    )
    peak_front_local = int(np.argmax(normal_compression[front_indices]))
    peak_vertex = int(front_indices[peak_front_local])
    peak_normal_mm = float(normal_compression[peak_vertex] * 1000.0)
    peak_center_distance_mm = float(front_radial_distance_mm[peak_front_local])
    center_vertex = int(front_indices[int(np.argmin(front_radial_distance_mm))])
    center_normal_mm = float(normal_compression[center_vertex] * 1000.0)
    center_shear_mm = float(shear_magnitude[center_vertex] * 1000.0)
    center_shear_normal_ratio = center_shear_mm / max(center_normal_mm, EPS)
    center_region = front_radial_distance_mm <= float(args.center_radius_mm)
    if not np.any(center_region):
        center_region[int(np.argmin(front_radial_distance_mm))] = True
    center_region_normal_mean_mm = float(
        np.mean(normal_compression[front_indices][center_region]) * 1000.0
    )
    center_region_shear_mean_mm = float(np.mean(shear_magnitude[front_indices][center_region]) * 1000.0)
    peak_back_residual_mm = float(
        np.max(np.linalg.norm(displacement[primary_run.back_mask], axis=1)) * 1000.0
    )
    radial_profile = _radial_profile(
        front_radial_distance_mm,
        normal_compression[front_indices] * 1000.0,
        shear_magnitude[front_indices] * 1000.0,
        float(args.radial_bin_width_mm),
    )

    # Test 4: independent repeats must preserve identity and agree in peak and full front field.
    identities = [_contract_identity(run) for run in normal_runs]
    repeat_identity_matches = [value == identity for value in identities]
    repeat_peak_frames = [select_peak_hold_frame(run) for run in normal_runs]
    repeat_peaks_mm = np.asarray(
        [
            np.max(np.clip(-run.displacement_l[frame, run.front_mask, 0], 0.0, None)) * 1000.0
            for run, frame in zip(normal_runs, repeat_peak_frames)
        ],
        dtype=np.float64,
    )
    repeat_peak_mean_mm = float(np.mean(repeat_peaks_mm))
    repeat_peak_relative_error = float(
        np.max(np.abs(repeat_peaks_mm - repeat_peak_mean_mm)) / max(repeat_peak_mean_mm, EPS)
    )
    repeat_fields = np.stack(
        [run.displacement_l[frame, run.front_mask] for run, frame in zip(normal_runs, repeat_peak_frames)],
        axis=0,
    )
    mean_repeat_field = np.mean(repeat_fields, axis=0)
    repeat_field_rms = np.sqrt(np.mean(np.square(repeat_fields - mean_repeat_field), axis=(1, 2)))
    repeat_field_nrmse = repeat_field_rms / max(repeat_peak_mean_mm * 1.0e-3, EPS)
    max_repeat_field_nrmse = float(np.max(repeat_field_nrmse))
    producer_errors = [run.producer_reconstruction_error_mm for run in normal_runs]
    finite_producer_errors = [float(value) for value in producer_errors if value is not None]
    max_producer_reconstruction_error_mm = max(finite_producer_errors, default=0.0)

    checks = {
        "producer_pad_local_matches_reconstruction": max_producer_reconstruction_error_mm
        <= float(args.accept_max_transform_reconstruction_mm),
        "front_rest_area_is_positive": bool(np.all(vertex_area[primary_run.front_mask] > 0.0)),
        "source_normal_runs_passed": all(
            _source_passed(run, "static_contact_deformation_passed") for run in normal_runs
        ),
        "test1_no_contact_frames_present": bool(no_contact_indices),
        "test1_no_contact_residual_bounded": max_no_contact_residual_mm
        <= float(args.accept_max_no_contact_residual_mm),
        "test2_source_rigid_run_passed": _source_passed(rigid_run, "backface_attachment_follow_passed"),
        "test2_world_motion_detected": rigid_world_motion_mm
        >= float(args.accept_min_rigid_world_motion_mm),
        "test2_pad_local_rigid_residual_bounded": max_rigid_local_residual_mm
        <= float(args.accept_max_rigid_local_residual_mm),
        "test3_peak_normal_above_noise": peak_normal_mm
        >= max(float(args.accept_min_peak_normal_mm), 3.0 * max_no_contact_residual_mm),
        "test3_peak_matches_authored_center": peak_center_distance_mm
        <= float(args.accept_max_peak_center_distance_mm),
        "test3_center_shear_is_small": center_shear_normal_ratio
        <= float(args.accept_max_center_shear_normal_ratio),
        "test3_back_surface_residual_bounded": peak_back_residual_mm
        <= float(args.accept_max_back_residual_mm),
        "test4_repeat_count_sufficient": len(normal_runs) >= int(args.accept_min_repeat_runs),
        "test4_repeat_contract_identity_matches": all(repeat_identity_matches),
        "test4_repeat_peak_stable": repeat_peak_relative_error
        <= float(args.accept_max_repeat_peak_relative_error),
        "test4_repeat_field_stable": max_repeat_field_nrmse
        <= float(args.accept_max_repeat_field_nrmse),
    }

    observed = {
        "vertex_count": primary_run.vertex_count,
        "front_vertex_count": int(np.count_nonzero(primary_run.front_mask)),
        "back_vertex_count": int(np.count_nonzero(primary_run.back_mask)),
        "front_triangle_count": int(front_triangles.shape[0]),
        "front_rest_area_m2": float(np.sum(vertex_area)),
        "contract_frame": int(peak_frame),
        "contract_phase": primary_run.phases[peak_frame],
        "contract_commanded_indentation_mm": float(primary_run.commanded_indentation_mm[peak_frame]),
        "contract_actual_indentation_mm": float(primary_run.actual_indentation_mm[peak_frame]),
        "max_producer_reconstruction_error_mm": max_producer_reconstruction_error_mm,
        "no_contact_frame_count": len(no_contact_indices),
        "max_no_contact_residual_mm": max_no_contact_residual_mm,
        "p99_no_contact_residual_mm": p99_no_contact_residual_mm,
        "rigid_hold_frame_count": len(rigid_eval_indices),
        "rigid_world_surface_motion_mm": rigid_world_motion_mm,
        "rigid_pad_translation_mm": pad_translation_mm,
        "max_rigid_hold_local_residual_mm": max_rigid_local_residual_mm,
        "max_rigid_all_frame_local_residual_mm": all_frame_max_rigid_local_residual_mm,
        "peak_vertex_index": peak_vertex,
        "peak_normal_compression_mm": peak_normal_mm,
        "peak_to_authored_center_distance_mm": peak_center_distance_mm,
        "center_vertex_index": center_vertex,
        "center_normal_compression_mm": center_normal_mm,
        "center_shear_magnitude_mm": center_shear_mm,
        "center_shear_to_normal_ratio": center_shear_normal_ratio,
        "center_region_normal_mean_mm": center_region_normal_mean_mm,
        "center_region_shear_mean_mm": center_region_shear_mean_mm,
        "peak_back_surface_residual_mm": peak_back_residual_mm,
        "repeat_run_count": len(normal_runs),
        "repeat_peak_normal_mm": [float(value) for value in repeat_peaks_mm],
        "repeat_peak_relative_error": repeat_peak_relative_error,
        "repeat_field_nrmse": [float(value) for value in repeat_field_nrmse],
        "max_repeat_field_nrmse": max_repeat_field_nrmse,
    }
    thresholds = {
        "max_no_contact_residual_mm": float(args.accept_max_no_contact_residual_mm),
        "min_rigid_world_motion_mm": float(args.accept_min_rigid_world_motion_mm),
        "max_rigid_local_residual_mm": float(args.accept_max_rigid_local_residual_mm),
        "max_transform_reconstruction_mm": float(args.accept_max_transform_reconstruction_mm),
        "min_peak_normal_mm": float(args.accept_min_peak_normal_mm),
        "peak_must_exceed_no_contact_noise_factor": 3.0,
        "max_peak_center_distance_mm": float(args.accept_max_peak_center_distance_mm),
        "max_center_shear_normal_ratio": float(args.accept_max_center_shear_normal_ratio),
        "max_back_residual_mm": float(args.accept_max_back_residual_mm),
        "min_repeat_runs": int(args.accept_min_repeat_runs),
        "max_repeat_peak_relative_error": float(args.accept_max_repeat_peak_relative_error),
        "max_repeat_field_nrmse": float(args.accept_max_repeat_field_nrmse),
    }
    verdict: dict[str, object] = {
        "deformation_contract_passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": thresholds,
        "observed": observed,
    }

    saved_arrays: list[Path] = []
    _save_array(output_dir / "rest_surface_pad_local.npy", primary_run.rest_surface_l.astype(np.float32), saved_arrays)
    _save_array(output_dir / "current_surface_pad_local.npy", current_surface.astype(np.float32), saved_arrays)
    _save_array(output_dir / "surface_displacement_pad_local.npy", displacement.astype(np.float32), saved_arrays)
    _save_array(output_dir / "vertex_area.npy", vertex_area.astype(np.float64), saved_arrays)
    _save_array(output_dir / "front_surface_mask.npy", primary_run.front_mask, saved_arrays)
    _save_array(output_dir / "back_surface_mask.npy", primary_run.back_mask, saved_arrays)
    _save_array(output_dir / "normal_compression.npy", normal_compression.astype(np.float32), saved_arrays)
    _save_array(output_dir / "shear_displacement.npy", shear_displacement.astype(np.float32), saved_arrays)
    _save_array(output_dir / "surface_triangles.npy", primary_run.surface_triangles.astype(np.int64), saved_arrays)
    _save_array(output_dir / "front_surface_triangles.npy", front_triangles.astype(np.int64), saved_arrays)
    _save_array(output_dir / "vertex_id.npy", np.arange(primary_run.vertex_count, dtype=np.int64), saved_arrays)
    _save_array(
        diagnostics_dir / "primary_surface_displacement_history_pad_local.npy",
        primary_run.displacement_l.astype(np.float32),
        saved_arrays,
    )
    _save_array(
        diagnostics_dir / "no_contact_surface_displacement_pad_local.npy",
        no_contact_field.astype(np.float32),
        saved_arrays,
    )
    _save_array(
        diagnostics_dir / "rigid_surface_displacement_history_pad_local.npy",
        rigid_run.displacement_l.astype(np.float32),
        saved_arrays,
    )
    _save_array(
        diagnostics_dir / "repeat_peak_surface_displacement_pad_local.npy",
        np.stack(
            [run.displacement_l[frame] for run, frame in zip(normal_runs, repeat_peak_frames)], axis=0
        ).astype(np.float32),
        saved_arrays,
    )

    metadata = {
        "contract_version": CONTRACT_VERSION,
        "coordinate_frame": "pad_local",
        "normal_axis": "+X_outward",
        "tangent_axes": ["+Y", "+Z"],
        "deformation_definition": "Xt-X0",
        "normal_compression_definition": "max(-u_x,0)",
        "shear_displacement_definition": "[u_y,u_z]",
        "surface_prim_role": "simulation/membrane_sim_mesh",
        "rest_state_source": str(primary_run.input_dir / "rest_surface_vertices_pad_local.npy"),
        "contract_source_frame": int(peak_frame),
        "contract_source_phase": primary_run.phases[peak_frame],
        "npy_length_unit": "meter",
        "vertex_area_unit": "meter^2",
        "front_vertex_area_definition": "one_third_rest_triangle_area; zero outside front_surface_mask",
        "vertex_correspondence": "stable UIPC compact surface vertex_id",
        "identity": identity,
        "force_source": "none",
        "pressure_source": "none",
        "contact_penetration_used": False,
        "contact_geometry_used": False,
        "native_uipc_gradient_used": False,
        "proxy_force_used": False,
        "diagnostic_contact_center_source": center_source,
        "allowed_7g_inputs": [
            "surface_displacement_pad_local.npy",
            "vertex_area.npy",
            "front_surface_mask.npy",
        ],
    }
    contact_center = {
        "coordinate_frame": "pad_local",
        "source": center_source,
        "uses_contact_geometry": False,
        "expected_center_yz_m": [float(value) for value in expected_center_yz],
        "peak_vertex_index": peak_vertex,
        "peak_position_yz_m": [
            float(value) for value in primary_run.rest_surface_l[peak_vertex, 1:3]
        ],
        "peak_to_center_distance_mm": peak_center_distance_mm,
    }
    source_runs = {
        "rigid_input_dir": str(rigid_run.input_dir),
        "normal_input_dirs": [str(run.input_dir) for run in normal_runs],
        "normal_contract_identities": identities,
        "repeat_peak_frames": repeat_peak_frames,
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "contact_center.json", contact_center)
    _write_json(output_dir / "radial_profile.json", radial_profile)
    _write_json(output_dir / "source_runs.json", source_runs)
    _write_json(output_dir / "verdict.json", verdict)
    _write_json(output_dir / "summary.json", verdict)

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "arrays": {
            str(path.relative_to(output_dir)): {
                "shape": list(np.load(path, mmap_mode="r", allow_pickle=False).shape),
                "dtype": str(np.load(path, mmap_mode="r", allow_pickle=False).dtype),
                "sha256": _file_sha256(path),
            }
            for path in saved_arrays
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return verdict


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    verdict = run_probe(args)
    print(json.dumps(verdict, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if bool(args.fail_on_verdict_fail) and not bool(verdict["deformation_contract_passed"]):
        raise RuntimeError(f"7f deformation contract verdict failed: {verdict}")


if __name__ == "__main__":
    main()
