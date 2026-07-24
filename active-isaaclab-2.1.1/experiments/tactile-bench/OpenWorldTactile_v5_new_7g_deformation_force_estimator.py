from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


ESTIMATOR_VERSION = "v5_new_7g_deformation_based_force_estimator_v1"
EPS = 1.0e-12


@dataclass(frozen=True)
class EstimatorConfig:
    normal_gain_tu_per_m3: float = 1.0e9
    tangent_y_gain_tu_per_m3: float = 1.0e9
    tangent_z_gain_tu_per_m3: float = 1.0e9
    activation_start_m: float = 0.01e-3
    activation_full_m: float = 0.05e-3

    def validate(self) -> None:
        for name, value in (
            ("normal_gain_tu_per_m3", self.normal_gain_tu_per_m3),
            ("tangent_y_gain_tu_per_m3", self.tangent_y_gain_tu_per_m3),
            ("tangent_z_gain_tu_per_m3", self.tangent_z_gain_tu_per_m3),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0.")
        if not math.isfinite(self.activation_start_m) or self.activation_start_m < 0.0:
            raise ValueError("activation_start_m must be finite and >= 0.")
        if not math.isfinite(self.activation_full_m) or self.activation_full_m <= self.activation_start_m:
            raise ValueError("activation_full_m must be finite and greater than activation_start_m.")


@dataclass(frozen=True)
class EstimatorResult:
    normal_compression_m: np.ndarray
    shear_displacement_m: np.ndarray
    contact_activation_weight: np.ndarray
    vertex_deformation_volume_contribution_m3: np.ndarray
    raw_normal_deformation_volume_m3: np.ndarray
    raw_shear_deformation_volume_m3: np.ndarray
    normal_deformation_volume_m3: np.ndarray
    shear_deformation_volume_m3: np.ndarray
    baseline_normal_deformation_volume_m3: float
    baseline_shear_deformation_volume_m3: np.ndarray
    force_pad_local_tu: np.ndarray
    tactile_force_channels_tu: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V5 new 7g-a static, area-weighted deformation-based tactile force estimator. "
            "It reads only the frozen 7f displacement, vertex-area, and front-mask contract; "
            "it does not use contact geometry, penetration, native gradients, damping, or pressure."
        )
    )
    parser.add_argument("--contract_dir", required=True)
    parser.add_argument(
        "--displacement_path",
        default="",
        help=(
            "Optional [T,N,3] deformation-contract history. Defaults to the contract's "
            "surface_displacement_pad_local.npy snapshot."
        ),
    )
    parser.add_argument(
        "--baseline_displacement_path",
        default="",
        help="Optional no-contact deformation field [N,3] or [B,N,3] used only to calculate Q0.",
    )
    parser.add_argument(
        "--baseline_frame_count",
        type=int,
        default=0,
        help="Use the first B frames of the input sequence as Q0. Mutually exclusive with baseline_displacement_path.",
    )
    parser.add_argument("--commanded_indentation_path", default="", help="Optional T-vector for trend validation only.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--normal_gain_tu_per_m3", type=float, default=1.0e9)
    parser.add_argument("--tangent_y_gain_tu_per_m3", type=float, default=1.0e9)
    parser.add_argument("--tangent_z_gain_tu_per_m3", type=float, default=1.0e9)
    parser.add_argument("--activation_start_mm", type=float, default=0.01)
    parser.add_argument("--activation_full_mm", type=float, default=0.05)
    parser.add_argument("--normal_only_validation", action="store_true")
    parser.add_argument("--accept_min_loading_correlation", type=float, default=0.90)
    parser.add_argument("--accept_max_release_peak_ratio", type=float, default=0.05)
    parser.add_argument("--accept_max_normal_run_shear_ratio", type=float, default=0.25)
    parser.add_argument("--fail_on_verdict_fail", action="store_true")
    return parser


def _validate_cli_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if int(args.baseline_frame_count) < 0:
        parser.error("--baseline_frame_count must be >= 0.")
    if str(args.baseline_displacement_path).strip() and int(args.baseline_frame_count) > 0:
        parser.error("--baseline_displacement_path and --baseline_frame_count are mutually exclusive.")
    if float(args.activation_start_mm) < 0.0:
        parser.error("--activation_start_mm must be >= 0.")
    if float(args.activation_full_mm) <= float(args.activation_start_mm):
        parser.error("--activation_full_mm must be greater than --activation_start_mm.")
    if not (-1.0 <= float(args.accept_min_loading_correlation) <= 1.0):
        parser.error("--accept_min_loading_correlation must be in [-1,1].")
    for name in ("accept_max_release_peak_ratio", "accept_max_normal_run_shear_ratio"):
        if not (0.0 <= float(getattr(args, name)) <= 1.0):
            parser.error(f"--{name} must be in [0,1].")


def _as_displacement_history(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 3:
        array = array.reshape(1, *array.shape)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape [N,3] or [T,N,3], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _validate_spatial_contract(
    displacement_history: np.ndarray,
    vertex_area_m2: np.ndarray,
    front_surface_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertex_count = int(displacement_history.shape[1])
    area = np.asarray(vertex_area_m2, dtype=np.float64).reshape(-1)
    mask = np.asarray(front_surface_mask, dtype=bool).reshape(-1)
    if area.shape != (vertex_count,) or mask.shape != (vertex_count,):
        raise ValueError(
            f"Area/mask must match displacement vertex count {vertex_count}, got {area.shape} and {mask.shape}."
        )
    if not np.all(np.isfinite(area)) or np.any(area < 0.0):
        raise ValueError("vertex_area must be finite and non-negative.")
    if not np.any(mask):
        raise ValueError("front_surface_mask cannot be empty.")
    if np.any(area[~mask] != 0.0):
        raise ValueError("vertex_area must be zero outside front_surface_mask.")
    if np.any(area[mask] <= 0.0):
        raise ValueError("Every front-surface vertex must have positive rest area.")
    return area, mask


def smoothstep_contact_activation(
    normal_compression_m: np.ndarray,
    activation_start_m: float,
    activation_full_m: float,
) -> np.ndarray:
    compression = np.asarray(normal_compression_m, dtype=np.float64)
    scaled = np.clip(
        (compression - float(activation_start_m)) / (float(activation_full_m) - float(activation_start_m)),
        0.0,
        1.0,
    )
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _raw_deformation_volumes(
    displacement_history: np.ndarray,
    vertex_area_m2: np.ndarray,
    front_surface_mask: np.ndarray,
    config: EstimatorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normal_compression = np.clip(-displacement_history[..., 0], 0.0, None)
    shear_displacement = displacement_history[..., 1:3]
    activation = smoothstep_contact_activation(
        normal_compression,
        config.activation_start_m,
        config.activation_full_m,
    )
    active_area = vertex_area_m2 * front_surface_mask.astype(np.float64)
    normal_vertex_volume = normal_compression * active_area.reshape(1, -1)
    shear_vertex_volume = (
        shear_displacement
        * activation[..., None]
        * active_area.reshape(1, -1, 1)
    )
    raw_normal_volume = np.sum(normal_vertex_volume, axis=1, dtype=np.float64)
    raw_shear_volume = np.sum(shear_vertex_volume, axis=1, dtype=np.float64)
    vertex_volume = np.concatenate((normal_vertex_volume[..., None], shear_vertex_volume), axis=2)
    return normal_compression, shear_displacement, activation, vertex_volume, np.column_stack(
        (raw_normal_volume, raw_shear_volume)
    )


def estimate_deformation_force(
    displacement: np.ndarray,
    vertex_area_m2: np.ndarray,
    front_surface_mask: np.ndarray,
    config: EstimatorConfig,
    *,
    baseline_displacement: np.ndarray | None = None,
    baseline_frame_count: int = 0,
) -> EstimatorResult:
    config.validate()
    history = _as_displacement_history(displacement, "displacement")
    area, mask = _validate_spatial_contract(history, vertex_area_m2, front_surface_mask)
    if int(baseline_frame_count) < 0 or int(baseline_frame_count) > history.shape[0]:
        raise ValueError(f"baseline_frame_count must be in [0,{history.shape[0]}].")
    if baseline_displacement is not None and int(baseline_frame_count) > 0:
        raise ValueError("baseline_displacement and baseline_frame_count are mutually exclusive.")

    normal, shear, activation, vertex_volume, raw_volume = _raw_deformation_volumes(history, area, mask, config)
    raw_normal = raw_volume[:, 0]
    raw_shear = raw_volume[:, 1:3]

    if baseline_displacement is not None:
        baseline_history = _as_displacement_history(baseline_displacement, "baseline_displacement")
        if baseline_history.shape[1:] != history.shape[1:]:
            raise ValueError("baseline_displacement vertex shape does not match displacement.")
        _, _, _, _, baseline_raw = _raw_deformation_volumes(baseline_history, area, mask, config)
        baseline_normal = float(np.mean(baseline_raw[:, 0]))
        baseline_shear = np.mean(baseline_raw[:, 1:3], axis=0)
    elif int(baseline_frame_count) > 0:
        baseline_normal = float(np.mean(raw_normal[: int(baseline_frame_count)]))
        baseline_shear = np.mean(raw_shear[: int(baseline_frame_count)], axis=0)
    else:
        baseline_normal = 0.0
        baseline_shear = np.zeros(2, dtype=np.float64)

    normal_volume = np.clip(raw_normal - baseline_normal, 0.0, None)
    shear_volume = raw_shear - baseline_shear.reshape(1, 2)
    normal_score = config.normal_gain_tu_per_m3 * normal_volume
    tangent_y_score = config.tangent_y_gain_tu_per_m3 * shear_volume[:, 0]
    tangent_z_score = config.tangent_z_gain_tu_per_m3 * shear_volume[:, 1]

    force_pad = np.column_stack((-normal_score, tangent_y_score, tangent_z_score))
    tactile_channels = np.column_stack((force_pad[:, 1], -force_pad[:, 2], -force_pad[:, 0]))
    if not np.all(np.isfinite(force_pad)) or not np.all(np.isfinite(tactile_channels)):
        raise ValueError("Estimator produced NaN or infinite output.")

    return EstimatorResult(
        normal_compression_m=normal,
        shear_displacement_m=shear,
        contact_activation_weight=activation,
        vertex_deformation_volume_contribution_m3=vertex_volume,
        raw_normal_deformation_volume_m3=raw_normal,
        raw_shear_deformation_volume_m3=raw_shear,
        normal_deformation_volume_m3=normal_volume,
        shear_deformation_volume_m3=shear_volume,
        baseline_normal_deformation_volume_m3=baseline_normal,
        baseline_shear_deformation_volume_m3=np.asarray(baseline_shear, dtype=np.float64),
        force_pad_local_tu=force_pad,
        tactile_force_channels_tu=tactile_channels,
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
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _validate_7f_metadata(metadata: dict[str, object]) -> dict[str, bool]:
    allowed = metadata.get("allowed_7g_inputs")
    return {
        "7f_coordinate_frame_is_pad_local": metadata.get("coordinate_frame") == "pad_local",
        "7f_normal_axis_is_positive_x_outward": metadata.get("normal_axis") == "+X_outward",
        "7f_deformation_is_current_minus_rest": metadata.get("deformation_definition") == "Xt-X0",
        "7f_has_no_force_source": metadata.get("force_source") == "none",
        "7f_allows_frozen_estimator_inputs": isinstance(allowed, list)
        and {
            "surface_displacement_pad_local.npy",
            "vertex_area.npy",
            "front_surface_mask.npy",
        }.issubset(set(str(value) for value in allowed)),
    }


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = np.argsort(np.argsort(np.asarray(first, dtype=np.float64), kind="stable"), kind="stable")
    second_rank = np.argsort(np.argsort(np.asarray(second, dtype=np.float64), kind="stable"), kind="stable")
    if first_rank.size < 2 or float(np.std(first_rank)) <= EPS or float(np.std(second_rank)) <= EPS:
        return 0.0
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _loading_diagnostics(commanded_mm: np.ndarray, normal_score: np.ndarray) -> dict[str, object]:
    command = np.asarray(commanded_mm, dtype=np.float64).reshape(-1)
    score = np.asarray(normal_score, dtype=np.float64).reshape(-1)
    if command.shape != score.shape:
        raise ValueError(f"Commanded indentation shape {command.shape} does not match force history {score.shape}.")
    maximum = float(np.max(command))
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, command.size + 1):
        if index == command.size or not np.isclose(command[index], command[start], rtol=0.0, atol=1.0e-9):
            runs.append((start, index))
            start = index
    maximum_runs = [run for run in runs if np.isclose(command[run[0]], maximum, rtol=0.0, atol=1.0e-9)]
    loading_end = int(maximum_runs[0][1] - 1) if maximum_runs else int(np.argmax(command))
    loading_command = command[: loading_end + 1]
    loading_score = score[: loading_end + 1]
    if loading_command.size < 2 or float(np.std(loading_command)) <= EPS or float(np.std(loading_score)) <= EPS:
        raw_pearson = 0.0
    else:
        raw_pearson = float(np.corrcoef(loading_command, loading_score)[0, 1])

    settled_runs = [(begin, end) for begin, end in runs if end - begin >= 2 and end - 1 <= loading_end]
    settled_command = np.asarray([command[begin] for begin, _ in settled_runs], dtype=np.float64)
    settled_score = np.asarray(
        [float(np.mean(score[max(begin, end - 3) : end])) for begin, end in settled_runs],
        dtype=np.float64,
    )
    settled_rank_correlation = _rank_correlation(settled_command, settled_score)
    settled_monotonic = bool(
        settled_score.size >= 2
        and np.all(np.diff(settled_command) > 0.0)
        and np.all(np.diff(settled_score) >= -EPS)
    )
    return {
        "loading_end_frame": loading_end,
        "raw_ramp_pearson_correlation": raw_pearson,
        "settled_hold_commanded_indentation_mm": [float(value) for value in settled_command],
        "settled_hold_normal_tactile_unit": [float(value) for value in settled_score],
        "settled_hold_rank_correlation": settled_rank_correlation,
        "settled_hold_response_monotonic": settled_monotonic,
    }


def validate_estimator_result(
    result: EstimatorResult,
    *,
    source_checks: dict[str, bool],
    commanded_indentation_mm: np.ndarray | None,
    normal_only_validation: bool,
    accept_min_loading_correlation: float,
    accept_max_release_peak_ratio: float,
    accept_max_normal_run_shear_ratio: float,
) -> dict[str, object]:
    force_pad = result.force_pad_local_tu
    tactile = result.tactile_force_channels_tu
    normal_score = tactile[:, 2]
    tangent_score = tactile[:, :2]
    peak_normal = float(np.max(normal_score))
    final_normal = float(normal_score[-1])
    release_ratio = final_normal / max(peak_normal, EPS)
    peak_tangent = float(np.max(np.linalg.norm(tangent_score, axis=1)))
    tangent_normal_ratio = peak_tangent / max(peak_normal, EPS)
    transform_expected = np.column_stack((force_pad[:, 1], -force_pad[:, 2], -force_pad[:, 0]))

    checks: dict[str, bool] = {
        **source_checks,
        "outputs_are_finite": bool(np.all(np.isfinite(force_pad)) and np.all(np.isfinite(tactile))),
        "normal_score_is_nonnegative": bool(np.all(normal_score >= -EPS)),
        "pad_normal_vector_points_inward": bool(np.all(force_pad[:, 0] <= EPS)),
        "tactile_transform_is_exact": bool(np.allclose(tactile, transform_expected, rtol=0.0, atol=1.0e-12)),
    }
    loading_diagnostics: dict[str, object] | None = None
    if commanded_indentation_mm is not None:
        loading_diagnostics = _loading_diagnostics(commanded_indentation_mm, normal_score)
        checks["normal_score_tracks_settled_loading"] = bool(
            loading_diagnostics["settled_hold_response_monotonic"]
        ) and float(loading_diagnostics["settled_hold_rank_correlation"]) >= float(
            accept_min_loading_correlation
        )
        checks["release_returns_near_zero"] = release_ratio <= float(accept_max_release_peak_ratio)
    if normal_only_validation:
        checks["normal_run_net_shear_is_bounded"] = tangent_normal_ratio <= float(
            accept_max_normal_run_shear_ratio
        )

    observed = {
        "frame_count": int(force_pad.shape[0]),
        "peak_normal_tactile_unit": peak_normal,
        "final_normal_tactile_unit": final_normal,
        "release_to_peak_ratio": release_ratio,
        "peak_tangent_resultant_tactile_unit": peak_tangent,
        "peak_tangent_to_normal_ratio": tangent_normal_ratio,
        "loading_diagnostics": loading_diagnostics,
        "baseline_normal_deformation_volume_m3": result.baseline_normal_deformation_volume_m3,
        "baseline_shear_deformation_volume_m3": [
            float(value) for value in result.baseline_shear_deformation_volume_m3
        ],
    }
    thresholds = {
        "min_settled_loading_rank_correlation": float(accept_min_loading_correlation),
        "max_release_peak_ratio": float(accept_max_release_peak_ratio),
        "max_normal_run_shear_ratio": float(accept_max_normal_run_shear_ratio),
    }
    return {
        "deformation_based_force_estimator_passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": thresholds,
        "observed": observed,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    contract_dir = Path(args.contract_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    displacement_path = (
        Path(args.displacement_path).expanduser().resolve()
        if str(args.displacement_path).strip()
        else contract_dir / "surface_displacement_pad_local.npy"
    )
    displacement = _load_array(displacement_path, "surface displacement")
    vertex_area = _load_array(contract_dir / "vertex_area.npy", "vertex area")
    front_mask = _load_array(contract_dir / "front_surface_mask.npy", "front surface mask")
    contract_metadata = _load_json(contract_dir / "metadata.json", "7f metadata")
    contract_verdict = _load_json(contract_dir / "verdict.json", "7f verdict")
    source_checks = _validate_7f_metadata(contract_metadata)
    source_checks["source_7f_contract_passed"] = bool(contract_verdict.get("deformation_contract_passed", False))

    baseline_displacement = None
    if str(args.baseline_displacement_path).strip():
        baseline_displacement = _load_array(
            Path(args.baseline_displacement_path).expanduser().resolve(), "baseline displacement"
        )
    config = EstimatorConfig(
        normal_gain_tu_per_m3=float(args.normal_gain_tu_per_m3),
        tangent_y_gain_tu_per_m3=float(args.tangent_y_gain_tu_per_m3),
        tangent_z_gain_tu_per_m3=float(args.tangent_z_gain_tu_per_m3),
        activation_start_m=float(args.activation_start_mm) * 1.0e-3,
        activation_full_m=float(args.activation_full_mm) * 1.0e-3,
    )
    result = estimate_deformation_force(
        displacement,
        vertex_area,
        front_mask,
        config,
        baseline_displacement=baseline_displacement,
        baseline_frame_count=int(args.baseline_frame_count),
    )

    commanded_indentation = None
    if str(args.commanded_indentation_path).strip():
        commanded_indentation = _load_array(
            Path(args.commanded_indentation_path).expanduser().resolve(), "commanded indentation"
        )
    verdict = validate_estimator_result(
        result,
        source_checks=source_checks,
        commanded_indentation_mm=commanded_indentation,
        normal_only_validation=bool(args.normal_only_validation),
        accept_min_loading_correlation=float(args.accept_min_loading_correlation),
        accept_max_release_peak_ratio=float(args.accept_max_release_peak_ratio),
        accept_max_normal_run_shear_ratio=float(args.accept_max_normal_run_shear_ratio),
    )

    arrays = {
        "normal_compression.npy": result.normal_compression_m.astype(np.float32),
        "shear_displacement.npy": result.shear_displacement_m.astype(np.float32),
        "contact_activation_weight.npy": result.contact_activation_weight.astype(np.float32),
        "vertex_deformation_volume_contribution.npy": result.vertex_deformation_volume_contribution_m3,
        "raw_normal_deformation_volume.npy": result.raw_normal_deformation_volume_m3,
        "raw_shear_deformation_volume.npy": result.raw_shear_deformation_volume_m3,
        "normal_deformation_volume.npy": result.normal_deformation_volume_m3,
        "shear_deformation_volume.npy": result.shear_deformation_volume_m3,
        "force_pad_local.npy": result.force_pad_local_tu,
        "tactile_force_channels.npy": result.tactile_force_channels_tu,
    }
    saved_paths: list[Path] = []
    for name, value in arrays.items():
        path = output_dir / name
        np.save(path, value)
        saved_paths.append(path)

    metadata = {
        "estimator_version": ESTIMATOR_VERSION,
        "estimator_role": "area_weighted_reduced_order_deformation_based_tactile_force_estimator",
        "estimator_type": "static_linear_F_equals_KQ",
        "input_contract_dir": str(contract_dir),
        "input_displacement_path": str(displacement_path),
        "allowed_estimator_inputs": [
            "surface_displacement_pad_local.npy",
            "vertex_area.npy",
            "front_surface_mask.npy",
        ],
        "coordinate_frame": "pad_local",
        "pad_axes": {"+X": "outward_normal", "+Y": "tangent_1", "+Z": "tangent_2"},
        "force_direction": "object_on_sensor",
        "force_pad_local_definition": "[-Fn, Ft_pad_y, Ft_pad_z]",
        "tactile_frame_axes": {"+X": "pad_+Y", "+Y": "pad_-Z", "+Z": "pad_-X"},
        "tactile_channel_definition": "[force_pad_y, -force_pad_z, -force_pad_x]",
        "normal_volume_definition": "sum_i A_i*max(-u_x_i,0)-Qn0; clipped nonnegative after sum",
        "shear_volume_definition": "sum_i A_i*w_i*[u_y_i,u_z_i]-Qt0",
        "vertex_deformation_volume_contribution_definition": (
            "raw_pre_baseline_[A_i*d_n_i, A_i*w_i*u_y_i, A_i*w_i*u_z_i]"
        ),
        "activation_definition": "smoothstep(clip((d_n-d0)/(d1-d0),0,1))",
        "activation_start_mm": float(args.activation_start_mm),
        "activation_full_mm": float(args.activation_full_mm),
        "normal_gain_tu_per_m3": config.normal_gain_tu_per_m3,
        "tangent_y_gain_tu_per_m3": config.tangent_y_gain_tu_per_m3,
        "tangent_z_gain_tu_per_m3": config.tangent_z_gain_tu_per_m3,
        "output_unit": "relative_tactile_unit",
        "calibrated_newton": False,
        "damping_enabled": False,
        "force_reconstruction_claimed": False,
        "contact_geometry_used": False,
        "penetration_used": False,
        "native_uipc_gradient_used": False,
        "proxy_force_used": False,
        "pressure_output_enabled": False,
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "source_7f_metadata.json", contract_metadata)
    _write_json(output_dir / "verdict.json", verdict)
    _write_json(output_dir / "summary.json", verdict)
    manifest = {
        "estimator_version": ESTIMATOR_VERSION,
        "arrays": {
            path.name: {
                "shape": list(np.load(path, mmap_mode="r", allow_pickle=False).shape),
                "dtype": str(np.load(path, mmap_mode="r", allow_pickle=False).dtype),
                "sha256": _file_sha256(path),
            }
            for path in saved_paths
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return verdict


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(args, parser)
    verdict = run_cli(args)
    print(json.dumps(verdict, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if bool(args.fail_on_verdict_fail) and not bool(verdict["deformation_based_force_estimator_passed"]):
        raise RuntimeError(f"7g deformation-based estimator verdict failed: {verdict}")


if __name__ == "__main__":
    main()
