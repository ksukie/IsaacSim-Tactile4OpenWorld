from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import OpenWorldTactile_v5_new_7g_deformation_force_estimator as frozen_7g


EPS = 1.0e-12
SCRIPT_VERSION = "v5_new_7g_shear_z_frozen_force_validation_v1"


@dataclass(frozen=True)
class CaseResult:
    name: str
    direction_sign: float
    phases: tuple[str, ...]
    command_mm: np.ndarray
    actual_mm: np.ndarray
    force_pad_tu: np.ndarray
    tactile_tu: np.ndarray
    shear_displacement_m: np.ndarray
    metrics: dict[str, object]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate membrane-local +Z/-Z UIPC deformation with the restored frozen 7g "
            "area-weighted estimator. The output is relative tactile value in TU, never Newton."
        )
    )
    parser.add_argument("--contract_dir", required=True)
    parser.add_argument("--positive_probe_dir", required=True)
    parser.add_argument("--negative_probe_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_frame_count", type=int, default=2)
    parser.add_argument("--baseline_tail_frames", type=int, default=5)
    parser.add_argument("--active_command_ratio", type=float, default=0.10)
    parser.add_argument("--accept_min_active_response_ratio", type=float, default=0.95)
    parser.add_argument("--accept_min_direction_cosine", type=float, default=0.90)
    parser.add_argument("--accept_min_sign_flip_rate", type=float, default=0.95)
    parser.add_argument("--accept_max_normal_pollution_ratio", type=float, default=0.20)
    parser.add_argument("--accept_max_orthogonal_crosstalk_ratio", type=float, default=0.10)
    parser.add_argument("--accept_max_release_peak_ratio", type=float, default=0.02)
    parser.add_argument("--fail_on_verdict_fail", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if int(args.baseline_frame_count) <= 0 or int(args.baseline_tail_frames) <= 0:
        parser.error("baseline frame counts must be > 0")
    for name in (
        "active_command_ratio",
        "accept_min_active_response_ratio",
        "accept_min_direction_cosine",
        "accept_min_sign_flip_rate",
        "accept_max_normal_pollution_ratio",
        "accept_max_orthogonal_crosstalk_ratio",
        "accept_max_release_peak_ratio",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name} must be in [0,1]")


def _load_npy(directory: Path, name: str) -> np.ndarray:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def _load_json(directory: Path, name: str) -> object:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def membrane_vectors_to_pad_local(
    vectors_membrane_local: np.ndarray, membrane_from_pad_rotation: np.ndarray
) -> np.ndarray:
    vectors = np.asarray(vectors_membrane_local, dtype=np.float64)
    rotation = np.asarray(membrane_from_pad_rotation, dtype=np.float64).reshape(3, 3)
    # Column-vector contract: v_M = R_M<-P v_P. For row-vector histories,
    # v_P(row) = v_M(row) R_M<-P.
    return vectors @ rotation


def _phase_indices(phases: tuple[str, ...], exact: str) -> np.ndarray:
    return np.asarray([i for i, phase in enumerate(phases) if phase == exact], dtype=np.int64)


def _prepare_pad_input(
    probe_dir: Path,
    input_dir: Path,
    *,
    expected_direction: str,
) -> tuple[Path, tuple[str, ...], np.ndarray, np.ndarray, dict[str, object]]:
    metadata = _load_json(probe_dir, "metadata.json")
    verdict = _load_json(probe_dir, "verdict.json")
    frame_contract = _load_json(probe_dir, "membrane_frame_contract.json")
    phases_value = _load_json(probe_dir, "phase_history.json")
    if not all(isinstance(value, dict) for value in (metadata, verdict, frame_contract)):
        raise ValueError("probe metadata, verdict, and frame contract must be JSON objects")
    if not isinstance(phases_value, list):
        raise ValueError("phase_history.json must contain a list")
    checks = {
        "source_deformation_probe_passed": bool(
            verdict.get("lateral_shear_deformation_probe_passed", False)
        ),
        "membrane_frame_contract_passed": bool(
            frame_contract.get("membrane_frame_contract_passed", False)
        ),
        "coordinate_frame_is_membrane_local": metadata.get("coordinate_frame")
        == "membrane_local",
        "lateral_axis_is_membrane_local_z": metadata.get("lateral_axis_membrane_local")
        == "z",
        "lateral_direction_matches": metadata.get("lateral_direction")
        == expected_direction,
    }
    if not all(checks.values()):
        raise ValueError(f"probe contract failed: {checks}")

    displacement_membrane = np.asarray(
        _load_npy(probe_dir, "membrane_displacement_local.npy"), dtype=np.float64
    )
    rotation = np.asarray(
        _load_npy(probe_dir, "membrane_from_pad_rotation.npy"), dtype=np.float64
    ).reshape(3, 3)
    command = np.asarray(_load_npy(probe_dir, "commanded_lateral_mm.npy"), dtype=np.float64).reshape(-1)
    actual = np.asarray(_load_npy(probe_dir, "actual_lateral_mm.npy"), dtype=np.float64).reshape(-1)
    phases = tuple(str(value) for value in phases_value)
    if displacement_membrane.ndim != 3 or displacement_membrane.shape[2] != 3:
        raise ValueError("membrane displacement must have shape [T,N,3]")
    if command.shape != (len(phases),) or actual.shape != (len(phases),):
        raise ValueError("command/actual histories do not match phase history")
    if displacement_membrane.shape[0] != len(phases):
        raise ValueError("displacement and phase frame counts differ")
    if not all(np.all(np.isfinite(value)) for value in (displacement_membrane, rotation, command, actual)):
        raise ValueError("probe input contains NaN or Inf")

    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    determinant = float(np.linalg.det(rotation))
    displacement_pad = membrane_vectors_to_pad_local(displacement_membrane, rotation)
    round_trip = displacement_pad @ rotation.T
    round_trip_error = float(
        np.linalg.norm(round_trip - displacement_membrane)
        / max(np.linalg.norm(displacement_membrane), EPS)
    )
    if orthogonality_error >= 1.0e-6 or abs(determinant - 1.0) >= 1.0e-6 or round_trip_error >= 1.0e-8:
        raise ValueError("membrane-to-pad vector rotation audit failed")

    input_dir.mkdir(parents=True, exist_ok=True)
    displacement_path = input_dir / "surface_displacement_pad_local.npy"
    np.save(displacement_path, displacement_pad.astype(np.float32))
    conversion = {
        "source_probe_dir": str(probe_dir),
        "source_coordinate_frame": "membrane_local",
        "output_coordinate_frame": "pad_local",
        "vector_conversion": "v_pad_row = v_membrane_row @ R_membrane_from_pad",
        "orthogonality_error": orthogonality_error,
        "determinant": determinant,
        "vector_round_trip_relative_error": round_trip_error,
        "source_sha256": _sha256(probe_dir / "membrane_displacement_local.npy"),
        "output_sha256": _sha256(displacement_path),
    }
    _write_json(input_dir / "conversion_metadata.json", conversion)
    return displacement_path, phases, command, actual, conversion


def _run_frozen_estimator(
    contract_dir: Path,
    displacement_path: Path,
    force_dir: Path,
    baseline_frame_count: int,
) -> dict[str, object]:
    frozen_args = frozen_7g.build_parser().parse_args(
        [
            "--contract_dir",
            str(contract_dir),
            "--displacement_path",
            str(displacement_path),
            "--baseline_frame_count",
            str(int(baseline_frame_count)),
            "--output_dir",
            str(force_dir),
        ]
    )
    return frozen_7g.run_cli(frozen_args)


def _case_metrics(
    name: str,
    direction_sign: float,
    phases: tuple[str, ...],
    command: np.ndarray,
    actual: np.ndarray,
    force_pad: np.ndarray,
    tactile: np.ndarray,
    *,
    baseline_tail_frames: int,
    active_command_ratio: float,
) -> dict[str, object]:
    word = "positive" if direction_sign > 0.0 else "negative"
    ramp = _phase_indices(phases, f"shear_{word}_ramp")
    hold = _phase_indices(phases, f"shear_{word}_hold")
    shear_indices = np.concatenate((ramp, hold))
    if shear_indices.size == 0 or hold.size == 0:
        raise ValueError(f"{name}: required shear phases are missing")
    shear_start = int(np.min(shear_indices))
    baseline = np.asarray(
        [
            i
            for i, phase in enumerate(phases[:shear_start])
            if phase.startswith("load_hold_") and abs(float(command[i])) <= EPS
        ],
        dtype=np.int64,
    )
    if baseline.size == 0:
        raise ValueError(f"{name}: pre-shear normal hold baseline is missing")
    baseline = baseline[-min(int(baseline_tail_frames), int(baseline.size)) :]
    baseline_force = np.mean(force_pad[baseline], axis=0)
    delta = force_pad - baseline_force.reshape(1, 3)
    baseline_noise = np.linalg.norm(delta[baseline, 1:3], axis=1)
    noise = max(
        float(np.percentile(baseline_noise, 99.0)),
        3.0 * float(np.sqrt(np.mean(np.square(baseline_noise)))),
        1.0e-9,
    )
    amplitude = float(np.max(np.abs(command[shear_indices])))
    active = shear_indices[
        np.abs(command[shear_indices]) >= float(active_command_ratio) * max(amplitude, EPS)
    ]
    tangent = delta[active, 1:3]
    tangent_magnitude = np.linalg.norm(tangent, axis=1)
    signed_z = float(direction_sign) * tangent[:, 1]
    cosine = signed_z / np.maximum(tangent_magnitude, EPS)
    normal_delta = delta[active, 0]
    expected_tactile_fy_sign = -float(direction_sign)
    tactile_fy_delta = tactile[active, 1] - float(np.mean(tactile[baseline, 1]))
    release = _phase_indices(phases, f"shear_zero_after_{word}_hold")
    tail = np.arange(max(0, len(phases) - 3), len(phases), dtype=np.int64)
    peak_tangent = float(np.max(np.linalg.norm(force_pad[:, 1:3], axis=1)))
    release_tail_peak = float(np.max(np.linalg.norm(force_pad[tail, 1:3], axis=1)))
    return {
        "name": name,
        "active_frame_count": int(active.size),
        "response_active": tangent_magnitude > noise,
        "direction_cosine": cosine,
        "pad_z_sign_correct": signed_z > 0.0,
        "tactile_fy_sign_correct": expected_tactile_fy_sign * tactile_fy_delta > 0.0,
        "normal_delta": normal_delta,
        "orthogonal_y_delta": tangent[:, 0],
        "requested_z_delta": tangent[:, 1],
        "tangent_delta": tangent,
        "force_noise_tu": noise,
        "command_amplitude_mm": amplitude,
        "settled_actual_lateral_mm": float(np.mean(actual[hold])),
        "peak_tangent_tu": peak_tangent,
        "release_tail_peak_tu": release_tail_peak,
        "release_to_peak_ratio": release_tail_peak / max(peak_tangent, EPS),
        "return_tangent_mean_tu": (
            float(np.mean(np.linalg.norm(force_pad[release, 1:3], axis=1)))
            if release.size
            else float("inf")
        ),
        "mean_tactile_fy_hold_tu": float(np.mean(tactile[hold, 1])),
    }


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    contract_dir = Path(args.contract_dir).expanduser().resolve()
    positive_probe = Path(args.positive_probe_dir).expanduser().resolve()
    negative_probe = Path(args.negative_probe_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator_path = Path(frozen_7g.__file__).resolve()
    estimator_hash_before = _sha256(estimator_path)

    prepared: list[tuple[str, float, Path, tuple[str, ...], np.ndarray, np.ndarray, dict[str, object]]] = []
    for name, sign, probe, direction in (
        ("positive_z", 1.0, positive_probe, "positive"),
        ("negative_z", -1.0, negative_probe, "negative"),
    ):
        case_root = output_dir / name
        displacement_path, phases, command, actual, conversion = _prepare_pad_input(
            probe, case_root / "pad_local_input", expected_direction=direction
        )
        prepared.append((name, sign, displacement_path, phases, command, actual, conversion))

    cases: list[CaseResult] = []
    estimator_verdicts: dict[str, object] = {}
    conversions: dict[str, object] = {}
    for name, sign, displacement_path, phases, command, actual, conversion in prepared:
        force_dir = output_dir / name / "frozen_7g_force"
        estimator_verdict = _run_frozen_estimator(
            contract_dir, displacement_path, force_dir, int(args.baseline_frame_count)
        )
        estimator_verdicts[name] = estimator_verdict
        conversions[name] = conversion
        force_pad = np.asarray(_load_npy(force_dir, "force_pad_local.npy"), dtype=np.float64)
        tactile = np.asarray(_load_npy(force_dir, "tactile_force_channels.npy"), dtype=np.float64)
        shear = np.asarray(_load_npy(force_dir, "shear_displacement.npy"), dtype=np.float32)
        if force_pad.shape != (len(phases), 3) or tactile.shape != (len(phases), 3):
            raise ValueError(f"{name}: frozen force output frame count mismatch")
        metrics = _case_metrics(
            name,
            sign,
            phases,
            command,
            actual,
            force_pad,
            tactile,
            baseline_tail_frames=int(args.baseline_tail_frames),
            active_command_ratio=float(args.active_command_ratio),
        )
        cases.append(
            CaseResult(name, sign, phases, command, actual, force_pad, tactile, shear, metrics)
        )

    response = np.concatenate([np.asarray(case.metrics["response_active"], dtype=bool) for case in cases])
    cosine = np.concatenate([np.asarray(case.metrics["direction_cosine"], dtype=np.float64) for case in cases])
    pad_sign = np.concatenate([np.asarray(case.metrics["pad_z_sign_correct"], dtype=bool) for case in cases])
    tactile_sign = np.concatenate(
        [np.asarray(case.metrics["tactile_fy_sign_correct"], dtype=bool) for case in cases]
    )
    normal = np.concatenate([np.asarray(case.metrics["normal_delta"], dtype=np.float64) for case in cases])
    orthogonal = np.concatenate(
        [np.asarray(case.metrics["orthogonal_y_delta"], dtype=np.float64) for case in cases]
    )
    requested = np.concatenate(
        [np.asarray(case.metrics["requested_z_delta"], dtype=np.float64) for case in cases]
    )
    tangent = np.concatenate([np.asarray(case.metrics["tangent_delta"], dtype=np.float64) for case in cases])
    active_response_ratio = float(np.mean(response))
    direction_cosine = float(np.mean(cosine))
    sign_flip_rate = float(np.mean(pad_sign & tactile_sign))
    tangent_rms = float(np.sqrt(np.mean(np.sum(np.square(tangent), axis=1))))
    requested_rms = float(np.sqrt(np.mean(np.square(requested))))
    normal_pollution = float(np.sqrt(np.mean(np.square(normal))) / max(tangent_rms, EPS))
    orthogonal_crosstalk = float(
        np.sqrt(np.mean(np.square(orthogonal))) / max(requested_rms, EPS)
    )
    maximum_release_ratio = max(float(case.metrics["release_to_peak_ratio"]) for case in cases)
    estimator_hash_after = _sha256(estimator_path)

    checks = {
        "positive_source_estimator_passed": bool(
            estimator_verdicts["positive_z"].get("deformation_based_force_estimator_passed", False)
        ),
        "negative_source_estimator_passed": bool(
            estimator_verdicts["negative_z"].get("deformation_based_force_estimator_passed", False)
        ),
        "frozen_estimator_hash_unchanged": estimator_hash_before == estimator_hash_after,
        "active_response_ratio_above_threshold": active_response_ratio
        > float(args.accept_min_active_response_ratio),
        "direction_cosine_above_threshold": direction_cosine
        > float(args.accept_min_direction_cosine),
        "sign_flip_rate_above_threshold": sign_flip_rate
        > float(args.accept_min_sign_flip_rate),
        "normal_pollution_below_threshold": normal_pollution
        < float(args.accept_max_normal_pollution_ratio),
        "orthogonal_crosstalk_below_threshold": orthogonal_crosstalk
        < float(args.accept_max_orthogonal_crosstalk_ratio),
        "release_below_peak_ratio": maximum_release_ratio
        < float(args.accept_max_release_peak_ratio),
    }
    per_case = {
        case.name: {
            "active_frame_count": int(case.metrics["active_frame_count"]),
            "active_response_ratio": float(np.mean(case.metrics["response_active"])),
            "direction_cosine": float(np.mean(case.metrics["direction_cosine"])),
            "pad_z_sign_correct_rate": float(np.mean(case.metrics["pad_z_sign_correct"])),
            "tactile_fy_sign_correct_rate": float(
                np.mean(case.metrics["tactile_fy_sign_correct"])
            ),
            "normal_pollution_ratio": float(
                np.sqrt(np.mean(np.square(case.metrics["normal_delta"])))
                / max(
                    np.sqrt(np.mean(np.sum(np.square(case.metrics["tangent_delta"]), axis=1))),
                    EPS,
                )
            ),
            "orthogonal_y_over_z_crosstalk_ratio": float(
                np.sqrt(np.mean(np.square(case.metrics["orthogonal_y_delta"])))
                / max(np.sqrt(np.mean(np.square(case.metrics["requested_z_delta"]))), EPS)
            ),
            "command_amplitude_mm": float(case.metrics["command_amplitude_mm"]),
            "settled_actual_lateral_mm": float(case.metrics["settled_actual_lateral_mm"]),
            "peak_tangent_tu": float(case.metrics["peak_tangent_tu"]),
            "release_tail_peak_tu": float(case.metrics["release_tail_peak_tu"]),
            "release_to_peak_ratio": float(case.metrics["release_to_peak_ratio"]),
            "return_tangent_mean_tu": float(case.metrics["return_tangent_mean_tu"]),
            "mean_tactile_fy_hold_tu": float(case.metrics["mean_tactile_fy_hold_tu"]),
        }
        for case in cases
    }
    verdict = {
        "membrane_local_z_frozen_7g_force_validation_passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "active_response_ratio_strictly_greater_than": float(
                args.accept_min_active_response_ratio
            ),
            "direction_cosine_strictly_greater_than": float(args.accept_min_direction_cosine),
            "sign_flip_rate_strictly_greater_than": float(args.accept_min_sign_flip_rate),
            "normal_pollution_strictly_less_than": float(
                args.accept_max_normal_pollution_ratio
            ),
            "orthogonal_crosstalk_strictly_less_than": float(
                args.accept_max_orthogonal_crosstalk_ratio
            ),
            "release_to_peak_strictly_less_than": float(args.accept_max_release_peak_ratio),
        },
        "observed": {
            "active_response_ratio": active_response_ratio,
            "direction_cosine": direction_cosine,
            "sign_flip_rate": sign_flip_rate,
            "normal_pollution_ratio": normal_pollution,
            "orthogonal_y_over_z_crosstalk_ratio": orthogonal_crosstalk,
            "maximum_release_to_peak_ratio": maximum_release_ratio,
            "per_case": per_case,
        },
        "force_source": "frozen_v5_new_7g_deformation_based_force_estimator",
        "force_unit": "TU",
        "newton_calibrated": False,
    }
    metadata = {
        "script_version": SCRIPT_VERSION,
        "force_source": "uipc_membrane_surface_deformation_reduced_order",
        "force_unit": "TU",
        "newton_calibrated": False,
        "normal_axis": "membrane_local_x",
        "tactile_fx_axis": "membrane_local_y",
        "tactile_fy_axis": "negative_membrane_local_z",
        "damping_used": False,
        "tactile_channel_definition": "[force_pad_y, -force_pad_z, -force_pad_x]",
        "z_command_to_tactile_fy_sign": {"+Z": "negative", "-Z": "positive"},
        "frozen_estimator_path": str(estimator_path),
        "frozen_estimator_sha256_before": estimator_hash_before,
        "frozen_estimator_sha256_after": estimator_hash_after,
        "baseline_frame_count": int(args.baseline_frame_count),
        "conversions": conversions,
    }
    np.save(output_dir / "force_pad_local.npy", np.concatenate([case.force_pad_tu for case in cases]))
    np.save(
        output_dir / "tactile_force_channels.npy",
        np.concatenate([case.tactile_tu for case in cases]),
    )
    np.save(
        output_dir / "shear_displacement.npy",
        np.concatenate([case.shear_displacement_m for case in cases]),
    )
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "shear_z_response_metrics.json", verdict)
    _write_json(output_dir / "verdict.json", verdict)
    return verdict


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    verdict = run_cli(args)
    print(json.dumps(verdict, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if bool(args.fail_on_verdict_fail) and not bool(
        verdict["membrane_local_z_frozen_7g_force_validation_passed"]
    ):
        raise RuntimeError(f"membrane-local Z frozen-7g validation failed: {verdict}")


if __name__ == "__main__":
    main()
