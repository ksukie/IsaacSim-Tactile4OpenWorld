from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


EPS = 1.0e-12


@dataclass(frozen=True)
class ShearCase:
    name: str
    direction_sign: float
    phases: tuple[str, ...]
    commanded_lateral_mm: np.ndarray
    actual_lateral_mm: np.ndarray
    force_pad_local: np.ndarray
    tactile_force_channels: np.ndarray
    shear_displacement: np.ndarray
    lateral_axis: str
    source_probe_passed: bool
    source_estimator_passed: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate two independent physical +Y/-Y UIPC shear runs after the frozen 7g estimator. "
            "The four acceptance metrics are active response ratio, direction cosine, sign flip rate, "
            "and normal-channel contamination."
        )
    )
    parser.add_argument("--positive_probe_dir", required=True)
    parser.add_argument("--positive_force_dir", required=True)
    parser.add_argument("--negative_probe_dir", required=True)
    parser.add_argument("--negative_force_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_tail_frames", type=int, default=5)
    parser.add_argument("--active_command_ratio", type=float, default=0.10)
    parser.add_argument("--accept_min_active_response_ratio", type=float, default=0.95)
    parser.add_argument("--accept_min_direction_cosine", type=float, default=0.90)
    parser.add_argument("--accept_min_sign_flip_rate", type=float, default=0.95)
    parser.add_argument("--accept_max_normal_pollution_ratio", type=float, default=0.20)
    parser.add_argument("--fail_on_verdict_fail", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if int(args.baseline_tail_frames) <= 0:
        parser.error("--baseline_tail_frames must be > 0.")
    for name in (
        "active_command_ratio",
        "accept_min_active_response_ratio",
        "accept_min_direction_cosine",
        "accept_min_sign_flip_rate",
        "accept_max_normal_pollution_ratio",
    ):
        value = float(getattr(args, name))
        if not (0.0 <= value <= 1.0):
            parser.error(f"--{name} must be in [0,1].")


def _load_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _load_case(name: str, direction_sign: float, probe_dir: Path, force_dir: Path) -> ShearCase:
    phases_value = _load_json(probe_dir / "phase_history.json")
    probe_metadata = _load_json(probe_dir / "metadata.json")
    probe_verdict = _load_json(probe_dir / "verdict.json")
    force_verdict = _load_json(force_dir / "verdict.json")
    if not isinstance(phases_value, list):
        raise ValueError(f"{probe_dir}/phase_history.json must contain a list.")
    if not isinstance(probe_metadata, dict) or not isinstance(probe_verdict, dict):
        raise ValueError("Probe metadata/verdict must be JSON objects.")
    if not isinstance(force_verdict, dict):
        raise ValueError("Force verdict must be a JSON object.")
    phases = tuple(str(value) for value in phases_value)
    command = np.asarray(_load_array(probe_dir / "commanded_lateral_mm.npy"), dtype=np.float64).reshape(-1)
    actual = np.asarray(_load_array(probe_dir / "actual_lateral_mm.npy"), dtype=np.float64).reshape(-1)
    force_pad = np.asarray(_load_array(force_dir / "force_pad_local.npy"), dtype=np.float64)
    tactile = np.asarray(_load_array(force_dir / "tactile_force_channels.npy"), dtype=np.float64)
    shear = np.asarray(_load_array(force_dir / "shear_displacement.npy"), dtype=np.float32)
    frame_count = len(phases)
    if command.shape != (frame_count,) or actual.shape != (frame_count,):
        raise ValueError(f"{name}: command/actual histories do not match {frame_count} phase frames.")
    if force_pad.shape != (frame_count, 3) or tactile.shape != (frame_count, 3):
        raise ValueError(f"{name}: force histories must have shape {(frame_count, 3)}.")
    if shear.ndim != 3 or shear.shape[0] != frame_count or shear.shape[2] != 2:
        raise ValueError(f"{name}: shear displacement must have shape [T,N,2].")
    expected_direction = "positive" if direction_sign > 0.0 else "negative"
    if str(probe_metadata.get("lateral_direction")) != expected_direction:
        raise ValueError(f"{name}: metadata lateral_direction does not match requested case.")
    return ShearCase(
        name=name,
        direction_sign=float(direction_sign),
        phases=phases,
        commanded_lateral_mm=command,
        actual_lateral_mm=actual,
        force_pad_local=force_pad,
        tactile_force_channels=tactile,
        shear_displacement=shear,
        lateral_axis=str(probe_metadata.get("lateral_axis_pad_local", "")),
        source_probe_passed=bool(probe_verdict.get("lateral_shear_deformation_probe_passed", False)),
        source_estimator_passed=bool(force_verdict.get("deformation_based_force_estimator_passed", False)),
    )


def _indices(case: ShearCase, phase_name: str) -> np.ndarray:
    return np.asarray(
        [index for index, phase in enumerate(case.phases) if str(phase) == phase_name], dtype=np.int64
    )


def _baseline_indices(case: ShearCase, tail_frames: int) -> np.ndarray:
    shear_start = next(
        (index for index, phase in enumerate(case.phases) if str(phase).startswith("shear_")),
        len(case.phases),
    )
    candidates = np.asarray(
        [
            index
            for index, phase in enumerate(case.phases[:shear_start])
            if str(phase).startswith("load_hold_") and abs(float(case.commanded_lateral_mm[index])) <= EPS
        ],
        dtype=np.int64,
    )
    if candidates.size == 0:
        raise ValueError(f"{case.name}: no pre-shear normal hold baseline was found.")
    return candidates[-min(int(tail_frames), int(candidates.size)) :]


def _case_metrics(case: ShearCase, baseline_tail_frames: int, active_command_ratio: float) -> dict[str, object]:
    if case.lateral_axis not in ("y", "z"):
        raise ValueError(f"{case.name}: lateral axis must be y or z.")
    baseline_indices = _baseline_indices(case, baseline_tail_frames)
    baseline_tangent = np.mean(case.force_pad_local[baseline_indices, 1:3], axis=0)
    baseline_normal = float(np.mean(case.tactile_force_channels[baseline_indices, 2]))
    baseline_delta = case.force_pad_local[baseline_indices, 1:3] - baseline_tangent.reshape(1, 2)
    baseline_noise_norm = np.linalg.norm(baseline_delta, axis=1)
    force_noise = max(
        float(np.percentile(baseline_noise_norm, 99.0)),
        3.0 * float(np.sqrt(np.mean(np.square(baseline_noise_norm)))),
        1.0e-9,
    )

    label = "positive" if case.direction_sign > 0.0 else "negative"
    shear_phase_indices = np.concatenate(
        (_indices(case, f"shear_{label}_ramp"), _indices(case, f"shear_{label}_hold"))
    )
    if shear_phase_indices.size == 0:
        raise ValueError(f"{case.name}: no shear ramp/hold frames were found.")
    command_amplitude = float(np.max(np.abs(case.commanded_lateral_mm[shear_phase_indices])))
    command_threshold = float(active_command_ratio) * max(command_amplitude, EPS)
    active_indices = shear_phase_indices[
        np.abs(case.commanded_lateral_mm[shear_phase_indices]) >= command_threshold
    ]
    if active_indices.size == 0:
        raise ValueError(f"{case.name}: no active shear frames exceed the command threshold.")

    delta_tangent = case.force_pad_local[:, 1:3] - baseline_tangent.reshape(1, 2)
    active_delta = delta_tangent[active_indices]
    active_magnitude = np.linalg.norm(active_delta, axis=1)
    expected_direction = np.asarray(
        [case.direction_sign, 0.0] if case.lateral_axis == "y" else [0.0, case.direction_sign],
        dtype=np.float64,
    )
    signed_projection = active_delta @ expected_direction
    cosine = signed_projection / np.maximum(active_magnitude, EPS)
    response_active = active_magnitude > force_noise
    sign_correct = signed_projection > 0.0
    delta_normal = case.tactile_force_channels[active_indices, 2] - baseline_normal

    return_phase = _indices(case, f"shear_zero_after_{label}_hold")
    return_tangent_magnitude = (
        float(np.mean(np.linalg.norm(case.force_pad_local[return_phase, 1:3], axis=1)))
        if return_phase.size
        else float("inf")
    )
    return {
        "name": case.name,
        "frame_count": len(case.phases),
        "active_indices": active_indices,
        "baseline_tangent": baseline_tangent,
        "baseline_normal": baseline_normal,
        "force_noise": force_noise,
        "active_magnitude": active_magnitude,
        "signed_projection": signed_projection,
        "cosine": cosine,
        "response_active": response_active,
        "sign_correct": sign_correct,
        "delta_normal": delta_normal,
        "delta_tangent": active_delta,
        "command_amplitude_mm": command_amplitude,
        "settled_actual_lateral_mm": float(np.mean(case.actual_lateral_mm[_indices(case, f"shear_{label}_hold")])),
        "return_tangent_magnitude_tactile_unit": return_tangent_magnitude,
    }


def validate_cases(
    positive: ShearCase,
    negative: ShearCase,
    *,
    baseline_tail_frames: int,
    active_command_ratio: float,
    accept_min_active_response_ratio: float,
    accept_min_direction_cosine: float,
    accept_min_sign_flip_rate: float,
    accept_max_normal_pollution_ratio: float,
) -> tuple[dict[str, object], dict[str, object]]:
    if positive.lateral_axis != negative.lateral_axis:
        raise ValueError("Positive and negative cases must use the same Pad-local lateral axis.")
    metrics = [
        _case_metrics(positive, baseline_tail_frames, active_command_ratio),
        _case_metrics(negative, baseline_tail_frames, active_command_ratio),
    ]
    response_active = np.concatenate([np.asarray(value["response_active"], dtype=bool) for value in metrics])
    cosine = np.concatenate([np.asarray(value["cosine"], dtype=np.float64) for value in metrics])
    sign_correct = np.concatenate([np.asarray(value["sign_correct"], dtype=bool) for value in metrics])
    delta_normal = np.concatenate([np.asarray(value["delta_normal"], dtype=np.float64) for value in metrics])
    delta_tangent = np.concatenate([np.asarray(value["delta_tangent"], dtype=np.float64) for value in metrics])
    active_response_ratio = float(np.mean(response_active))
    direction_cosine = float(np.mean(cosine))
    sign_flip_rate = float(np.mean(sign_correct))
    normal_pollution_ratio = float(
        np.sqrt(np.mean(np.square(delta_normal)))
        / max(float(np.sqrt(np.mean(np.sum(np.square(delta_tangent), axis=1)))), EPS)
    )

    checks = {
        "positive_source_probe_passed": positive.source_probe_passed,
        "negative_source_probe_passed": negative.source_probe_passed,
        "positive_source_estimator_passed": positive.source_estimator_passed,
        "negative_source_estimator_passed": negative.source_estimator_passed,
        "shear_active_response_ratio_above_95_percent": active_response_ratio
        > float(accept_min_active_response_ratio),
        "direction_cosine_above_0_9": direction_cosine > float(accept_min_direction_cosine),
        "sign_flip_rate_above_95_percent": sign_flip_rate > float(accept_min_sign_flip_rate),
        "normal_pollution_below_0_2": normal_pollution_ratio
        < float(accept_max_normal_pollution_ratio),
    }
    per_case_observed = {
        value["name"]: {
            "active_frame_count": int(np.asarray(value["active_indices"]).size),
            "force_noise_tactile_unit": float(value["force_noise"]),
            "active_response_ratio": float(np.mean(value["response_active"])),
            "direction_cosine_mean": float(np.mean(value["cosine"])),
            "correct_sign_rate": float(np.mean(value["sign_correct"])),
            "command_amplitude_mm": float(value["command_amplitude_mm"]),
            "settled_actual_lateral_mm": float(value["settled_actual_lateral_mm"]),
            "return_tangent_magnitude_tactile_unit": float(value["return_tangent_magnitude_tactile_unit"]),
        }
        for value in metrics
    }
    observed = {
        "lateral_axis_pad_local": positive.lateral_axis,
        "shear_active_response_ratio": active_response_ratio,
        "direction_cosine_similarity": direction_cosine,
        "sign_flip_rate": sign_flip_rate,
        "normal_pollution_delta_fz_over_delta_ft": normal_pollution_ratio,
        "per_case": per_case_observed,
    }
    thresholds = {
        "shear_active_response_ratio_strictly_greater_than": float(accept_min_active_response_ratio),
        "direction_cosine_strictly_greater_than": float(accept_min_direction_cosine),
        "sign_flip_rate_strictly_greater_than": float(accept_min_sign_flip_rate),
        "normal_pollution_strictly_less_than": float(accept_max_normal_pollution_ratio),
        "active_command_ratio": float(active_command_ratio),
    }
    verdict = {
        "lateral_shear_validation_passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": thresholds,
        "observed": observed,
        "force_source": "frozen_v5_new_7g_deformation_based_force_estimator",
        "calibrated_newton": False,
    }
    direction_error = {
        "lateral_axis_pad_local": positive.lateral_axis,
        "positive_wrong_sign_frames": [
            int(value)
            for value in np.asarray(metrics[0]["active_indices"])[~np.asarray(metrics[0]["sign_correct"])]
        ],
        "negative_wrong_sign_frames": [
            int(value)
            for value in np.asarray(metrics[1]["active_indices"])[~np.asarray(metrics[1]["sign_correct"])]
        ],
        "positive_cosine_mean": float(np.mean(metrics[0]["cosine"])),
        "negative_cosine_mean": float(np.mean(metrics[1]["cosine"])),
        "combined_cosine_mean": direction_cosine,
    }
    return verdict, direction_error


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    positive = _load_case(
        "positive_y",
        1.0,
        Path(args.positive_probe_dir).expanduser().resolve(),
        Path(args.positive_force_dir).expanduser().resolve(),
    )
    negative = _load_case(
        "negative_y",
        -1.0,
        Path(args.negative_probe_dir).expanduser().resolve(),
        Path(args.negative_force_dir).expanduser().resolve(),
    )
    verdict, direction_error = validate_cases(
        positive,
        negative,
        baseline_tail_frames=int(args.baseline_tail_frames),
        active_command_ratio=float(args.active_command_ratio),
        accept_min_active_response_ratio=float(args.accept_min_active_response_ratio),
        accept_min_direction_cosine=float(args.accept_min_direction_cosine),
        accept_min_sign_flip_rate=float(args.accept_min_sign_flip_rate),
        accept_max_normal_pollution_ratio=float(args.accept_max_normal_pollution_ratio),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "force_pad_local.npy", np.concatenate((positive.force_pad_local, negative.force_pad_local)))
    np.save(
        output_dir / "tactile_force_channels.npy",
        np.concatenate((positive.tactile_force_channels, negative.tactile_force_channels)),
    )
    np.save(
        output_dir / "shear_displacement.npy",
        np.concatenate((positive.shear_displacement, negative.shear_displacement)),
    )
    _write_json(output_dir / "shear_direction_error.json", direction_error)
    _write_json(output_dir / "shear_response_metrics.json", verdict)
    _write_json(output_dir / "verdict.json", verdict)
    print(json.dumps(verdict, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return verdict


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    verdict = run_cli(args)
    if bool(args.fail_on_verdict_fail) and not bool(verdict["lateral_shear_validation_passed"]):
        raise RuntimeError(f"7g-shear validation failed: {verdict}")


if __name__ == "__main__":
    main()
