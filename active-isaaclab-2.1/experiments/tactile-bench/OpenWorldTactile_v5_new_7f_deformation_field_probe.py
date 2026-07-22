from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_INPUT_DIR = Path("/tmp/openworldtactile_uipc_v5_new_7e_check")
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "Extract and validate Pad-local UIPC membrane deformation fields from a completed 7e run. "
        "This stage does not estimate force, pressure, stiffness, or damping."
    )
)
parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--deformation_threshold_mm", type=float, default=0.01)
parser.add_argument("--center_radius_mm", type=float, default=3.0)
parser.add_argument("--far_field_radius_mm", type=float, default=8.0)
parser.add_argument("--radial_bin_width_mm", type=float, default=2.0)
parser.add_argument("--accept_min_peak_normal_mm", type=float, default=0.10)
parser.add_argument("--accept_max_far_field_ratio", type=float, default=0.10)
parser.add_argument("--accept_max_peak_contact_distance_mm", type=float, default=2.5)
parser.add_argument("--accept_max_back_displacement_mm", type=float, default=0.20)
parser.add_argument("--accept_max_recovery_mm", type=float, default=0.05)
parser.add_argument("--accept_max_deformed_front_fraction", type=float, default=0.50)
parser.add_argument("--fail_on_verdict_fail", action="store_true")
args = parser.parse_args()


def _validate_args() -> None:
    for name in (
        "deformation_threshold_mm",
        "center_radius_mm",
        "far_field_radius_mm",
        "radial_bin_width_mm",
        "accept_min_peak_normal_mm",
        "accept_max_far_field_ratio",
        "accept_max_peak_contact_distance_mm",
        "accept_max_back_displacement_mm",
        "accept_max_recovery_mm",
        "accept_max_deformed_front_fraction",
    ):
        if float(getattr(args, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    if float(args.center_radius_mm) >= float(args.far_field_radius_mm):
        parser.error("--center_radius_mm must be smaller than --far_field_radius_mm.")
    if float(args.accept_max_far_field_ratio) > 1.0:
        parser.error("--accept_max_far_field_ratio must be <= 1.")
    if float(args.accept_max_deformed_front_fraction) > 1.0:
        parser.error("--accept_max_deformed_front_fraction must be <= 1.")


def _load_array(input_dir: Path, name: str) -> np.ndarray:
    path = input_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Required 7e output is missing: {path}")
    return np.load(path)


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"Required 7e output is missing: {path}")
    return json.loads(path.read_text())


def _phase_end_indices(phases: list[str], prefix: str) -> list[int]:
    return [
        index
        for index, phase in enumerate(phases)
        if str(phase).startswith(prefix)
        and (index == len(phases) - 1 or phases[index + 1] != phase)
    ]


def _radial_profile(
    radial_distance_mm: np.ndarray,
    normal_compression_mm: np.ndarray,
    shear_magnitude_mm: np.ndarray,
) -> np.ndarray:
    width = float(args.radial_bin_width_mm)
    max_radius = max(width, float(np.max(radial_distance_mm)))
    edges = np.arange(0.0, max_radius + width, width, dtype=np.float64)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (radial_distance_mm >= lower) & (radial_distance_mm < upper)
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        rows.append(
            (
                0.5 * (lower + upper),
                float(np.mean(normal_compression_mm[selected])),
                float(np.max(normal_compression_mm[selected])),
                float(np.mean(shear_magnitude_mm[selected])),
                count,
            )
        )
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    _validate_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else input_dir / "7f_deformation_field"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    displacement_l = _load_array(input_dir, "surface_deformation.npy").astype(np.float32)
    rest_surface_l = _load_array(input_dir, "rest_surface_vertices_pad_local.npy").astype(np.float32)
    front_indices = _load_array(input_dir, "front_surface_indices.npy").astype(np.int64)
    back_indices = _load_array(input_dir, "back_surface_indices.npy").astype(np.int64)
    geometric_contact_mask = _load_array(input_dir, "contact_vertex_mask.npy").astype(bool)
    commanded_indentation_mm = _load_array(input_dir, "commanded_indentation_mm.npy").astype(np.float64)
    actual_indentation_mm = _load_array(input_dir, "actual_indentation_mm.npy").astype(np.float64)
    phases = [str(value) for value in _load_json(input_dir / "phase_history.json")]
    source_verdict = _load_json(input_dir / "verdict.json")
    source_metadata = _load_json(input_dir / "metadata.json")

    if displacement_l.ndim != 3 or displacement_l.shape[2] != 3:
        raise ValueError(f"Expected T x N x 3 displacement, got {displacement_l.shape}.")
    frame_count, vertex_count, _ = displacement_l.shape
    if rest_surface_l.shape != (vertex_count, 3):
        raise ValueError(
            f"Rest surface shape {rest_surface_l.shape} does not match displacement vertices {vertex_count}."
        )
    if geometric_contact_mask.shape != (frame_count, vertex_count):
        raise ValueError(
            f"Contact mask shape {geometric_contact_mask.shape} does not match {(frame_count, vertex_count)}."
        )
    if len(phases) != frame_count or commanded_indentation_mm.shape != (frame_count,):
        raise ValueError("Phase and indentation histories must have one value per displacement frame.")
    if not np.all(np.isfinite(displacement_l)):
        raise ValueError("Displacement field contains NaN or infinite values.")

    normal_displacement_m = displacement_l[..., 0]
    normal_compression_m = np.clip(-normal_displacement_m, 0.0, None)
    shear_displacement_m = displacement_l[..., 1:3]
    shear_magnitude_m = np.linalg.norm(shear_displacement_m, axis=2)
    displacement_magnitude_m = np.linalg.norm(displacement_l, axis=2)
    front_vertex_mask = np.zeros(vertex_count, dtype=bool)
    front_vertex_mask[front_indices] = True
    threshold_m = float(args.deformation_threshold_mm) * 1.0e-3
    contact_deformation_mask = front_vertex_mask.reshape(1, -1) & (
        (normal_compression_m >= threshold_m) | (shear_magnitude_m >= threshold_m)
    )

    loading_hold_ends = _phase_end_indices(phases, "load_hold_")
    if not loading_hold_ends:
        raise ValueError("7e history contains no completed loading hold phase.")
    peak_hold_frame = max(
        loading_hold_ends,
        key=lambda index: (commanded_indentation_mm[index], actual_indentation_mm[index]),
    )
    final_frame = frame_count - 1

    peak_normal_frame_mm = normal_compression_m[peak_hold_frame] * 1000.0
    peak_shear_frame_mm = shear_magnitude_m[peak_hold_frame] * 1000.0
    peak_front_local = int(np.argmax(peak_normal_frame_mm[front_indices]))
    peak_vertex_index = int(front_indices[peak_front_local])

    contact_indices = np.flatnonzero(geometric_contact_mask[peak_hold_frame] & front_vertex_mask)
    if contact_indices.size:
        contact_center_yz_l = np.mean(rest_surface_l[contact_indices, 1:3], axis=0)
        peak_contact_distance_mm = float(
            np.min(
                np.linalg.norm(
                    rest_surface_l[contact_indices, 1:3] - rest_surface_l[peak_vertex_index, 1:3], axis=1
                )
            )
            * 1000.0
        )
    else:
        contact_center_yz_l = rest_surface_l[peak_vertex_index, 1:3]
        peak_contact_distance_mm = float("inf")

    radial_distance_front_mm = (
        np.linalg.norm(rest_surface_l[front_indices, 1:3] - contact_center_yz_l.reshape(1, 2), axis=1)
        * 1000.0
    )
    radial_profile = _radial_profile(
        radial_distance_front_mm,
        peak_normal_frame_mm[front_indices],
        peak_shear_frame_mm[front_indices],
    )
    center_selected = radial_distance_front_mm <= float(args.center_radius_mm)
    far_selected = radial_distance_front_mm >= float(args.far_field_radius_mm)
    center_mean_normal_mm = float(np.mean(peak_normal_frame_mm[front_indices][center_selected]))
    far_field_p95_normal_mm = (
        float(np.percentile(peak_normal_frame_mm[front_indices][far_selected], 95.0))
        if np.any(far_selected)
        else float("inf")
    )
    peak_normal_compression_mm = float(peak_normal_frame_mm[peak_vertex_index])
    far_field_ratio = far_field_p95_normal_mm / max(peak_normal_compression_mm, EPS)
    deformed_front_fraction = float(
        np.count_nonzero(contact_deformation_mask[peak_hold_frame, front_indices])
        / float(max(1, front_indices.size))
    )
    peak_back_displacement_mm = float(
        np.max(np.linalg.norm(displacement_l[peak_hold_frame, back_indices], axis=1)) * 1000.0
    )
    final_recovery_mm = float(
        np.max(np.linalg.norm(displacement_l[final_frame, front_indices], axis=1)) * 1000.0
    )
    peak_contact_shear_mm = (
        float(np.max(peak_shear_frame_mm[contact_indices])) if contact_indices.size else float("nan")
    )
    contact_shear_to_normal_ratio = peak_contact_shear_mm / max(peak_normal_compression_mm, EPS)

    hold_response_by_level_mm = {}
    for index in loading_hold_ends:
        level = float(commanded_indentation_mm[index])
        hold_response_by_level_mm[f"{level:g}"] = float(
            np.max(normal_compression_m[index, front_indices]) * 1000.0
        )
    hold_responses = list(hold_response_by_level_mm.values())
    monotonic_hold_response = all(
        current + 1.0e-6 >= previous for previous, current in zip(hold_responses, hold_responses[1:])
    )

    checks = {
        "source_7e_passed": bool(source_verdict.get("static_contact_deformation_passed", False)),
        "field_is_finite": bool(np.all(np.isfinite(displacement_l))),
        "normal_deformation_detected": peak_normal_compression_mm
        >= float(args.accept_min_peak_normal_mm),
        "normal_peak_matches_contact": peak_contact_distance_mm
        <= float(args.accept_max_peak_contact_distance_mm),
        "normal_field_is_localized": far_field_ratio <= float(args.accept_max_far_field_ratio),
        "deformed_region_is_not_global": deformed_front_fraction
        <= float(args.accept_max_deformed_front_fraction),
        "back_boundary_remains_fixed": peak_back_displacement_mm
        <= float(args.accept_max_back_displacement_mm),
        "normal_hold_response_is_monotonic": monotonic_hold_response,
        "surface_recovers_after_unload": final_recovery_mm <= float(args.accept_max_recovery_mm),
    }
    verdict = {
        "deformation_field_passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "deformation_threshold_mm": float(args.deformation_threshold_mm),
            "min_peak_normal_mm": float(args.accept_min_peak_normal_mm),
            "max_peak_contact_distance_mm": float(args.accept_max_peak_contact_distance_mm),
            "max_far_field_ratio": float(args.accept_max_far_field_ratio),
            "max_deformed_front_fraction": float(args.accept_max_deformed_front_fraction),
            "max_back_displacement_mm": float(args.accept_max_back_displacement_mm),
            "max_recovery_mm": float(args.accept_max_recovery_mm),
        },
        "observed": {
            "frame_count": frame_count,
            "surface_vertex_count": vertex_count,
            "front_vertex_count": int(front_indices.size),
            "back_vertex_count": int(back_indices.size),
            "peak_hold_frame": int(peak_hold_frame),
            "peak_hold_phase": phases[peak_hold_frame],
            "commanded_indentation_mm": float(commanded_indentation_mm[peak_hold_frame]),
            "actual_indentation_mm": float(actual_indentation_mm[peak_hold_frame]),
            "peak_vertex_index": peak_vertex_index,
            "geometric_contact_vertex_count": int(contact_indices.size),
            "peak_normal_compression_mm": peak_normal_compression_mm,
            "center_mean_normal_compression_mm": center_mean_normal_mm,
            "far_field_p95_normal_compression_mm": far_field_p95_normal_mm,
            "far_field_to_peak_ratio": far_field_ratio,
            "peak_to_contact_distance_mm": peak_contact_distance_mm,
            "deformed_front_fraction": deformed_front_fraction,
            "peak_back_displacement_mm": peak_back_displacement_mm,
            "final_recovery_mm": final_recovery_mm,
            "peak_contact_shear_mm": peak_contact_shear_mm,
            "contact_shear_to_normal_ratio_diagnostic_only": contact_shear_to_normal_ratio,
            "normal_hold_response_by_level_mm": hold_response_by_level_mm,
        },
        "coordinate_contract": {
            "frame": "UIPC_Pad local",
            "normal_axis": "+X outward; compression = max(-u_x, 0)",
            "shear_axes": "u_y and u_z",
            "units": "meters in npy arrays; millimeters in JSON diagnostics",
        },
        "force_source": "none",
        "pressure_source": "none",
    }

    np.save(output_dir / "surface_displacement_pad_local.npy", displacement_l)
    np.save(output_dir / "vertex_displacement.npy", displacement_l)
    np.save(output_dir / "normal_displacement.npy", normal_displacement_m.astype(np.float32))
    np.save(output_dir / "normal_compression.npy", normal_compression_m.astype(np.float32))
    np.save(output_dir / "shear_displacement.npy", shear_displacement_m.astype(np.float32))
    np.save(output_dir / "shear_magnitude.npy", shear_magnitude_m.astype(np.float32))
    np.save(output_dir / "displacement_magnitude.npy", displacement_magnitude_m.astype(np.float32))
    np.save(output_dir / "contact_deformation_mask.npy", contact_deformation_mask)
    np.save(output_dir / "geometric_contact_mask.npy", geometric_contact_mask)
    np.save(output_dir / "radial_profile_peak_hold.npy", radial_profile)
    (output_dir / "source_7e_metadata.json").write_text(json.dumps(source_metadata, indent=2) + "\n")
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    if bool(args.fail_on_verdict_fail) and not bool(verdict["deformation_field_passed"]):
        raise RuntimeError(f"7f deformation-field verdict failed: {verdict}")


if __name__ == "__main__":
    main()
