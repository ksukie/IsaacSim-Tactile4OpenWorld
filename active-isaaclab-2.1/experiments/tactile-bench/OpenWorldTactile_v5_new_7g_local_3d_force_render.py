from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from membrane_local_frame import EPS, quat_wxyz_to_matrix


SCRIPT_VERSION = "v5_new_7g_membrane_local_3d_force_render_diagnostic_v1"
PROTECTED_FORCE_FILES = (
    "force_pad_local.npy",
    "tactile_force_channels.npy",
    "shear_displacement.npy",
)


@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    probe_dir: Path
    force_dir: Path
    video_name: str
    kind: str
    direction_sign: float


@dataclass(frozen=True)
class CaseData:
    spec: CaseSpec
    phases: tuple[str, ...]
    rest_membrane_local: np.ndarray
    current_membrane_local: np.ndarray
    displacement_pad_local: np.ndarray
    front_mask: np.ndarray
    front_triangles: np.ndarray
    vertex_area_m2: np.ndarray
    contact_vertex_count: np.ndarray
    commanded_indentation_mm: np.ndarray
    commanded_lateral_mm: np.ndarray | None
    force_pad_local_tu: np.ndarray
    force_membrane_local_tu: np.ndarray
    tactile_force_channels_tu: np.ndarray
    contact_activation_weight: np.ndarray
    normal_deformation_volume_m3: np.ndarray
    shear_deformation_volume_m3: np.ndarray
    vertex_force_pad_local_tu: np.ndarray
    active_arrow_count: np.ndarray
    contribution_absolute_error_tu: np.ndarray
    contribution_relative_error: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline, read-only visualization of the restored frozen 7g normal Fz and membrane-Y "
            "tactile Fx cases. Frozen force arrays decide force magnitude; probe arrays only decide "
            "where membrane geometry and arrows are drawn."
        )
    )
    parser.add_argument("--contract_dir", required=True)
    parser.add_argument("--frame_contract_dir", required=True)
    parser.add_argument("--normal_probe_dir", required=True)
    parser.add_argument("--normal_force_dir", required=True)
    parser.add_argument("--plus_y_probe_dir", required=True)
    parser.add_argument("--plus_y_force_dir", required=True)
    parser.add_argument("--minus_y_probe_dir", required=True)
    parser.add_argument("--minus_y_force_dir", required=True)
    parser.add_argument("--shear_validation_dir", required=True)
    parser.add_argument(
        "--output_dir",
        default="/tmp/openworldtactile_uipc_v5_new_7g_membrane_local_3d_force_diagnostic",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--deformation_visual_scale", type=float, default=1.0)
    parser.add_argument("--force_arrow_span_ratio", type=float, default=0.28)
    parser.add_argument("--max_vertex_arrows", type=int, default=96)
    parser.add_argument("--arrow_zero_tolerance_tu", type=float, default=1.0e-12)
    parser.add_argument("--max_absolute_contribution_error_tu", type=float, default=1.0e-8)
    parser.add_argument("--max_relative_contribution_error", type=float, default=1.0e-6)
    parser.add_argument("--release_peak_ratio", type=float, default=0.02)
    parser.add_argument("--fail_on_verdict_fail", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in (
        "fps",
        "deformation_visual_scale",
        "force_arrow_span_ratio",
        "max_absolute_contribution_error_tu",
        "max_relative_contribution_error",
        "release_peak_ratio",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name} must be finite and > 0.")
    if float(args.arrow_zero_tolerance_tu) < 0.0:
        parser.error("--arrow_zero_tolerance_tu must be >= 0.")
    if int(args.width) < 640 or int(args.height) < 480:
        parser.error("--width/--height must be at least 640x480.")
    if int(args.max_vertex_arrows) <= 0:
        parser.error("--max_vertex_arrows must be > 0.")


def _load_npy(directory: Path, name: str) -> np.ndarray:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return np.load(path, allow_pickle=False)


def _load_json(directory: Path, name: str) -> object:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return json.loads(path.read_text())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_paths(specs: list[CaseSpec], shear_validation_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        paths.extend(spec.force_dir / name for name in PROTECTED_FORCE_FILES)
    paths.append(shear_validation_dir / "shear_response_metrics.json")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected frozen inputs are missing: {missing}")
    return paths


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha256(path) for path in paths}


def _indices_to_mask(indices: np.ndarray, vertex_count: int, name: str) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0 or np.any(values < 0) or np.any(values >= vertex_count):
        raise ValueError(f"{name} contains no valid vertices for N={vertex_count}.")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} contains duplicate vertex ids.")
    mask = np.zeros(vertex_count, dtype=bool)
    mask[values] = True
    return mask


def _front_triangles(triangles: np.ndarray, front_mask: np.ndarray) -> np.ndarray:
    topology = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if topology.size == 0 or np.any(topology < 0) or np.any(topology >= front_mask.size):
        raise ValueError("Surface triangle topology is empty or out of bounds.")
    result = topology[np.all(front_mask[topology], axis=1)]
    if result.size == 0:
        raise ValueError("No front-surface triangles were found.")
    return result


def _pad_points_to_membrane_local(
    points_pad_local: np.ndarray,
    membrane_pos_pad_local: np.ndarray,
    rotation_pad_from_membrane: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_pad_local, dtype=np.float64)
    return (points - membrane_pos_pad_local) @ rotation_pad_from_membrane


def _pad_vectors_to_membrane_local(
    vectors_pad_local: np.ndarray,
    rotation_pad_from_membrane: np.ndarray,
) -> np.ndarray:
    return np.asarray(vectors_pad_local, dtype=np.float64) @ rotation_pad_from_membrane


def _allocate_vertex_force(
    displacement_pad_local: np.ndarray,
    vertex_area_m2: np.ndarray,
    front_mask: np.ndarray,
    activation: np.ndarray,
    normal_volume: np.ndarray,
    shear_volume: np.ndarray,
    force_pad_local: np.ndarray,
    *,
    normal_gain: float,
    tangent_y_gain: float,
    tangent_z_gain: float,
    zero_tolerance_tu: float,
) -> np.ndarray:
    displacement = np.asarray(displacement_pad_local, dtype=np.float64)
    area = np.asarray(vertex_area_m2, dtype=np.float64).reshape(-1)
    mask = np.asarray(front_mask, dtype=bool).reshape(-1)
    weight_area = area * mask.astype(np.float64)
    normal_weight = np.clip(-displacement[..., 0], 0.0, None) * weight_area.reshape(1, -1)
    shear_raw = (
        displacement[..., 1:3]
        * np.asarray(activation, dtype=np.float64)[..., None]
        * weight_area.reshape(1, -1, 1)
    )
    frames, vertices, _ = displacement.shape
    volume = np.zeros((frames, vertices, 3), dtype=np.float64)
    for frame in range(frames):
        if float(np.linalg.norm(force_pad_local[frame])) <= float(zero_tolerance_tu):
            continue
        normal_sum = float(np.sum(normal_weight[frame], dtype=np.float64))
        if float(normal_volume[frame]) > 0.0 and normal_sum > EPS:
            volume[frame, :, 0] = normal_weight[frame] * (
                float(normal_volume[frame]) / normal_sum
            )
        for component in range(2):
            raw = shear_raw[frame, :, component]
            residual = float(shear_volume[frame, component] - np.sum(raw, dtype=np.float64))
            correction_weight = weight_area * activation[frame]
            correction_sum = float(np.sum(correction_weight, dtype=np.float64))
            volume[frame, :, component + 1] = raw
            if abs(residual) > 0.0:
                if correction_sum <= EPS:
                    candidates = np.flatnonzero(mask)
                    if candidates.size == 0:
                        raise ValueError("Cannot allocate frozen shear baseline correction.")
                    volume[frame, int(candidates[0]), component + 1] += residual
                else:
                    volume[frame, :, component + 1] += correction_weight * (
                        residual / correction_sum
                    )
    vertex_force = np.empty_like(volume)
    vertex_force[..., 0] = -float(normal_gain) * volume[..., 0]
    vertex_force[..., 1] = float(tangent_y_gain) * volume[..., 1]
    vertex_force[..., 2] = float(tangent_z_gain) * volume[..., 2]
    return vertex_force


def _load_case(
    spec: CaseSpec,
    contract_dir: Path,
    membrane_pos_pad_local: np.ndarray,
    rotation_pad_from_membrane: np.ndarray,
    *,
    zero_tolerance_tu: float,
) -> CaseData:
    probe_dir = spec.probe_dir
    force_dir = spec.force_dir
    phases_value = _load_json(probe_dir, "phase_history.json")
    force_metadata = _load_json(force_dir, "metadata.json")
    force_verdict = _load_json(force_dir, "verdict.json")
    if not isinstance(phases_value, list) or not isinstance(force_metadata, dict):
        raise ValueError(f"{spec.key}: invalid phase or force metadata JSON.")
    if not isinstance(force_verdict, dict) or not bool(
        force_verdict.get("deformation_based_force_estimator_passed", False)
    ):
        raise ValueError(f"{spec.key}: frozen 7g source verdict did not pass.")
    displacement = np.asarray(_load_npy(probe_dir, "surface_deformation.npy"), dtype=np.float64)
    rest_pad = np.asarray(
        _load_npy(probe_dir, "rest_surface_vertices_pad_local.npy"), dtype=np.float64
    )
    triangles = np.asarray(_load_npy(probe_dir, "surface_triangles.npy"), dtype=np.int64)
    front_indices = np.asarray(_load_npy(probe_dir, "front_surface_indices.npy"), dtype=np.int64)
    contact_count = np.asarray(_load_npy(probe_dir, "contact_vertex_count.npy"), dtype=np.int64).reshape(-1)
    command_normal = np.asarray(
        _load_npy(probe_dir, "commanded_indentation_mm.npy"), dtype=np.float64
    ).reshape(-1)
    command_lateral = None
    if spec.kind == "shear":
        command_lateral = np.asarray(
            _load_npy(probe_dir, "commanded_lateral_mm.npy"), dtype=np.float64
        ).reshape(-1)
    force_pad = np.asarray(_load_npy(force_dir, "force_pad_local.npy"), dtype=np.float64)
    tactile = np.asarray(_load_npy(force_dir, "tactile_force_channels.npy"), dtype=np.float64)
    activation = np.asarray(
        _load_npy(force_dir, "contact_activation_weight.npy"), dtype=np.float64
    )
    normal_volume = np.asarray(
        _load_npy(force_dir, "normal_deformation_volume.npy"), dtype=np.float64
    ).reshape(-1)
    shear_volume = np.asarray(
        _load_npy(force_dir, "shear_deformation_volume.npy"), dtype=np.float64
    )
    area = np.asarray(_load_npy(contract_dir, "vertex_area.npy"), dtype=np.float64).reshape(-1)
    contract_front = np.asarray(
        _load_npy(contract_dir, "front_surface_mask.npy"), dtype=bool
    ).reshape(-1)
    if displacement.ndim != 3 or displacement.shape[2] != 3:
        raise ValueError(f"{spec.key}: displacement must be [T,N,3].")
    frames, vertices, _ = displacement.shape
    front_mask = _indices_to_mask(front_indices, vertices, "front_surface_indices")
    expected_shapes = {
        "rest": rest_pad.shape == (vertices, 3),
        "force": force_pad.shape == (frames, 3),
        "tactile": tactile.shape == (frames, 3),
        "activation": activation.shape == (frames, vertices),
        "normal_volume": normal_volume.shape == (frames,),
        "shear_volume": shear_volume.shape == (frames, 2),
        "area": area.shape == (vertices,),
        "contract_front": contract_front.shape == (vertices,),
        "phase": len(phases_value) == frames,
        "contact": contact_count.shape == (frames,),
        "normal_command": command_normal.shape == (frames,),
        "lateral_command": command_lateral is None or command_lateral.shape == (frames,),
    }
    if not all(expected_shapes.values()):
        raise ValueError(f"{spec.key}: input dimensions disagree: {expected_shapes}")
    arrays = (
        displacement,
        rest_pad,
        force_pad,
        tactile,
        activation,
        normal_volume,
        shear_volume,
        area,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError(f"{spec.key}: an input array contains NaN or Inf.")
    if not np.array_equal(front_mask, contract_front):
        raise ValueError(f"{spec.key}: probe and frozen contract front masks differ.")
    front_triangles = _front_triangles(triangles, front_mask)
    current_pad = rest_pad.reshape(1, vertices, 3) + displacement
    rest_membrane = _pad_points_to_membrane_local(
        rest_pad, membrane_pos_pad_local, rotation_pad_from_membrane
    )
    current_membrane = _pad_points_to_membrane_local(
        current_pad, membrane_pos_pad_local, rotation_pad_from_membrane
    )
    vertex_force = _allocate_vertex_force(
        displacement,
        area,
        front_mask,
        activation,
        normal_volume,
        shear_volume,
        force_pad,
        normal_gain=float(force_metadata["normal_gain_tu_per_m3"]),
        tangent_y_gain=float(force_metadata["tangent_y_gain_tu_per_m3"]),
        tangent_z_gain=float(force_metadata["tangent_z_gain_tu_per_m3"]),
        zero_tolerance_tu=zero_tolerance_tu,
    )
    vertex_sum = np.sum(vertex_force, axis=1, dtype=np.float64)
    absolute_error = np.linalg.norm(vertex_sum - force_pad, axis=1)
    relative_error = absolute_error / np.maximum(np.linalg.norm(force_pad, axis=1), EPS)
    active_count = np.count_nonzero(
        np.linalg.norm(vertex_force, axis=2) > float(zero_tolerance_tu), axis=1
    )
    force_membrane = _pad_vectors_to_membrane_local(force_pad, rotation_pad_from_membrane)
    return CaseData(
        spec=spec,
        phases=tuple(str(value) for value in phases_value),
        rest_membrane_local=rest_membrane,
        current_membrane_local=current_membrane,
        displacement_pad_local=displacement,
        front_mask=front_mask,
        front_triangles=front_triangles,
        vertex_area_m2=area,
        contact_vertex_count=contact_count,
        commanded_indentation_mm=command_normal,
        commanded_lateral_mm=command_lateral,
        force_pad_local_tu=force_pad,
        force_membrane_local_tu=force_membrane,
        tactile_force_channels_tu=tactile,
        contact_activation_weight=activation,
        normal_deformation_volume_m3=normal_volume,
        shear_deformation_volume_m3=shear_volume,
        vertex_force_pad_local_tu=vertex_force,
        active_arrow_count=active_count,
        contribution_absolute_error_tu=absolute_error,
        contribution_relative_error=relative_error,
    )


def _phase_indices(case: CaseData, exact: str | None = None, prefix: str | None = None) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, phase in enumerate(case.phases)
            if (exact is not None and phase == exact)
            or (prefix is not None and phase.startswith(prefix))
        ],
        dtype=np.int64,
    )


def _case_direction_metrics(case: CaseData) -> dict[str, object]:
    force = case.force_pad_local_tu
    if case.spec.kind == "normal":
        hold_candidates = _phase_indices(case, prefix="load_hold_")
        if hold_candidates.size == 0:
            raise ValueError("Normal case has no load_hold phase.")
        maximum_command = float(np.max(case.commanded_indentation_mm[hold_candidates]))
        indices = hold_candidates[
            np.isclose(case.commanded_indentation_mm[hold_candidates], maximum_command)
        ]
        selected = force[indices]
        cosine = (-selected[:, 0]) / np.maximum(np.linalg.norm(selected, axis=1), EPS)
        passed = bool(
            np.all(selected[:, 0] < 0.0)
            and np.all(case.tactile_force_channels_tu[indices, 2] > 0.0)
            and float(np.mean(cosine)) > 0.95
        )
        return {
            "passed": passed,
            "hold_frame_count": int(indices.size),
            "mean_minus_x_direction_cosine": float(np.mean(cosine)),
            "mean_force_pad_x_tu": float(np.mean(selected[:, 0])),
            "mean_tactile_fz_tu": float(
                np.mean(case.tactile_force_channels_tu[indices, 2])
            ),
        }
    label = "positive" if case.spec.direction_sign > 0.0 else "negative"
    indices = _phase_indices(case, exact=f"shear_{label}_hold")
    if indices.size == 0:
        raise ValueError(f"{case.spec.key}: shear hold phase is missing.")
    tangent = force[indices, 1:3]
    expected = np.asarray((case.spec.direction_sign, 0.0), dtype=np.float64)
    cosine = (tangent @ expected) / np.maximum(np.linalg.norm(tangent, axis=1), EPS)
    signed_y = force[indices, 1] * case.spec.direction_sign
    signed_tactile = case.tactile_force_channels_tu[indices, 0] * case.spec.direction_sign
    return {
        "passed": bool(
            np.all(signed_y > 0.0)
            and np.all(signed_tactile > 0.0)
            and float(np.mean(cosine)) > 0.90
        ),
        "hold_frame_count": int(indices.size),
        "mean_tangent_direction_cosine": float(np.mean(cosine)),
        "mean_signed_force_pad_y_tu": float(np.mean(signed_y)),
        "mean_signed_tactile_fx_tu": float(np.mean(signed_tactile)),
    }


def _release_metrics(case: CaseData, zero_tolerance_tu: float, release_ratio: float) -> dict[str, object]:
    magnitude = np.linalg.norm(case.force_pad_local_tu, axis=1)
    peak = float(np.max(magnitude))
    tail_count = min(3, magnitude.size)
    final_peak = float(np.max(magnitude[-tail_count:]))
    ratio = final_peak / max(peak, EPS)
    final_active = int(np.max(case.active_arrow_count[-tail_count:]))
    return {
        "passed": bool(
            ratio < float(release_ratio)
            and (
                final_active == 0
                or final_peak <= float(zero_tolerance_tu)
            )
        ),
        "peak_force_magnitude_tu": peak,
        "release_tail_peak_tu": final_peak,
        "release_to_peak_ratio": ratio,
        "release_tail_max_active_arrow_count": final_active,
    }


def _no_contact_metrics(case: CaseData, zero_tolerance_tu: float) -> dict[str, object]:
    candidates = np.flatnonzero(case.contact_vertex_count == 0)
    if candidates.size == 0:
        return {"passed": False, "no_contact_frame_count": 0}
    magnitudes = np.linalg.norm(case.force_pad_local_tu[candidates], axis=1)
    active = case.active_arrow_count[candidates]
    failing = candidates[
        (magnitudes > float(zero_tolerance_tu)) | (active != 0)
    ]
    return {
        "passed": bool(
            np.all(magnitudes <= float(zero_tolerance_tu))
            and np.all(active == 0)
        ),
        "no_contact_frame_count": int(candidates.size),
        "max_no_contact_force_tu": float(np.max(magnitudes)),
        "max_no_contact_active_arrow_count": int(np.max(active)),
        "failing_frame_indices": [int(value) for value in failing],
        "initial_warmup_frames_excluded": False,
    }


def _project_basis() -> np.ndarray:
    return np.asarray([[0.38, 1.0, 0.0], [-0.27, 0.0, -1.0]], dtype=np.float64)


def _draw_arrow(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    first = tuple(np.round(start).astype(int))
    second = tuple(np.round(end).astype(int))
    if first != second:
        cv2.arrowedLine(image, first, second, color, thickness, cv2.LINE_AA, tipLength=0.20)


def _render_frame(
    case: CaseData,
    frame_index: int,
    *,
    width: int,
    height: int,
    center_mm: np.ndarray,
    pixels_per_mm: float,
    net_force_mm_per_tu: float,
    vertex_force_mm_per_tu: float,
    deformation_scale: float,
    vertex_indices: np.ndarray,
    rotation_pad_from_membrane: np.ndarray,
    zero_tolerance_tu: float,
) -> np.ndarray:
    image = np.full((height, width, 3), (20, 23, 28), dtype=np.uint8)
    origin = np.asarray((width * 0.53, height * 0.54), dtype=np.float64)
    basis = _project_basis()

    def project(points_m: np.ndarray) -> np.ndarray:
        points_mm = np.asarray(points_m, dtype=np.float64) * 1000.0 - center_mm
        return origin.reshape(1, 2) + (basis @ points_mm.T).T * pixels_per_mm

    rest = case.rest_membrane_local
    current_raw = case.current_membrane_local[frame_index]
    current = rest + float(deformation_scale) * (current_raw - rest)
    rest_px = project(rest)
    current_px = project(current)
    for triangle in case.front_triangles:
        cv2.polylines(
            image,
            [np.round(rest_px[triangle]).astype(np.int32)],
            True,
            (95, 95, 95),
            1,
            cv2.LINE_AA,
        )
    depth = np.mean(current[case.front_triangles, 0], axis=1)
    for triangle_id in np.argsort(depth):
        triangle = case.front_triangles[int(triangle_id)]
        polygon = np.round(current_px[triangle]).astype(np.int32)
        cv2.fillConvexPoly(image, polygon, (50, 61, 72), cv2.LINE_AA)
        cv2.polylines(image, [polygon], True, (110, 128, 139), 1, cv2.LINE_AA)

    vertex_force_pad = case.vertex_force_pad_local_tu[frame_index]
    vertex_force_membrane = _pad_vectors_to_membrane_local(
        vertex_force_pad, rotation_pad_from_membrane
    )
    if float(np.linalg.norm(case.force_pad_local_tu[frame_index])) > float(zero_tolerance_tu):
        for vertex in vertex_indices:
            start = current_px[int(vertex)]
            normal = np.asarray((vertex_force_membrane[int(vertex), 0], 0.0, 0.0))
            tangent = np.asarray(
                (0.0, vertex_force_membrane[int(vertex), 1], vertex_force_membrane[int(vertex), 2])
            )
            if float(np.linalg.norm(normal)) > float(zero_tolerance_tu):
                endpoint = current[int(vertex)] + normal * vertex_force_mm_per_tu * 1.0e-3
                _draw_arrow(image, start, project(endpoint.reshape(1, 3))[0], (0, 145, 255), 1)
            if float(np.linalg.norm(tangent)) > float(zero_tolerance_tu):
                endpoint = current[int(vertex)] + tangent * vertex_force_mm_per_tu * 1.0e-3
                _draw_arrow(image, start, project(endpoint.reshape(1, 3))[0], (255, 220, 70), 1)
        center = np.mean(current[case.front_mask], axis=0)
        net_end = center + case.force_membrane_local_tu[frame_index] * net_force_mm_per_tu * 1.0e-3
        _draw_arrow(image, project(center.reshape(1, 3))[0], project(net_end.reshape(1, 3))[0], (0, 235, 255), 5)

    axis_anchor = center_mm * 1.0e-3 + np.asarray((0.0, -7.2e-3, -6.2e-3))
    axis_start = project(axis_anchor.reshape(1, 3))[0]
    for vector, color, label in (
        (np.asarray((2.0e-3, 0.0, 0.0)), (50, 70, 255), "Membrane +X"),
        (np.asarray((0.0, 2.0e-3, 0.0)), (70, 225, 80), "Membrane +Y / tactile Fx"),
        (np.asarray((0.0, 0.0, 2.0e-3)), (255, 120, 60), "Membrane +Z / tactile -Fy"),
    ):
        end = project((axis_anchor + vector).reshape(1, 3))[0]
        _draw_arrow(image, axis_start, end, color, 3)
        cv2.putText(image, label, tuple(np.round(end + (5, -5)).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    force = case.force_pad_local_tu[frame_index]
    tactile = case.tactile_force_channels_tu[frame_index]
    command = float(case.commanded_indentation_mm[frame_index])
    if case.commanded_lateral_mm is not None:
        command_text = f"normal={command:+.4f} mm, membrane Y={float(case.commanded_lateral_mm[frame_index]):+.4f} mm"
    else:
        command_text = f"membrane X press command={command:+.4f} mm (+X is unload)"
    lines = (
        case.spec.title,
        f"phase: {case.phases[frame_index]}",
        f"command: {command_text}",
        f"force_pad_local = [{force[0]:+.6g}, {force[1]:+.6g}, {force[2]:+.6g}] TU",
        f"tactile = [{tactile[0]:+.6g}, {tactile[1]:+.6g}, {tactile[2]:+.6g}] TU",
        f"active arrows: {int(case.active_arrow_count[frame_index])}",
        f"deformation visual scale: {deformation_scale:g}x",
        f"arrow scales: vertex={vertex_force_mm_per_tu:.6g}, net={net_force_mm_per_tu:.6g} mm/TU",
    )
    for row, line in enumerate(lines):
        cv2.putText(image, line, (26, 38 + row * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52 if row else 0.72, (238, 240, 242), 1 if row else 2, cv2.LINE_AA)
    cv2.putText(image, "gray: rest | surface: current | orange: normal | cyan: tangent | yellow: frozen total", (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 215), 1, cv2.LINE_AA)
    return image


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height))
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    return writer


def _render_videos(
    cases: list[CaseData],
    output_dir: Path,
    *,
    fps: float,
    width: int,
    height: int,
    deformation_scale: float,
    force_span_ratio: float,
    max_vertex_arrows: int,
    rotation_pad_from_membrane: np.ndarray,
    zero_tolerance_tu: float,
) -> tuple[dict[str, str], dict[str, object]]:
    all_display_points = np.concatenate(
        [
            (
                case.rest_membrane_local.reshape(1, *case.rest_membrane_local.shape)
                + deformation_scale
                * (
                    case.current_membrane_local
                    - case.rest_membrane_local.reshape(1, *case.rest_membrane_local.shape)
                )
            ).reshape(-1, 3)
            for case in cases
        ],
        axis=0,
    )
    minimum_mm = np.min(all_display_points, axis=0) * 1000.0
    maximum_mm = np.max(all_display_points, axis=0) * 1000.0
    center_mm = 0.5 * (minimum_mm + maximum_mm)
    corners = []
    for x in (minimum_mm[0], maximum_mm[0]):
        for y in (minimum_mm[1], maximum_mm[1]):
            for z in (minimum_mm[2], maximum_mm[2]):
                corners.append(_project_basis() @ (np.asarray((x, y, z)) - center_mm))
    projected_extent = np.maximum(np.ptp(np.asarray(corners), axis=0), 1.0e-6)
    pixels_per_mm = min(0.66 * width / projected_extent[0], 0.66 * height / projected_extent[1])
    global_peak = max(
        float(np.max(np.linalg.norm(case.force_pad_local_tu, axis=1))) for case in cases
    )
    membrane_span = float(max(maximum_mm[1] - minimum_mm[1], maximum_mm[2] - minimum_mm[2]))
    force_mm_per_tu = float(force_span_ratio) * membrane_span / max(global_peak, EPS)
    global_peak_vertex = max(
        float(np.max(np.linalg.norm(case.vertex_force_pad_local_tu, axis=2))) for case in cases
    )
    vertex_force_mm_per_tu = (
        float(force_span_ratio) * 0.35 * membrane_span / max(global_peak_vertex, EPS)
    )
    combined_path = output_dir / "validated_fx_fz_combined.mp4"
    combined = _open_writer(combined_path, fps, width, height)
    paths: dict[str, str] = {}
    expected_counts: dict[str, int] = {}
    try:
        for case in cases:
            path = output_dir / case.spec.video_name
            writer = _open_writer(path, fps, width, height)
            front_vertices = np.flatnonzero(case.front_mask)
            if front_vertices.size > int(max_vertex_arrows):
                peak_by_vertex = np.max(
                    np.linalg.norm(case.vertex_force_pad_local_tu, axis=2), axis=0
                )
                order = np.argsort(peak_by_vertex[front_vertices], kind="stable")
                vertex_indices = np.sort(front_vertices[order[-int(max_vertex_arrows) :]])
            else:
                vertex_indices = front_vertices
            try:
                for frame in range(case.force_pad_local_tu.shape[0]):
                    image = _render_frame(
                        case,
                        frame,
                        width=width,
                        height=height,
                        center_mm=center_mm,
                        pixels_per_mm=pixels_per_mm,
                        net_force_mm_per_tu=force_mm_per_tu,
                        vertex_force_mm_per_tu=vertex_force_mm_per_tu,
                        deformation_scale=deformation_scale,
                        vertex_indices=vertex_indices,
                        rotation_pad_from_membrane=rotation_pad_from_membrane,
                        zero_tolerance_tu=zero_tolerance_tu,
                    )
                    writer.write(image)
                    combined.write(image)
            finally:
                writer.release()
            paths[case.spec.key] = str(path)
            expected_counts[str(path)] = int(case.force_pad_local_tu.shape[0])
    finally:
        combined.release()
    paths["combined"] = str(combined_path)
    expected_counts[str(combined_path)] = int(
        sum(case.force_pad_local_tu.shape[0] for case in cases)
    )
    metadata = {
        "camera": "fixed_membrane_local_orthographic_oblique",
        "coordinate_minimum_mm": minimum_mm.tolist(),
        "coordinate_maximum_mm": maximum_mm.tolist(),
        "pixels_per_mm": float(pixels_per_mm),
        "force_arrow_mm_per_tu": float(force_mm_per_tu),
        "vertex_force_arrow_mm_per_tu": float(vertex_force_mm_per_tu),
        "global_peak_vertex_force_tu": float(global_peak_vertex),
        "deformation_visual_scale": float(deformation_scale),
        "global_peak_force_tu": float(global_peak),
        "per_frame_normalization": False,
        "same_scale_all_cases": True,
        "expected_frame_counts": expected_counts,
    }
    return paths, metadata


def _verify_video(path: Path, expected_frames: int, width: int, height: int) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"passed": False, "reason": "cannot_open"}
    frame_count = 0
    black_frames = 0
    wrong_shape = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            if frame.shape[:2] != (height, width):
                wrong_shape += 1
            if float(np.mean(frame)) < 1.0 or float(np.std(frame)) < 1.0:
                black_frames += 1
    finally:
        capture.release()
    return {
        "passed": bool(
            frame_count == int(expected_frames)
            and black_frames == 0
            and wrong_shape == 0
        ),
        "expected_frame_count": int(expected_frames),
        "decoded_frame_count": int(frame_count),
        "black_or_blank_frame_count": int(black_frames),
        "wrong_shape_frame_count": int(wrong_shape),
    }


def _specs(args: argparse.Namespace) -> list[CaseSpec]:
    return [
        CaseSpec("normal", "Frozen 7g: membrane -X press / +X unload -> tactile Fz", Path(args.normal_probe_dir).resolve(), Path(args.normal_force_dir).resolve(), "normal_press_release.mp4", "normal", 0.0),
        CaseSpec("plus_y", "Frozen 7g: membrane +Y shear -> tactile Fx > 0", Path(args.plus_y_probe_dir).resolve(), Path(args.plus_y_force_dir).resolve(), "plus_y_shear.mp4", "shear", 1.0),
        CaseSpec("minus_y", "Frozen 7g: membrane -Y shear -> tactile Fx < 0", Path(args.minus_y_probe_dir).resolve(), Path(args.minus_y_force_dir).resolve(), "minus_y_shear.mp4", "shear", -1.0),
    ]


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    contract_dir = Path(args.contract_dir).expanduser().resolve()
    frame_contract_dir = Path(args.frame_contract_dir).expanduser().resolve()
    shear_validation_dir = Path(args.shear_validation_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _specs(args)
    for name in (
        *(spec.video_name for spec in specs),
        "validated_fx_fz_combined.mp4",
        "vertex_force_contribution_pad_local.npy",
    ):
        (output_dir / name).unlink(missing_ok=True)
    protected_paths = _protected_paths(specs, shear_validation_dir)
    before_hashes = _hashes(protected_paths)

    frame_metadata = _load_json(frame_contract_dir, "metadata.json")
    frame_contract = _load_json(frame_contract_dir, "membrane_frame_contract.json")
    if not isinstance(frame_metadata, dict) or not isinstance(frame_contract, dict):
        raise ValueError("Frame contract metadata must be JSON objects.")
    if not bool(frame_contract.get("membrane_frame_contract_passed", False)):
        raise ValueError("The independent membrane frame audit did not pass.")
    membrane_pos_pad = np.asarray(
        frame_metadata.get("membrane_pos_pad_local_m"), dtype=np.float64
    ).reshape(3)
    membrane_quat_pad = np.asarray(
        frame_metadata.get("membrane_quat_pad_local_wxyz"), dtype=np.float64
    ).reshape(4)
    rotation_pad_from_membrane = quat_wxyz_to_matrix(membrane_quat_pad)

    input_error: str | None = None
    cases: list[CaseData] = []
    try:
        cases = [
            _load_case(
                spec,
                contract_dir,
                membrane_pos_pad,
                rotation_pad_from_membrane,
                zero_tolerance_tu=float(args.arrow_zero_tolerance_tu),
            )
            for spec in specs
        ]
    except Exception as exc:
        input_error = f"{type(exc).__name__}: {exc}"

    gate1 = {
        "passed": input_error is None,
        "error": input_error,
        "case_count": len(cases),
    }
    direction_metrics: dict[str, object] = {}
    release_metrics: dict[str, object] = {}
    no_contact_metrics: dict[str, object] = {}
    contribution_metrics: dict[str, object] = {}
    if input_error is None:
        for case in cases:
            direction_metrics[case.spec.key] = _case_direction_metrics(case)
            release_metrics[case.spec.key] = _release_metrics(
                case,
                float(args.arrow_zero_tolerance_tu),
                float(args.release_peak_ratio),
            )
            no_contact_metrics[case.spec.key] = _no_contact_metrics(
                case, float(args.arrow_zero_tolerance_tu)
            )
            maximum_absolute = float(np.max(case.contribution_absolute_error_tu))
            maximum_relative = float(np.max(case.contribution_relative_error))
            contribution_metrics[case.spec.key] = {
                "passed": bool(
                    maximum_absolute < float(args.max_absolute_contribution_error_tu)
                    or maximum_relative < float(args.max_relative_contribution_error)
                ),
                "max_absolute_error_tu": maximum_absolute,
                "max_relative_error": maximum_relative,
            }
    gate3 = {
        "passed": bool(contribution_metrics) and all(
            bool(value["passed"]) for value in contribution_metrics.values()
        ),
        "cases": contribution_metrics,
    }
    gate4 = {
        "passed": bool(direction_metrics) and all(
            bool(value["passed"]) for value in direction_metrics.values()
        ),
        "cases": direction_metrics,
    }
    gate5 = {
        "passed": bool(release_metrics) and all(
            bool(value["passed"]) for value in release_metrics.values()
        ),
        "cases": release_metrics,
    }
    gate6 = {
        "passed": bool(no_contact_metrics) and all(
            bool(value["passed"]) for value in no_contact_metrics.values()
        ),
        "cases": no_contact_metrics,
    }
    gate7 = {
        "passed": True,
        "fixed_camera": True,
        "fixed_coordinate_range": True,
        "fixed_arrow_scale": True,
        "fixed_deformation_scale": True,
        "per_frame_normalization": False,
    }
    pre_render_passed = bool(
        gate1["passed"]
        and gate3["passed"]
        and gate4["passed"]
        and gate5["passed"]
        and gate6["passed"]
        and gate7["passed"]
    )

    video_paths: dict[str, str] = {}
    render_metadata: dict[str, object] | None = None
    video_checks: dict[str, object] = {}
    if pre_render_passed:
        video_paths, render_metadata = _render_videos(
            cases,
            output_dir,
            fps=float(args.fps),
            width=int(args.width),
            height=int(args.height),
            deformation_scale=float(args.deformation_visual_scale),
            force_span_ratio=float(args.force_arrow_span_ratio),
            max_vertex_arrows=int(args.max_vertex_arrows),
            rotation_pad_from_membrane=rotation_pad_from_membrane,
            zero_tolerance_tu=float(args.arrow_zero_tolerance_tu),
        )
        for path_string, expected in render_metadata["expected_frame_counts"].items():
            video_checks[path_string] = _verify_video(
                Path(path_string), int(expected), int(args.width), int(args.height)
            )
    gate8 = {
        "passed": bool(video_checks) and all(
            bool(value["passed"]) for value in video_checks.values()
        ),
        "videos": video_checks,
    }

    after_hashes = _hashes(protected_paths)
    unchanged = before_hashes == after_hashes
    gate2 = {
        "passed": unchanged,
        "all_protected_hashes_unchanged": unchanged,
        "before": before_hashes,
        "after": after_hashes,
    }
    if not gate2["passed"] or (video_paths and not gate8["passed"]):
        for path_string in video_paths.values():
            Path(path_string).unlink(missing_ok=True)
        video_paths = {}
    passed = bool(pre_render_passed and gate2["passed"] and gate8["passed"])

    if cases:
        vertex_force = np.concatenate(
            [case.vertex_force_pad_local_tu for case in cases], axis=0
        )
        np.save(output_dir / "vertex_force_contribution_pad_local.npy", vertex_force)
    metrics = {
        "gate_1_frozen_input_integrity": gate1,
        "gate_2_formal_data_unchanged": gate2,
        "gate_3_vertex_sum_matches_frozen_force": gate3,
        "gate_4_direction": gate4,
        "gate_5_release_zero": gate5,
        "gate_6_no_contact_suppression": gate6,
        "gate_7_fixed_display_scale": gate7,
        "gate_8_video_integrity": gate8,
    }
    metadata = {
        "script_version": SCRIPT_VERSION,
        "render_scope": "diagnostic_only",
        "validated_tactile_channels": ["Fx", "Fz"],
        "fy_force_validation_complete": False,
        "unit": "TU",
        "newton_calibrated": False,
        "frozen_7g_modified": False,
        "force_authority": "frozen force_pad_local.npy and tactile_force_channels.npy",
        "geometry_authority": "probe rest/deformation transformed by audited Pad-to-membrane extrinsic",
        "vertex_attribution": (
            "frozen normal/shear deformation volumes distributed over frozen activation and area weights; "
            "never used to redefine global force"
        ),
        "case_frame_offsets": {
            case.spec.key: {
                "frame_count": int(case.force_pad_local_tu.shape[0]),
                "probe_dir": str(case.spec.probe_dir),
                "force_dir": str(case.spec.force_dir),
            }
            for case in cases
        },
        "render": render_metadata,
        "videos": video_paths,
    }
    verdict = {
        "force_render_diagnostic_passed": passed,
        "render_scope": "diagnostic_only",
        "validated_tactile_channels": ["Fx", "Fz"],
        "fy_force_validation_complete": False,
        "unit": "TU",
        "newton_calibrated": False,
        "frozen_7g_modified": False,
        "videos_generated": bool(video_paths),
        "gates": {name: bool(value["passed"]) for name, value in metrics.items()},
    }
    _write_json(output_dir / "force_render_metrics.json", metrics)
    _write_json(output_dir / "force_render_metadata.json", metadata)
    _write_json(output_dir / "input_hashes.json", gate2)
    _write_json(output_dir / "verdict.json", verdict)
    return verdict


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    verdict = run_cli(args)
    print(json.dumps(verdict, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if bool(args.fail_on_verdict_fail) and not bool(verdict["force_render_diagnostic_passed"]):
        raise RuntimeError(f"Frozen 7g diagnostic render failed: {verdict}")


if __name__ == "__main__":
    main()
