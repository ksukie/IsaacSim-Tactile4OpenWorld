from __future__ import annotations

"""One-shot validation for the V6.2 one-sided UIPC/PhysX coupling run.

The runtime intentionally performs no synchronization audit or verdict work.
This program validates native UIPC reaction feedback, zero formal object-pose
writes, frozen 7g reconstruction, optional field conservation, penetration
tolerance, and release-to-zero. It does not start Isaac Sim or alter the data.
"""

import argparse
import datetime as datetime_module
import json
import math
import os
import uuid
from pathlib import Path

import numpy as np

import OpenWorldTactile_v5_new_7g_deformation_force_estimator as frozen_7g


VERSION = "v6.2_one_shot_validation_v4_contact_cone_limited_subcycling"
VIDEO_NAMES = (
    "tactile_fx_signed_sequence.mp4",
    "tactile_fy_signed_sequence.mp4",
    "tactile_fz_sequence.mp4",
    "tactile_shear_magnitude_sequence.mp4",
    "tactile_fxyz_composite_sequence.mp4",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate one V6.2 simple-grasp tactile dataset offline."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--field_dir",
        default="",
        help="Defaults to INPUT_DIR/offline_tactile_field when that directory exists.",
    )
    parser.add_argument(
        "--output_json",
        default="",
        help="Defaults to INPUT_DIR/v6_2_validation.json.",
    )
    parser.add_argument("--force_atol_tu", type=float, default=1.0e-8)
    parser.add_argument("--field_atol_tu", type=float, default=1.0e-8)
    parser.add_argument("--quaternion_norm_atol", type=float, default=1.0e-4)
    parser.add_argument("--penetration_tolerance_mm", type=float, default=0.15)
    parser.add_argument(
        "--precontact_deformation_tolerance_mm", type=float, default=0.25
    )
    parser.add_argument("--minimum_object_lift_mm", type=float, default=20.0)
    parser.add_argument("--release_tail_frames", type=int, default=5)
    parser.add_argument("--fail_on_failure", action="store_true")
    return parser


def _load_array(directory: Path, filename: str) -> np.ndarray:
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required V6.2 array is missing: {path}")
    return np.load(path, allow_pickle=False)


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _timestamp() -> str:
    return datetime_module.datetime.now().astimezone().isoformat(
        sep=" ", timespec="milliseconds"
    )


def _maximum_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=np.float64)), initial=0.0))


def _maximum_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    first = np.asarray(actual, dtype=np.float64)
    second = np.asarray(expected, dtype=np.float64)
    if first.shape != second.shape:
        return math.inf
    return _maximum_abs(first - second)


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


def _pose_history_metrics(value: np.ndarray) -> dict[str, object]:
    poses = np.asarray(value, dtype=np.float64)
    correct_shape = poses.ndim == 2 and poses.shape[1] == 7
    finite = bool(correct_shape and np.all(np.isfinite(poses)))
    if not finite or poses.shape[0] == 0:
        return {
            "correct_shape": correct_shape,
            "finite": finite,
            "maximum_quaternion_norm_error": math.inf,
            "maximum_translation_step_m": math.inf,
        }
    quaternion_norm_error = np.abs(np.linalg.norm(poses[:, 3:7], axis=1) - 1.0)
    translation_step = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    return {
        "correct_shape": True,
        "finite": True,
        "maximum_quaternion_norm_error": float(
            np.max(quaternion_norm_error, initial=0.0)
        ),
        "maximum_translation_step_m": float(
            np.max(translation_step, initial=0.0)
        ),
    }


def validate_dataset(
    input_dir: Path,
    *,
    field_dir: Path | None,
    force_atol_tu: float,
    field_atol_tu: float,
    quaternion_norm_atol: float,
    release_tail_frames: int,
    penetration_tolerance_mm: float = 0.15,
    precontact_deformation_tolerance_mm: float = 0.25,
    minimum_object_lift_mm: float = 20.0,
) -> dict[str, object]:
    input_dir = Path(input_dir).expanduser().resolve()
    metadata = _load_json(input_dir / "metadata.json")
    frame_id = np.asarray(_load_array(input_dir, "frame_id.npy"), dtype=np.int64)
    motion_stage = np.asarray(_load_array(input_dir, "motion_stage.npy"), dtype=str)
    displacement = np.asarray(
        _load_array(input_dir, "surface_displacement_pad_local.npy"),
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
    gap_mm = np.asarray(
        _load_array(input_dir, "minimum_signed_gap_mm.npy"), dtype=np.float64
    ).reshape(-1)
    deformation_mm = np.asarray(
        _load_array(input_dir, "maximum_normal_deformation_mm.npy"),
        dtype=np.float64,
    ).reshape(-1)
    force_pad = np.asarray(
        _load_array(input_dir, "force_pad_local.npy"), dtype=np.float64
    )
    tactile = np.asarray(
        _load_array(input_dir, "tactile_force_channels.npy"), dtype=np.float64
    )
    object_pose = np.asarray(
        _load_array(input_dir, "object_pose_w.npy"), dtype=np.float64
    )
    object_pose_pad_local = np.asarray(
        _load_array(input_dir, "object_pose_pad_local.npy"), dtype=np.float64
    )
    object_pose_opposing_pad_local = np.asarray(
        _load_array(input_dir, "object_pose_opposing_pad_local.npy"),
        dtype=np.float64,
    )
    pad_pose = np.asarray(_load_array(input_dir, "pad_pose_w.npy"), dtype=np.float64)
    opposing_pad_pose = np.asarray(
        _load_array(input_dir, "opposing_pad_pose_w.npy"), dtype=np.float64
    )
    reaction_force = np.asarray(
        _load_array(input_dir, "uipc_reaction_force_w.npy"), dtype=np.float64
    )
    reaction_torque = np.asarray(
        _load_array(input_dir, "uipc_reaction_torque_w.npy"), dtype=np.float64
    )
    applied_force = np.asarray(
        _load_array(input_dir, "applied_uipc_force_w.npy"), dtype=np.float64
    )
    applied_torque = np.asarray(
        _load_array(input_dir, "applied_uipc_torque_w.npy"), dtype=np.float64
    )
    reaction_force_substeps = np.asarray(
        _load_array(input_dir, "uipc_reaction_force_substeps_w.npy"),
        dtype=np.float64,
    )
    reaction_torque_substeps = np.asarray(
        _load_array(input_dir, "uipc_reaction_torque_substeps_w.npy"),
        dtype=np.float64,
    )
    admissible_force_substeps = np.asarray(
        _load_array(input_dir, "uipc_admissible_force_substeps_w.npy"),
        dtype=np.float64,
    )
    admissible_torque_substeps = np.asarray(
        _load_array(input_dir, "uipc_admissible_torque_substeps_w.npy"),
        dtype=np.float64,
    )
    applied_force_substeps = np.asarray(
        _load_array(input_dir, "applied_uipc_force_substeps_w.npy"),
        dtype=np.float64,
    )
    applied_torque_substeps = np.asarray(
        _load_array(input_dir, "applied_uipc_torque_substeps_w.npy"),
        dtype=np.float64,
    )
    opposing_contact_force = np.asarray(
        _load_array(input_dir, "opposing_contact_force_w.npy"), dtype=np.float64
    )
    opposing_contact_force_substeps = np.asarray(
        _load_array(input_dir, "opposing_contact_force_substeps_w.npy"),
        dtype=np.float64,
    )
    backing_contact_force = np.asarray(
        _load_array(input_dir, "backing_contact_force_w.npy"), dtype=np.float64
    )
    backing_contact_force_substeps = np.asarray(
        _load_array(input_dir, "backing_contact_force_substeps_w.npy"),
        dtype=np.float64,
    )
    feedback_force_scale_substeps = np.asarray(
        _load_array(input_dir, "uipc_feedback_force_scale_substeps.npy"),
        dtype=np.float64,
    )
    feedback_torque_scale_substeps = np.asarray(
        _load_array(input_dir, "uipc_feedback_torque_scale_substeps.npy"),
        dtype=np.float64,
    )
    contact_cone_scale_substeps = np.asarray(
        _load_array(input_dir, "uipc_contact_cone_scale_substeps.npy"),
        dtype=np.float64,
    )
    boundary_surface_sync_error_mm = np.asarray(
        _load_array(input_dir, "uipc_boundary_surface_sync_error_mm.npy"),
        dtype=np.float64,
    ).reshape(-1)
    reaction_vertex_count = np.asarray(
        _load_array(input_dir, "uipc_reaction_vertex_count.npy"), dtype=np.int64
    ).reshape(-1)
    uipc_step_time = np.asarray(
        _load_array(input_dir, "uipc_step_time_sec.npy"), dtype=np.float64
    ).reshape(-1)
    frame_wall_time = np.asarray(
        _load_array(input_dir, "frame_wall_time_sec.npy"), dtype=np.float64
    ).reshape(-1)
    uipc_substep_time = np.asarray(
        _load_array(input_dir, "uipc_substep_time_sec.npy"), dtype=np.float64
    )

    if displacement.ndim != 3 or displacement.shape[2] != 3:
        raise ValueError("surface displacement must have shape [T,N,3]")
    frame_count, vertex_count, _ = displacement.shape
    if reaction_force_substeps.ndim != 3:
        raise ValueError("reaction substeps must have shape [T,S,3]")
    coupling_substeps = reaction_force_substeps.shape[1]
    expected_vector_shape = (frame_count, 3)
    expected_substep_vector_shape = (frame_count, coupling_substeps, 3)
    core_shapes = {
        "frame_id": frame_id.shape == (frame_count,),
        "motion_stage": motion_stage.shape == (frame_count,),
        "vertex_area": vertex_area.shape == (vertex_count,),
        "front_surface_mask": front_mask.shape == (vertex_count,),
        "contact_active": contact_active.shape == (frame_count,),
        "minimum_signed_gap_mm": gap_mm.shape == (frame_count,),
        "maximum_normal_deformation_mm": deformation_mm.shape == (frame_count,),
        "force_pad_local": force_pad.shape == expected_vector_shape,
        "tactile_force_channels": tactile.shape == expected_vector_shape,
        "object_pose_w": object_pose.shape == (frame_count, 7),
        "object_pose_pad_local": object_pose_pad_local.shape == (frame_count, 7),
        "object_pose_opposing_pad_local": object_pose_opposing_pad_local.shape
        == (frame_count, 7),
        "pad_pose_w": pad_pose.shape == (frame_count, 7),
        "opposing_pad_pose_w": opposing_pad_pose.shape == (frame_count, 7),
        "uipc_reaction_force_w": reaction_force.shape == expected_vector_shape,
        "uipc_reaction_torque_w": reaction_torque.shape == expected_vector_shape,
        "applied_uipc_force_w": applied_force.shape == expected_vector_shape,
        "applied_uipc_torque_w": applied_torque.shape == expected_vector_shape,
        "uipc_reaction_force_substeps_w": reaction_force_substeps.shape
        == expected_substep_vector_shape,
        "uipc_reaction_torque_substeps_w": reaction_torque_substeps.shape
        == expected_substep_vector_shape,
        "uipc_admissible_force_substeps_w": admissible_force_substeps.shape
        == expected_substep_vector_shape,
        "uipc_admissible_torque_substeps_w": admissible_torque_substeps.shape
        == expected_substep_vector_shape,
        "applied_uipc_force_substeps_w": applied_force_substeps.shape
        == expected_substep_vector_shape,
        "applied_uipc_torque_substeps_w": applied_torque_substeps.shape
        == expected_substep_vector_shape,
        "opposing_contact_force_w": opposing_contact_force.shape
        == expected_vector_shape,
        "opposing_contact_force_substeps_w": opposing_contact_force_substeps.shape
        == expected_substep_vector_shape,
        "backing_contact_force_w": backing_contact_force.shape
        == expected_vector_shape,
        "backing_contact_force_substeps_w": backing_contact_force_substeps.shape
        == expected_substep_vector_shape,
        "uipc_feedback_force_scale_substeps": feedback_force_scale_substeps.shape
        == (frame_count, coupling_substeps),
        "uipc_feedback_torque_scale_substeps": feedback_torque_scale_substeps.shape
        == (frame_count, coupling_substeps),
        "uipc_contact_cone_scale_substeps": contact_cone_scale_substeps.shape
        == (frame_count, coupling_substeps),
        "uipc_boundary_surface_sync_error_mm": boundary_surface_sync_error_mm.shape
        == (frame_count,),
        "uipc_reaction_vertex_count": reaction_vertex_count.shape == (frame_count,),
        "uipc_step_time_sec": uipc_step_time.shape == (frame_count,),
        "frame_wall_time_sec": frame_wall_time.shape == (frame_count,),
        "uipc_substep_time_sec": uipc_substep_time.shape
        == (frame_count, coupling_substeps),
    }
    if not all(core_shapes.values()):
        failed = [name for name, passed in core_shapes.items() if not passed]
        raise ValueError(f"V6.2 core array shape mismatch: {failed}")

    config = _estimator_config(metadata)
    result = frozen_7g.estimate_deformation_force(
        displacement, vertex_area, front_mask, config
    )
    expected_force_pad = np.asarray(result.force_pad_local_tu, dtype=np.float64)
    expected_tactile = np.asarray(
        result.tactile_force_channels_tu, dtype=np.float64
    )
    expected_force_pad[~contact_active] = 0.0
    expected_tactile[~contact_active] = 0.0

    force_error = _maximum_abs_error(force_pad, expected_force_pad)
    tactile_error = _maximum_abs_error(tactile, expected_tactile)
    inactive_force_max = _maximum_abs(force_pad[~contact_active])
    inactive_tactile_max = _maximum_abs(tactile[~contact_active])
    transform_expected = np.column_stack(
        (force_pad[:, 1], -force_pad[:, 2], -force_pad[:, 0])
    )
    transform_error = _maximum_abs_error(tactile, transform_expected)

    gate = metadata.get("contact_gate")
    if not isinstance(gate, dict):
        raise ValueError("metadata.json has no contact_gate configuration")
    reaction_force_threshold_n = float(gate["reaction_force_threshold_n"])
    expected_active = np.any(
        np.linalg.norm(reaction_force_substeps, axis=2)
        > reaction_force_threshold_n,
        axis=1,
    )
    gate_mismatch_count = int(np.count_nonzero(contact_active != expected_active))

    time_step = metadata.get("time_step")
    if not isinstance(time_step, dict):
        raise ValueError("metadata.json has no time_step configuration")
    record_dt = float(time_step["record_dt_sec"])
    physx_dt = float(time_step["physx_step_dt_sec"])
    uipc_dt = float(time_step["uipc_step_dt_sec"])
    metadata_substeps = int(time_step["coupling_substeps_per_record"])
    feedback_relaxation = float(time_step["reaction_feedback_relaxation"])
    feedback_force_limit_n = float(time_step["reaction_feedback_force_limit_n"])
    feedback_torque_limit_nm = float(time_step["reaction_feedback_torque_limit_nm"])
    slow_frame_threshold = float(time_step["slow_frame_threshold_sec"])
    feedback_relaxation_valid = 0.0 < feedback_relaxation <= 1.0
    time_step_error = max(
        abs(record_dt - metadata_substeps * physx_dt),
        abs(physx_dt - uipc_dt),
    )

    flattened_reaction_force = reaction_force_substeps.reshape(-1, 3)
    flattened_reaction_torque = reaction_torque_substeps.reshape(-1, 3)
    flattened_admissible_force = admissible_force_substeps.reshape(-1, 3)
    flattened_admissible_torque = admissible_torque_substeps.reshape(-1, 3)
    flattened_applied_force = applied_force_substeps.reshape(-1, 3)
    flattened_applied_torque = applied_torque_substeps.reshape(-1, 3)
    flattened_force_scale = feedback_force_scale_substeps.reshape(-1)
    flattened_torque_scale = feedback_torque_scale_substeps.reshape(-1)
    flattened_cone_scale = contact_cone_scale_substeps.reshape(-1)
    relaxed_force = flattened_applied_force + feedback_relaxation * (
        flattened_admissible_force - flattened_applied_force
    )
    relaxed_torque = flattened_applied_torque + feedback_relaxation * (
        flattened_admissible_torque - flattened_applied_torque
    )
    relaxed_force_norm = np.linalg.norm(relaxed_force, axis=1)
    relaxed_torque_norm = np.linalg.norm(relaxed_torque, axis=1)
    expected_force_scale = np.minimum(
        1.0, feedback_force_limit_n / np.maximum(relaxed_force_norm, 1.0e-12)
    )
    expected_torque_scale = np.minimum(
        1.0, feedback_torque_limit_nm / np.maximum(relaxed_torque_norm, 1.0e-12)
    )
    expected_limited_force = relaxed_force * expected_force_scale[:, None]
    expected_limited_torque = relaxed_torque * expected_torque_scale[:, None]
    feedback_force_error = _maximum_abs_error(
        flattened_applied_force[1:], expected_limited_force[:-1]
    )
    feedback_torque_error = _maximum_abs_error(
        flattened_applied_torque[1:], expected_limited_torque[:-1]
    )
    feedback_force_scale_error = _maximum_abs_error(
        flattened_force_scale, expected_force_scale
    )
    feedback_torque_scale_error = _maximum_abs_error(
        flattened_torque_scale, expected_torque_scale
    )
    raw_force_norm = np.linalg.norm(flattened_reaction_force, axis=1)
    admissible_force_norm = np.linalg.norm(flattened_admissible_force, axis=1)
    expected_cone_scale = np.ones_like(raw_force_norm)
    nonzero_raw_force = raw_force_norm > 1.0e-12
    expected_cone_scale[nonzero_raw_force] = np.minimum(
        1.0,
        admissible_force_norm[nonzero_raw_force]
        / raw_force_norm[nonzero_raw_force],
    )
    contact_cone_scale_error = _maximum_abs_error(
        flattened_cone_scale, expected_cone_scale
    )
    admissible_torque_error = _maximum_abs_error(
        flattened_admissible_torque,
        flattened_reaction_torque * flattened_cone_scale[:, None],
    )
    reaction_force_mean_error = _maximum_abs_error(
        reaction_force, np.mean(reaction_force_substeps, axis=1)
    )
    reaction_torque_mean_error = _maximum_abs_error(
        reaction_torque, np.mean(reaction_torque_substeps, axis=1)
    )
    applied_force_mean_error = _maximum_abs_error(
        applied_force, np.mean(applied_force_substeps, axis=1)
    )
    applied_torque_mean_error = _maximum_abs_error(
        applied_torque, np.mean(applied_torque_substeps, axis=1)
    )
    opposing_contact_force_mean_error = _maximum_abs_error(
        opposing_contact_force,
        np.mean(opposing_contact_force_substeps, axis=1),
    )
    backing_contact_force_mean_error = _maximum_abs_error(
        backing_contact_force,
        np.mean(backing_contact_force_substeps, axis=1),
    )
    uipc_substep_sum_error = _maximum_abs_error(
        uipc_step_time, np.sum(uipc_substep_time, axis=1, dtype=np.float64)
    )
    slow_frame_indices = np.flatnonzero(frame_wall_time > slow_frame_threshold)

    object_pose_metrics = _pose_history_metrics(object_pose)
    object_pose_pad_local_metrics = _pose_history_metrics(object_pose_pad_local)
    object_pose_opposing_pad_local_metrics = _pose_history_metrics(
        object_pose_opposing_pad_local
    )
    pad_pose_metrics = _pose_history_metrics(pad_pose)
    opposing_pad_pose_metrics = _pose_history_metrics(opposing_pad_pose)
    active_indices = np.flatnonzero(contact_active)
    first_contact_frame = int(active_indices[0]) if active_indices.size else frame_count
    precontact_deformation_max_mm = float(
        np.max(deformation_mm[:first_contact_frame], initial=0.0)
    )
    tail_count = min(int(release_tail_frames), frame_count)
    tail_contact_max = int(np.count_nonzero(contact_active[-tail_count:])) if tail_count else 0
    tail_force_max = _maximum_abs(tactile[-tail_count:]) if tail_count else math.inf
    tail_stage = motion_stage[-tail_count:].tolist() if tail_count else []
    prelift_hold_mask = motion_stage == "hold"
    lifted_stage_mask = np.isin(motion_stage, ("lift", "hold_lifted"))
    if np.any(prelift_hold_mask) and np.any(lifted_stage_mask):
        object_prelift_z_m = float(np.median(object_pose[prelift_hold_mask, 2]))
        maximum_object_lift_mm: float | None = float(
            (np.max(object_pose[lifted_stage_mask, 2]) - object_prelift_z_m) * 1000.0
        )
    else:
        # A deliberately truncated diagnostic run can end before the lift phase.
        # Keep the missing measurement JSON-safe while letting the acceptance
        # check below report that task success has not yet been demonstrated.
        maximum_object_lift_mm = None

    completed_frames = int(metadata.get("completed_frame_count", -1))
    planned_frames = int(metadata.get("planned_frame_count", -1))
    termination_reason = str(metadata.get("termination_reason", ""))
    core_finite = bool(
        np.all(np.isfinite(displacement))
        and np.all(np.isfinite(vertex_area))
        and np.all(np.isfinite(gap_mm))
        and np.all(np.isfinite(deformation_mm))
        and np.all(np.isfinite(force_pad))
        and np.all(np.isfinite(tactile))
        and np.all(np.isfinite(reaction_force))
        and np.all(np.isfinite(reaction_torque))
        and np.all(np.isfinite(applied_force))
        and np.all(np.isfinite(applied_torque))
        and np.all(np.isfinite(reaction_force_substeps))
        and np.all(np.isfinite(reaction_torque_substeps))
        and np.all(np.isfinite(admissible_force_substeps))
        and np.all(np.isfinite(admissible_torque_substeps))
        and np.all(np.isfinite(applied_force_substeps))
        and np.all(np.isfinite(applied_torque_substeps))
        and np.all(np.isfinite(opposing_contact_force))
        and np.all(np.isfinite(opposing_contact_force_substeps))
        and np.all(np.isfinite(backing_contact_force))
        and np.all(np.isfinite(backing_contact_force_substeps))
        and np.all(np.isfinite(feedback_force_scale_substeps))
        and np.all(np.isfinite(feedback_torque_scale_substeps))
        and np.all(np.isfinite(contact_cone_scale_substeps))
        and np.all(np.isfinite(boundary_surface_sync_error_mm))
        and np.all(np.isfinite(uipc_step_time))
        and np.all(np.isfinite(frame_wall_time))
        and np.all(np.isfinite(uipc_substep_time))
    )
    authority = metadata.get("physx_object_authority")
    coupling = metadata.get("uipc_coupling")
    opposing_contact = metadata.get("opposing_contact")
    membrane_backing = metadata.get("membrane_rigid_backing")
    grasp_centering = metadata.get("grasp_centering")
    grasp_orientation = metadata.get("grasp_orientation")
    lift_motion = metadata.get("lift_motion")
    initial_reconstruction_error_m = (
        float(coupling.get("initial_world_reconstruction_error_m", math.inf))
        if isinstance(coupling, dict)
        else math.inf
    )

    checks: dict[str, bool] = {
        "dataset_has_frames": frame_count > 0,
        "core_arrays_are_finite": core_finite,
        "frame_ids_are_contiguous": bool(
            np.array_equal(frame_id, np.arange(frame_count, dtype=np.int64))
        ),
        "metadata_frame_count_matches": completed_frames == frame_count,
        "run_completed_planned_motion": frame_count == planned_frames
        and termination_reason == "completed",
        "physx_uipc_time_steps_match": time_step_error <= 1.0e-15
        and metadata_substeps == coupling_substeps
        and coupling_substeps >= 8,
        "physx_object_pose_is_not_written_during_formal_motion": bool(
            isinstance(authority, dict)
            and int(authority.get("formal_motion_pose_write_count", -1)) == 0
        ),
        "single_link8_solver_membrane_declared": bool(
            isinstance(coupling, dict)
            and int(coupling.get("solver_membrane_count", -1)) == 1
            and str(coupling.get("solver_membrane_path", "")).endswith(
                "/Robot/link8/UIPC_Pad/simulation/membrane_sim_mesh"
            )
            and coupling.get("link7_uipc_representation") is False
        ),
        "link7_has_rigid_opposing_pad": bool(
            isinstance(opposing_contact, dict)
            and str(opposing_contact.get("path", "")).endswith(
                "/Robot/openworldtactile_case_left/openworldtactile_pad_visual"
            )
            and opposing_contact.get("mounted_body") == "link7"
            and str(opposing_contact.get("contact_report_body", "")).endswith(
                "/Robot/openworldtactile_case_left"
            )
            and opposing_contact.get("fixed_joint_parent") == "link7"
            and opposing_contact.get("representation")
            == "PhysX_rigid_cube_collider"
            and opposing_contact.get("collision_enabled") is True
            and opposing_contact.get("filtered_contact_force")
            == "GraspCylinder_only"
            and opposing_contact.get("force_substep_array")
            == "opposing_contact_force_substeps_w.npy"
        ),
        "link8_has_physical_membrane_backing": bool(
            isinstance(membrane_backing, dict)
            and membrane_backing.get("mounted_body") == "link8"
            and membrane_backing.get("collision_enabled") is True
            and membrane_backing.get("representation")
            == "authored_box_mesh_convex_hull_collider"
            and membrane_backing.get("force_substep_array")
            == "backing_contact_force_substeps_w.npy"
        ),
        "grasp_target_is_centered_between_both_contact_faces": bool(
            isinstance(grasp_centering, dict)
            and grasp_centering.get("target_reference")
            == (
                "midpoint_of_open_link8_membrane_front_and_"
                "link7_rigid_pad_face"
            )
            and grasp_centering.get("former_one_sided_target_removed") is True
            and math.isfinite(
                float(grasp_centering.get("open_contact_face_separation_mm", math.nan))
            )
            and math.isfinite(
                float(
                    grasp_centering.get(
                        "predicted_closed_contact_face_separation_mm", math.nan
                    )
                )
            )
        ),
        "lift_motion_is_contact_resolved": bool(
            isinstance(lift_motion, dict)
            and lift_motion.get("interpolation") == "smoothstep"
            and int(lift_motion.get("frames", 0)) >= 240
            and float(
                lift_motion.get(
                    "nominal_peak_command_increment_per_coupling_substep_mm",
                    math.inf,
                )
            )
            <= 0.05
        ),
        "grasp_orientation_keeps_cylinder_axis_in_membrane_plane": bool(
            isinstance(grasp_orientation, dict)
            and grasp_orientation.get("controller_command_type") == "pose"
            and float(
                grasp_orientation.get(
                    "pad_normal_to_horizontal_error_deg", math.inf
                )
            )
            <= 0.1
        ),
        "uipc_solver_is_fixed_in_link8_pad_local_frame": bool(
            isinstance(coupling, dict)
            and coupling.get("coordinate_frame") == "link8_pad_local"
            and coupling.get("membrane_back_face") == "fixed_pad_local_rest_targets"
            and coupling.get("membrane_front_face") == "uipc_solved_in_pad_local_frame"
        ),
        "initial_pad_membrane_object_placement_is_preserved": bool(
            isinstance(coupling, dict)
            and coupling.get("initial_relative_placement_preserved") is True
            and math.isfinite(initial_reconstruction_error_m)
            and initial_reconstruction_error_m <= 1.0e-8
        ),
        "uipc_boundary_has_no_independent_dynamics": bool(
            isinstance(coupling, dict)
            and coupling.get("external_boundary_has_independent_dynamics") is False
        ),
        "uipc_reaction_is_applied_to_object_and_link8": bool(
            isinstance(coupling, dict)
            and coupling.get("physx_reaction_recipients")
            == "object_wrench_and_equal_opposite_link8_wrench"
            and coupling.get("link8_reaction_moment_transfer")
            == "-object_torque-(object_com-link8_com)x_object_force"
        ),
        "uipc_contact_cone_and_limited_feedback_reconstruct_exactly": bool(
            flattened_reaction_force.shape[0] > 1
            and feedback_relaxation_valid
            and feedback_force_error <= 1.0e-12
            and feedback_torque_error <= 1.0e-12
            and feedback_force_scale_error <= 1.0e-12
            and feedback_torque_scale_error <= 1.0e-12
            and contact_cone_scale_error <= 1.0e-12
            and admissible_torque_error <= 1.0e-12
            and np.all(admissible_force_norm <= raw_force_norm + 1.0e-12)
            and np.all((flattened_cone_scale >= 0.0) & (flattened_cone_scale <= 1.0))
        ),
        "record_wrenches_are_substep_time_averages": bool(
            reaction_force_mean_error <= 1.0e-12
            and reaction_torque_mean_error <= 1.0e-12
            and applied_force_mean_error <= 1.0e-12
            and applied_torque_mean_error <= 1.0e-12
            and opposing_contact_force_mean_error <= 1.0e-12
            and backing_contact_force_mean_error <= 1.0e-12
        ),
        "uipc_boundary_surface_tracks_physx_pose": bool(
            np.max(boundary_surface_sync_error_mm, initial=0.0) <= 1.0e-3
        ),
        "uipc_step_times_are_nonnegative": bool(np.all(uipc_step_time >= 0.0)),
        "slow_frame_diagnostics_are_declared": bool(
            slow_frame_threshold > 0.0
            and time_step.get("slow_frame_is_failure") is False
            and time_step.get("slow_frame_action")
            == "continue_and_record_substep_diagnostics"
            and uipc_substep_sum_error <= 1.0e-12
            and np.all(uipc_substep_time >= 0.0)
        ),
        "object_pose_input_history_is_valid": bool(
            object_pose_metrics["correct_shape"]
            and object_pose_metrics["finite"]
            and float(object_pose_metrics["maximum_quaternion_norm_error"])
            <= float(quaternion_norm_atol)
        ),
        "object_relative_to_pad_pose_input_history_is_valid": bool(
            object_pose_pad_local_metrics["correct_shape"]
            and object_pose_pad_local_metrics["finite"]
            and float(object_pose_pad_local_metrics["maximum_quaternion_norm_error"])
            <= float(quaternion_norm_atol)
        ),
        "object_relative_to_opposing_pad_pose_history_is_valid": bool(
            object_pose_opposing_pad_local_metrics["correct_shape"]
            and object_pose_opposing_pad_local_metrics["finite"]
            and float(
                object_pose_opposing_pad_local_metrics[
                    "maximum_quaternion_norm_error"
                ]
            )
            <= float(quaternion_norm_atol)
        ),
        "pad_world_pose_history_is_valid": bool(
            pad_pose_metrics["correct_shape"]
            and pad_pose_metrics["finite"]
            and float(pad_pose_metrics["maximum_quaternion_norm_error"])
            <= float(quaternion_norm_atol)
        ),
        "opposing_pad_world_pose_history_is_valid": bool(
            opposing_pad_pose_metrics["correct_shape"]
            and opposing_pad_pose_metrics["finite"]
            and float(opposing_pad_pose_metrics["maximum_quaternion_norm_error"])
            <= float(quaternion_norm_atol)
        ),
        "contact_gate_reconstructs_exactly": gate_mismatch_count == 0,
        "contact_has_native_boundary_vertices": bool(
            np.all(reaction_vertex_count[contact_active] > 0)
        ),
        "penetration_is_within_tolerance": bool(
            np.min(gap_mm, initial=math.inf) >= -float(penetration_tolerance_mm)
        ),
        "precontact_pad_motion_does_not_deform_membrane": bool(
            first_contact_frame > 0
            and precontact_deformation_max_mm
            <= float(precontact_deformation_tolerance_mm)
        ),
        "dataset_contains_inactive_frames": bool(np.any(~contact_active)),
        "dataset_contains_active_contact": bool(np.any(contact_active)),
        "object_is_lifted_by_gripper": bool(
            maximum_object_lift_mm is not None
            and math.isfinite(maximum_object_lift_mm)
            and maximum_object_lift_mm >= float(minimum_object_lift_mm)
        ),
        "inactive_pad_force_is_exact_zero": inactive_force_max == 0.0,
        "inactive_tactile_force_is_exact_zero": inactive_tactile_max == 0.0,
        "frozen_7g_pad_force_reconstructs": force_error <= float(force_atol_tu),
        "frozen_7g_tactile_force_reconstructs": tactile_error
        <= float(force_atol_tu),
        "pad_to_tactile_transform_is_exact": transform_error
        <= float(force_atol_tu),
        "release_tail_contact_is_inactive": tail_count > 0 and tail_contact_max == 0,
        "release_tail_force_returns_to_zero": tail_count > 0
        and tail_force_max <= float(force_atol_tu),
    }

    field_observed: dict[str, object] = {"available": False}
    if field_dir is not None:
        selected_field_dir = Path(field_dir).expanduser().resolve()
        field = np.asarray(
            _load_array(selected_field_dir, "tactile_force_field.npy"),
            dtype=np.float64,
        )
        correct_field_shape = (
            field.ndim == 4
            and field.shape[0] == frame_count
            and field.shape[3] == 3
        )
        field_total = (
            np.sum(field, axis=(1, 2), dtype=np.float64)
            if correct_field_shape
            else np.empty((0, 3), dtype=np.float64)
        )
        field_error = _maximum_abs_error(field_total, tactile)
        inactive_field_max = (
            _maximum_abs(field[~contact_active]) if correct_field_shape else math.inf
        )
        field_metadata = _load_json(selected_field_dir / "metadata.json")
        video_metadata = field_metadata.get("video")
        videos_decode = bool(
            isinstance(video_metadata, dict)
            and video_metadata.get("all_videos_decode_with_expected_frame_count")
        )
        videos_exist = all((selected_field_dir / name).is_file() for name in VIDEO_NAMES)
        checks.update(
            {
                "offline_field_shape_is_valid": correct_field_shape,
                "offline_field_conserves_tactile_force": field_error
                <= float(field_atol_tu),
                "inactive_offline_field_is_exact_zero": inactive_field_max == 0.0,
                "offline_force_videos_exist": videos_exist,
                "offline_force_videos_decode_completely": videos_decode,
            }
        )
        field_observed = {
            "available": True,
            "directory": str(selected_field_dir),
            "shape": list(field.shape),
            "maximum_conservation_error_tu": field_error,
            "maximum_inactive_value_tu": inactive_field_max,
            "videos_exist": videos_exist,
            "videos_decode_completely": videos_decode,
        }

    observed = {
        "frame_count": frame_count,
        "active_frame_count": int(np.count_nonzero(contact_active)),
        "inactive_frame_count": int(np.count_nonzero(~contact_active)),
        "termination_reason": termination_reason,
        "maximum_7g_pad_reconstruction_error_tu": force_error,
        "maximum_7g_tactile_reconstruction_error_tu": tactile_error,
        "maximum_pad_to_tactile_transform_error_tu": transform_error,
        "maximum_inactive_pad_force_tu": inactive_force_max,
        "maximum_inactive_tactile_force_tu": inactive_tactile_max,
        "contact_gate_mismatch_count": gate_mismatch_count,
        "reaction_feedback_relaxation": feedback_relaxation,
        "maximum_reaction_feedback_recurrence_force_error_n": feedback_force_error,
        "maximum_reaction_feedback_recurrence_torque_error_nm": feedback_torque_error,
        "maximum_feedback_force_scale_error": feedback_force_scale_error,
        "maximum_feedback_torque_scale_error": feedback_torque_scale_error,
        "maximum_contact_cone_scale_error": contact_cone_scale_error,
        "maximum_admissible_torque_scale_error_nm": admissible_torque_error,
        "contact_cone_projected_substep_count": int(
            np.count_nonzero(
                (flattened_cone_scale < 1.0 - 1.0e-12) & nonzero_raw_force
            )
        ),
        "force_limited_substep_count": int(
            np.count_nonzero(flattened_force_scale < 1.0 - 1.0e-12)
        ),
        "maximum_record_reaction_force_mean_error_n": reaction_force_mean_error,
        "maximum_record_reaction_torque_mean_error_nm": reaction_torque_mean_error,
        "maximum_record_applied_force_mean_error_n": applied_force_mean_error,
        "maximum_record_applied_torque_mean_error_nm": applied_torque_mean_error,
        "maximum_record_opposing_contact_force_mean_error_n": (
            opposing_contact_force_mean_error
        ),
        "maximum_record_backing_contact_force_mean_error_n": (
            backing_contact_force_mean_error
        ),
        "maximum_opposing_contact_force_n": float(
            np.max(
                np.linalg.norm(opposing_contact_force_substeps, axis=2),
                initial=0.0,
            )
        ),
        "minimum_signed_gap_mm": float(np.min(gap_mm, initial=math.inf)),
        "first_contact_frame": first_contact_frame,
        "maximum_precontact_deformation_mm": precontact_deformation_max_mm,
        "initial_world_reconstruction_error_m": initial_reconstruction_error_m,
        "maximum_boundary_surface_sync_error_mm": float(
            np.max(boundary_surface_sync_error_mm, initial=0.0)
        ),
        "maximum_uipc_step_time_sec": float(
            np.max(uipc_step_time, initial=0.0)
        ),
        "maximum_frame_wall_time_sec": float(
            np.max(frame_wall_time, initial=0.0)
        ),
        "slow_frame_threshold_sec": slow_frame_threshold,
        "slow_frame_count": int(slow_frame_indices.size),
        "slow_frame_indices": slow_frame_indices.tolist(),
        "maximum_uipc_substep_time_sec": float(
            np.max(uipc_substep_time, initial=0.0)
        ),
        "maximum_uipc_substep_sum_error_sec": uipc_substep_sum_error,
        "maximum_object_lift_mm": maximum_object_lift_mm,
        "time_step_contract_error_sec": time_step_error,
        "object_pose_input": object_pose_metrics,
        "object_pose_pad_local_input": object_pose_pad_local_metrics,
        "object_pose_opposing_pad_local_input": (
            object_pose_opposing_pad_local_metrics
        ),
        "pad_world_pose_input": pad_pose_metrics,
        "opposing_pad_world_pose_input": opposing_pad_pose_metrics,
        "grasp_centering": grasp_centering,
        "lift_motion": lift_motion,
        "release_tail_frame_count": tail_count,
        "release_tail_stages": tail_stage,
        "release_tail_active_frame_count": tail_contact_max,
        "release_tail_maximum_abs_tactile_force_tu": tail_force_max,
        "offline_field": field_observed,
        "synchronization_scope": (
            "Validates the PhysX authoritative pose history, fixed link8 Pad-local "
            "membrane, relative kinematic collision boundary, and one-frame native UIPC "
            "reaction feedback into the same PhysX rigid body."
        ),
    }
    return {
        "version": VERSION,
        "input_directory": str(input_dir),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": observed,
        "acceptance": {
            "force_atol_tu": float(force_atol_tu),
            "field_atol_tu": float(field_atol_tu),
            "quaternion_norm_atol": float(quaternion_norm_atol),
            "penetration_tolerance_mm": float(penetration_tolerance_mm),
            "precontact_deformation_tolerance_mm": float(
                precontact_deformation_tolerance_mm
            ),
            "minimum_object_lift_mm": float(minimum_object_lift_mm),
            "release_tail_frames": int(release_tail_frames),
        },
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    for name in (
        "force_atol_tu",
        "field_atol_tu",
        "quaternion_norm_atol",
        "penetration_tolerance_mm",
        "precontact_deformation_tolerance_mm",
        "minimum_object_lift_mm",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    if int(args.release_tail_frames) <= 0:
        parser.error("--release_tail_frames must be > 0")

    input_dir = Path(args.input_dir).expanduser().resolve()
    if str(args.field_dir).strip():
        field_dir: Path | None = Path(args.field_dir).expanduser().resolve()
    else:
        default_field_dir = input_dir / "offline_tactile_field"
        field_dir = default_field_dir if default_field_dir.is_dir() else None
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if str(args.output_json).strip()
        else input_dir / "v6_2_validation.json"
    )
    verdict = validate_dataset(
        input_dir,
        field_dir=field_dir,
        force_atol_tu=float(args.force_atol_tu),
        field_atol_tu=float(args.field_atol_tu),
        quaternion_norm_atol=float(args.quaternion_norm_atol),
        penetration_tolerance_mm=float(args.penetration_tolerance_mm),
        precontact_deformation_tolerance_mm=float(
            args.precontact_deformation_tolerance_mm
        ),
        minimum_object_lift_mm=float(args.minimum_object_lift_mm),
        release_tail_frames=int(args.release_tail_frames),
    )
    _atomic_json(output_json, verdict)
    failed = [name for name, passed in verdict["checks"].items() if not passed]
    print(
        f"[{_timestamp()}] [V62_VALIDATE] passed={verdict['passed']} failed={failed} "
        f"output={output_json}",
        flush=True,
    )
    if bool(args.fail_on_failure) and not bool(verdict["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
