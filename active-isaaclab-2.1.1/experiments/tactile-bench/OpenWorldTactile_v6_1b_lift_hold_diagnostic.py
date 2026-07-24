from __future__ import annotations

"""V6.1c contact-timing short grasp/lift/hold diagnostic on frozen 7f -> 7g -> v9.

PhysX is the sole authority for the robot and free cylinder.  The cylinder pose is
mirrored, one way, to an affine UIPC contact body before every UIPC solve.  The
only tactile source is the rest-relative deformation of the link8-mounted
``simulation/membrane_sim_mesh``; contact geometry is diagnostic only.

This file intentionally imports the passed v9 executable as a read-only runtime
library.  Its AppLauncher owns the Isaac application, while this module supplies
only the IK, free-rigid-body, phase-state, synchronization, and diagnostic
logic.  New v6.1c-only command-line options are removed before v9 parses the shared
IsaacLab and frozen-mainline options.

The formal path intentionally stops after twenty HOLD_LIFTED frames.  PhysX
advances one frame at a time while UIPC advances three matching substeps.
The pre-contact target keeps the membrane outside the UIPC contact activation
band, then freezes before the gripper closes.  Lightweight progress snapshots
are periodic; complete arrays are saved at phase boundaries, termination, and
normal completion.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np


_v6_parser = argparse.ArgumentParser(add_help=False)
_v6_parser.add_argument("--object_radius_mm", type=float, default=15.0)
_v6_parser.add_argument("--object_height_mm", type=float, default=105.0)
_v6_parser.add_argument("--object_mass_kg", type=float, default=0.018)
_v6_parser.add_argument("--object_x", type=float, default=0.34)
_v6_parser.add_argument("--object_y", type=float, default=-0.02)
_v6_parser.add_argument("--object_support_clearance_mm", type=float, default=0.25)
_v6_parser.add_argument("--object_friction", type=float, default=1.0)
_v6_parser.add_argument("--gripper_closed_mm", type=float, default=14.0)
_v6_parser.add_argument("--home_ee_x", type=float, default=0.28)
_v6_parser.add_argument("--home_ee_y", type=float, default=0.0)
_v6_parser.add_argument("--home_ee_z", type=float, default=0.20)
_v6_parser.add_argument("--approach_z", type=float, default=0.20)
_v6_parser.add_argument("--grasp_z_offset", type=float, default=0.020)
_v6_parser.add_argument("--grasp_forward_offset", type=float, default=0.020)
_v6_parser.add_argument("--lift_distance_mm", type=float, default=40.0)
_v6_parser.add_argument("--piper_gripper_body", type=str, default="gripper_base")
_v6_parser.add_argument("--piper_tip_offset", type=float, nargs=3, default=[0.0, 0.0, 0.1358])
_v6_parser.add_argument("--initial_settle_frames", type=int, default=90)
_v6_parser.add_argument("--init_unrecorded_max_frames", type=int, default=180)
_v6_parser.add_argument("--episode_index", type=int, default=0)
_v6_parser.add_argument("--accept_max_sync_position_error_mm", type=float, default=0.05)
_v6_parser.add_argument("--accept_max_sync_orientation_error_deg", type=float, default=0.05)
_v6_parser.add_argument("--accept_lift_height_mm", type=float, default=30.0)
_v6_parser.add_argument("--accept_gripper_distance_mm", type=float, default=80.0)
_v6_parser.add_argument("--accept_min_vertical_velocity_m_s", type=float, default=-0.02)
_v6_parser.add_argument("--accept_max_hold_position_drift_mm", type=float, default=2.0)
_v6_parser.add_argument("--accept_max_hold_orientation_drift_deg", type=float, default=5.0)
_v6_parser.add_argument("--accept_max_hold_field_centroid_drift_cells", type=float, default=2.0)
_v6_parser.add_argument("--accept_min_hold_field_correlation", type=float, default=0.95)
_v6_parser.add_argument("--support_height_m", type=float, default=0.0)
_v6_parser.add_argument("--support_contact_tolerance_mm", type=float, default=1.0)
_v6_parser.add_argument("--terminate_max_penetration_mm", type=float, default=0.15)
_v6_parser.add_argument("--terminate_obvious_slip_mm", type=float, default=5.0)
_v6_parser.add_argument("--uipc_substep_timeout_sec", type=float, default=300.0)
_v6_parser.add_argument("--physx_contact_force_epsilon_n", type=float, default=1.0e-6)
_v6_parser.add_argument("--max_grasp_centering_frames", type=int, default=180)
_v6_parser.add_argument(
    "--save_diagnostic_video",
    action="store_true",
    help="Render and encode the diagnostic scene video (disabled by default for timing runs).",
)
_v6_parser.add_argument(
    "--save_tactile_force_video",
    action="store_true",
    help="Encode Fx/Fy/Fz, shear, and composite tactile videos after the run.",
)
_v6_args, _v9_argv = _v6_parser.parse_known_args()

# The passed v9 module creates its AppLauncher during import and derives its
# camera enablement from --save_camera_rgb.  This diagnostic has an explicit,
# opt-in video flag, so force the inherited setting to match it before v9
# parses its command line.
_v9_argv = [
    value
    for value in _v9_argv
    if value not in ("--save_camera_rgb", "--no_save_camera_rgb")
]
_v9_argv.append(
    "--save_camera_rgb" if bool(_v6_args.save_diagnostic_video) else "--no_save_camera_rgb"
)

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0], *_v9_argv]
import OpenWorldTactile_v5_new_9_tu_tactile_field_rendering as mainline_v9
sys.argv = _original_argv

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from pxr import UsdGeom


args_cli = mainline_v9.args_cli
simulation_app = mainline_v9.simulation_app
frozen_7g = mainline_v9.frozen_7g
tactile_field_v9 = mainline_v9.tactile_field_v9

OBJECT_PATH = "/World/envs/env_0/GraspCylinder"
UIPC_OBJECT_ROOT = "/World/UIPC_v6_FreeObjectMirror"
UIPC_OBJECT_MESH_PATH = f"{UIPC_OBJECT_ROOT}/mesh"
SCENE_CAMERA_PATH = "/World/V6SceneCamera"
EPS = 1.0e-12
RUN_UUID = uuid.uuid4().hex
UIPC_SUBSTEPS = 3
PHASE_SPECS = (
    ("SETTLE_READY", 15),
    ("APPROACH_PICK", 30),
    ("LOWER_TO_GRASP", 50),
    ("CLOSE_GRIPPER", 50),
    ("CONFIRM_GRASP", 30),
    ("LIFT_OBJECT", 60),
    ("HOLD_LIFTED", 20),
)
SHORT_FRAME_COUNT = sum(count for _, count in PHASE_SPECS)
PHASE_TO_ID = {name: index for index, (name, _) in enumerate(PHASE_SPECS)}
LEFT_FINGER_PATH = f"{mainline_v9.ROBOT_ROOT}/link7"
RIGHT_FINGER_PATH = f"{mainline_v9.ROBOT_ROOT}/link8"
PAD_FINGER_PATH = RIGHT_FINGER_PATH
GRASP_CENTER_TOLERANCE_MM = 1.0
GRASP_CENTER_STABLE_FRAMES = 3
LIGHTWEIGHT_CHECKPOINT_INTERVAL_FRAMES = 10


class DiagnosticTermination(RuntimeError):
    """Expected fail-fast termination after all completed frames are persisted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_frozen_hashes() -> dict[str, str]:
    paths = list(tactile_field_v9.frozen_source_paths())
    paths.append(Path(tactile_field_v9.__file__).resolve())
    paths.append(Path(args_cli.asset_usd).expanduser().resolve())
    return {str(path): _sha256(path) for path in paths}


FROZEN_HASHES_BEFORE = _capture_frozen_hashes()


def _validate_v6_args() -> None:
    for name in (
        "object_radius_mm",
        "object_height_mm",
        "object_mass_kg",
        "object_friction",
        "lift_distance_mm",
        "accept_max_sync_position_error_mm",
        "accept_max_sync_orientation_error_deg",
        "accept_lift_height_mm",
        "accept_gripper_distance_mm",
        "accept_max_hold_position_drift_mm",
        "accept_max_hold_orientation_drift_deg",
        "accept_max_hold_field_centroid_drift_cells",
        "accept_min_hold_field_correlation",
        "support_contact_tolerance_mm",
        "terminate_max_penetration_mm",
        "terminate_obvious_slip_mm",
        "uipc_substep_timeout_sec",
        "physx_contact_force_epsilon_n",
    ):
        if float(getattr(_v6_args, name)) <= 0.0:
            raise ValueError(f"--{name} must be > 0")
    if not 0.0 <= float(_v6_args.gripper_closed_mm) < float(args_cli.gripper_opening_mm):
        raise ValueError("Require 0 <= --gripper_closed_mm < --gripper_opening_mm")
    if len(_v6_args.piper_tip_offset) != 3:
        raise ValueError("--piper_tip_offset requires three values")
    for name in (
        "initial_settle_frames",
        "init_unrecorded_max_frames",
    ):
        if int(getattr(_v6_args, name)) <= 0:
            raise ValueError(f"--{name} must be > 0")
    if int(_v6_args.init_unrecorded_max_frames) != 180:
        raise ValueError("v6.1c freezes INIT_UNRECORDED to a maximum of 180 frames")
    if int(args_cli.warmup_stability_frames) != 30:
        raise ValueError("v6.1c freezes the readiness window at 30 consecutive frames")
    if int(args_cli.uipc_substeps_per_record) != UIPC_SUBSTEPS:
        raise ValueError("v6.1c requires exactly three UIPC substeps per PhysX frame")
    if int(args_cli.membrane_cells_x) != 1 or int(args_cli.membrane_cells_y) != 22 or int(args_cli.membrane_cells_z) != 26:
        raise ValueError("v6.1c freezes the structured membrane at 1x22x26 cells")
    if str(args_cli.membrane_mesh_mode) != "structured":
        raise ValueError("v6.1c requires the frozen structured membrane")
    if not math.isclose(float(args_cli.youngs_modulus_mpa), 0.05) or not math.isclose(float(args_cli.poisson_rate), 0.49) or not math.isclose(float(args_cli.mass_density), 1050.0):
        raise ValueError("v6.1c freezes membrane material at E=0.05 MPa, nu=0.49, density=1050")
    if int(args_cli.field_height) != 81 or int(args_cli.field_width) != 65 or not math.isclose(float(args_cli.gaussian_sigma_cells), 1.25):
        raise ValueError("v6.1c freezes the v9 field at 81x65 and sigma=1.25 cells")
    if int(_v6_args.episode_index) < 0:
        raise ValueError("--episode_index must be >= 0")
    nominal_lower_frames = dict(PHASE_SPECS)["LOWER_TO_GRASP"]
    if int(_v6_args.max_grasp_centering_frames) < nominal_lower_frames:
        raise ValueError(
            "--max_grasp_centering_frames must be at least the nominal "
            f"LOWER_TO_GRASP duration ({nominal_lower_frames})"
        )
    if bool(_v6_args.save_diagnostic_video) != bool(args_cli.save_camera_rgb):
        raise RuntimeError("Diagnostic-video and inherited camera settings diverged")
    mainline_v9._validate_args()


def _smoothstep01(value: float) -> float:
    alpha = min(max(float(value), 0.0), 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _quat_matrix(quat_wxyz: np.ndarray | tuple[float, float, float, float]) -> np.ndarray:
    return mainline_v9._quat_to_matrix(tuple(float(v) for v in quat_wxyz))


def _pose_matrix(position: np.ndarray, quat_wxyz: np.ndarray | tuple[float, float, float, float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _quat_matrix(quat_wxyz)
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def _pose_delta_matrix(
    current_position: np.ndarray,
    current_quat: np.ndarray,
    initial_position: np.ndarray,
    initial_quat: np.ndarray,
) -> np.ndarray:
    return _pose_matrix(current_position, current_quat) @ np.linalg.inv(
        _pose_matrix(initial_position, initial_quat)
    )


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64) @ np.asarray(second, dtype=np.float64).T
    cosine = min(max(0.5 * (float(np.trace(relative)) - 1.0), -1.0), 1.0)
    return math.degrees(math.acos(cosine))


def _transform_to_q(transform: np.ndarray) -> np.ndarray:
    """Match libuipc's affine generalized-coordinate row layout exactly."""
    value = np.asarray(transform, dtype=np.float64)
    return np.concatenate((value[:3, 3], value[0, :3], value[1, :3], value[2, :3]))


def _validate_rigid_transform(transform: np.ndarray) -> None:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Rigid transform must be a finite (4, 4) array")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-9):
        raise ValueError("Rigid transform has an invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-8):
        raise ValueError("Rigid transform rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("Rigid transform rotation must have determinant +1")


def _interpolate_pose(
    previous_position: np.ndarray,
    previous_quat: np.ndarray,
    current_position: np.ndarray,
    current_quat: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(previous_position, dtype=np.float64) + float(alpha) * (
        np.asarray(current_position, dtype=np.float64)
        - np.asarray(previous_position, dtype=np.float64)
    )
    quat = _quat_slerp_wxyz(previous_quat, current_quat, alpha)
    quat /= max(float(np.linalg.norm(quat)), EPS)
    return position, quat


def _recover_rigid_transform(local_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    local = np.asarray(local_points, dtype=np.float64)
    world = np.asarray(world_points, dtype=np.float64)
    if local.shape != world.shape or local.ndim != 2 or local.shape[1] != 3 or local.shape[0] < 3:
        raise ValueError("Rigid pose recovery requires matching (N, 3) point arrays")
    local_center = np.mean(local, axis=0)
    world_center = np.mean(world, axis=0)
    covariance = (local - local_center).T @ (world - world_center)
    u_value, _, vt_value = np.linalg.svd(covariance)
    rotation = vt_value.T @ u_value.T
    if np.linalg.det(rotation) < 0.0:
        vt_value[-1] *= -1.0
        rotation = vt_value.T @ u_value.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = world_center - rotation @ local_center
    _validate_rigid_transform(transform)
    return transform


def _pose_errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]) * 1000.0),
        _rotation_error_deg(actual[:3, :3], expected[:3, :3]),
    )


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray((0.25 * scale,
                           (matrix[2, 1] - matrix[1, 2]) / scale,
                           (matrix[0, 2] - matrix[2, 0]) / scale,
                           (matrix[1, 0] - matrix[0, 1]) / scale))
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                               (matrix[0, 1] + matrix[1, 0]) / scale,
                               (matrix[0, 2] + matrix[2, 0]) / scale))
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(((matrix[0, 2] - matrix[2, 0]) / scale,
                               (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                               (matrix[1, 2] + matrix[2, 1]) / scale))
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(((matrix[1, 0] - matrix[0, 1]) / scale,
                               (matrix[0, 2] + matrix[2, 0]) / scale,
                               (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale))
    return quat / max(float(np.linalg.norm(quat)), EPS)


def _transform_pose_w(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    return np.asarray([*value[:3, 3], *_matrix_to_quat_wxyz(value[:3, :3])], dtype=np.float32)


def _relative_transform(pad_transform: np.ndarray, object_transform: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(pad_transform, dtype=np.float64)) @ np.asarray(
        object_transform, dtype=np.float64
    )


def _quat_slerp_wxyz(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    first_q = np.asarray(first, dtype=np.float64)
    second_q = np.asarray(second, dtype=np.float64)
    first_q /= max(float(np.linalg.norm(first_q)), EPS)
    second_q /= max(float(np.linalg.norm(second_q)), EPS)
    dot = float(np.dot(first_q, second_q))
    if dot < 0.0:
        second_q = -second_q
        dot = -dot
    if dot > 0.9995:
        result = first_q + float(alpha) * (second_q - first_q)
        return result / max(float(np.linalg.norm(result)), EPS)
    theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - float(alpha)) * theta) / sin_theta * first_q
        + math.sin(float(alpha) * theta) / sin_theta * second_q
    )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64).copy()
    value[1:] *= -1.0
    return value


def _quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray(mainline_v9._quat_multiply(tuple(first), tuple(second)), dtype=np.float64)


def _relative_pose(
    child_position: np.ndarray,
    child_quat: np.ndarray,
    parent_position: np.ndarray,
    parent_quat: np.ndarray,
) -> np.ndarray:
    position = (np.asarray(child_position) - np.asarray(parent_position)) @ _quat_matrix(parent_quat)
    quat = _quat_multiply(_quat_conjugate(np.asarray(parent_quat)), np.asarray(child_quat))
    return np.asarray([*position, *quat], dtype=np.float32)


def _capped_z_cylinder_signed_distance(
    points_object_l: np.ndarray, radius_m: float, height_m: float
) -> np.ndarray:
    """Exact signed distance to a finite cylinder whose local axis is +Z."""
    points = np.asarray(points_object_l, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Cylinder distance query points must have shape (N, 3)")
    radial_delta = np.linalg.norm(points[:, :2], axis=1) - float(radius_m)
    axial_delta = np.abs(points[:, 2]) - 0.5 * float(height_m)
    outside = np.linalg.norm(
        np.maximum(np.column_stack((radial_delta, axial_delta)), 0.0), axis=1
    )
    inside = np.minimum(np.maximum(radial_delta, axial_delta), 0.0)
    return outside + inside


def _free_cylinder_contact_diagnostics(
    current_surface_w: np.ndarray,
    current_surface_pad_l: np.ndarray,
    rest_surface_pad_l: np.ndarray,
    front_indices: np.ndarray,
    object_position_w: np.ndarray,
    object_quat_w: np.ndarray,
    *,
    radius_m: float,
    height_m: float,
    contact_distance_m: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Diagnose contact against the mirrored finite cylinder in its live rigid frame."""
    current_surface = np.asarray(current_surface_pad_l, dtype=np.float64)
    rest_surface = np.asarray(rest_surface_pad_l, dtype=np.float64)
    indices = np.asarray(front_indices, dtype=np.int64)
    current_front_w = np.asarray(current_surface_w, dtype=np.float64)[indices]
    current_front_pad_l = current_surface[indices]
    rest_front_pad_l = rest_surface[indices]
    object_rotation_w = _quat_matrix(object_quat_w)
    current_front_object_l = (
        current_front_w - np.asarray(object_position_w, dtype=np.float64).reshape(1, 3)
    ) @ object_rotation_w
    signed_gap_m = _capped_z_cylinder_signed_distance(
        current_front_object_l, radius_m, height_m
    )
    front_contact = signed_gap_m <= float(contact_distance_m)
    contact_mask = np.zeros(current_surface.shape[0], dtype=bool)
    contact_mask[indices] = front_contact
    normal_compression_m = np.clip(
        rest_front_pad_l[:, 0] - current_front_pad_l[:, 0], 0.0, None
    )
    deformation_m = np.linalg.norm(current_front_pad_l - rest_front_pad_l, axis=1)
    return contact_mask, {
        "contact_vertex_count": int(np.count_nonzero(front_contact)),
        "footprint_vertex_count": int(indices.size),
        "min_signed_gap_mm": float(np.min(signed_gap_m) * 1000.0)
        if signed_gap_m.size
        else 0.0,
        "max_penetration_mm": float(np.max(np.clip(-signed_gap_m, 0.0, None)) * 1000.0)
        if signed_gap_m.size
        else 0.0,
        "max_normal_compression_mm": float(np.max(normal_compression_m) * 1000.0)
        if normal_compression_m.size
        else 0.0,
        "max_front_deformation_mm": float(np.max(deformation_m) * 1000.0)
        if deformation_m.size
        else 0.0,
        "mean_front_deformation_mm": float(np.mean(deformation_m) * 1000.0)
        if deformation_m.size
        else 0.0,
    }


def _z_cylinder_surface(
    radius_m: float, height_m: float, segments: int = 32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segment_count = max(12, int(segments))
    half_height = 0.5 * float(height_m)
    points: list[tuple[float, float, float]] = []
    for z_value in (-half_height, half_height):
        for index in range(segment_count):
            theta = 2.0 * math.pi * float(index) / float(segment_count)
            points.append((float(radius_m) * math.cos(theta), float(radius_m) * math.sin(theta), z_value))
    bottom_center = len(points)
    points.append((0.0, 0.0, -half_height))
    top_center = len(points)
    points.append((0.0, 0.0, half_height))
    triangles: list[tuple[int, int, int]] = []
    tetrahedra: list[tuple[int, int, int, int]] = []
    for index in range(segment_count):
        following = (index + 1) % segment_count
        bottom_i, bottom_j = index, following
        top_i, top_j = segment_count + index, segment_count + following
        triangles.extend(((bottom_i, top_i, bottom_j), (bottom_j, top_i, top_j)))
        triangles.append((bottom_center, bottom_j, bottom_i))
        triangles.append((top_center, top_i, top_j))
        # Each triangular sector extrudes to a prism.  This consistent three-tet
        # split partitions the full cylinder without adding interior vertices.
        tetrahedra.extend(
            (
                (bottom_center, bottom_i, bottom_j, top_j),
                (bottom_center, top_i, bottom_i, top_j),
                (bottom_center, top_center, top_i, top_j),
            )
        )
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(triangles, dtype=np.int32),
        np.asarray(tetrahedra, dtype=np.int32),
    )


def _rigid_props() -> RigidBodyPropertiesCfg:
    return RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=4,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        kinematic_enabled=False,
        disable_gravity=False,
        enable_gyroscopic_forces=True,
    )


def _resolve_ee_body(robot) -> tuple[int, str]:
    indices, names = robot.find_bodies(str(_v6_args.piper_gripper_body))
    if len(indices) != 1:
        raise RuntimeError(
            f"Expected one body matching {_v6_args.piper_gripper_body!r}, got {list(names)}"
        )
    return int(indices[0]), str(names[0])


def _compute_frame_pose(robot, body_idx: int, offset_pos: torch.Tensor, offset_rot: torch.Tensor):
    body_pos_w = robot.data.body_link_pos_w[:, body_idx]
    body_quat_w = robot.data.body_link_quat_w[:, body_idx]
    position_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w, robot.data.root_link_quat_w, body_pos_w, body_quat_w
    )
    return math_utils.combine_frame_transforms(position_b, quat_b, offset_pos, offset_rot)


def _compute_frame_jacobian(robot, jacobi_body_idx: int, offset_pos: torch.Tensor, offset_rot: torch.Tensor):
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, :].clone()
    base_rotation = math_utils.matrix_from_quat(math_utils.quat_inv(robot.data.root_link_quat_w))
    jacobian[:, :3, :] = torch.bmm(base_rotation, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rotation, jacobian[:, 3:, :])
    jacobian[:, :3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_pos), jacobian[:, 3:, :])
    jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(offset_rot), jacobian[:, 3:, :])
    return jacobian


def _world_position_to_base(robot, target_position_w: np.ndarray) -> torch.Tensor:
    device = robot.data.root_link_pos_w.device
    position_w = torch.as_tensor(target_position_w, device=device, dtype=torch.float32).reshape(1, 3)
    position_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        position_w,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
    )
    return position_b


def _apply_ik_action(
    robot,
    controller,
    target_position_w: np.ndarray,
    opening_mm: float,
    body_idx: int,
    jacobi_body_idx: int,
    finger_joint_ids: list[int],
    finger_joint_signs: torch.Tensor,
    offset_pos: torch.Tensor,
    offset_rot: torch.Tensor,
) -> None:
    position_b, quat_b = _compute_frame_pose(robot, body_idx, offset_pos, offset_rot)
    controller.set_command(_world_position_to_base(robot, target_position_w), position_b, quat_b)
    jacobian = _compute_frame_jacobian(robot, jacobi_body_idx, offset_pos, offset_rot)
    desired = controller.compute(position_b, quat_b, jacobian, robot.data.joint_pos).clone()
    opening_m = min(max(float(opening_mm), 0.0), 35.0) * 1.0e-3
    desired[:, finger_joint_ids] = opening_m * finger_joint_signs.to(desired)
    robot.set_joint_position_target(desired)
    if hasattr(robot, "write_data_to_sim"):
        robot.write_data_to_sim()


def _ee_pose_w(robot, body_idx: int, offset_pos: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    body_position = robot.data.body_link_pos_w[:, body_idx]
    body_quat = robot.data.body_link_quat_w[:, body_idx]
    position = body_position + math_utils.quat_apply(body_quat, offset_pos)
    return (
        position[0].detach().cpu().numpy().astype(np.float64),
        body_quat[0].detach().cpu().numpy().astype(np.float64),
    )


def _object_state(cylinder: RigidObject) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = cylinder.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)
    quat = cylinder.data.root_link_quat_w[0].detach().cpu().numpy().astype(np.float64)
    velocity = cylinder.data.root_link_vel_w[0].detach().cpu().numpy().astype(np.float64)
    return position, quat, velocity[:3], velocity[3:]


def _build_waypoints(
    object_position: np.ndarray,
    centered_grasp_position: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    cx, cy = float(object_position[0]), float(object_position[1])
    norm = max(math.hypot(cx, cy), EPS)
    grasp = (
        np.asarray(centered_grasp_position, dtype=np.float64).copy()
        if centered_grasp_position is not None
        else np.asarray(
            (
                cx + cx / norm * float(_v6_args.grasp_forward_offset),
                cy + cy / norm * float(_v6_args.grasp_forward_offset),
                float(object_position[2]) + float(_v6_args.grasp_z_offset),
            ),
            dtype=np.float64,
        )
    )
    home = np.asarray(
        (float(_v6_args.home_ee_x), float(_v6_args.home_ee_y), float(_v6_args.home_ee_z)),
        dtype=np.float64,
    )
    above = grasp.copy()
    above[2] = float(_v6_args.approach_z)
    lift = grasp.copy()
    lift[2] += float(_v6_args.lift_distance_mm) * 1.0e-3
    return {"home": home, "above_pick": above, "grasp": grasp, "lift": lift, "place": grasp.copy()}


def _membrane_centered_grasp_target(
    object_position_w: np.ndarray,
    ee_position_w: np.ndarray,
    pad_position_w: np.ndarray,
    pad_quat_w: np.ndarray,
    membrane_front_center_pad_l: np.ndarray,
    *,
    object_radius_m: float,
    precontact_clearance_m: float,
    contact_normal_sign: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Center the membrane on the object while retaining a normal safety gap.

    Pad-local Y/Z are aligned with the object center.  Along Pad-local X the
    membrane stays one radius plus ``precontact_clearance_m`` from the center.
    This keeps LOWER_TO_GRASP outside the UIPC contact activation range instead
    of proactively putting the membrane at zero gap.
    """
    pad_rotation_w = _quat_matrix(pad_quat_w)
    membrane_center_w = np.asarray(pad_position_w, dtype=np.float64) + pad_rotation_w @ np.asarray(
        membrane_front_center_pad_l, dtype=np.float64
    )
    object_from_membrane_pad_l = pad_rotation_w.T @ (
        np.asarray(object_position_w, dtype=np.float64) - membrane_center_w
    )
    desired_object_from_membrane_pad_l = np.asarray(
        (
            float(contact_normal_sign)
            * (float(object_radius_m) + float(precontact_clearance_m)),
            0.0,
            0.0,
        ),
        dtype=np.float64,
    )
    correction_w = pad_rotation_w @ (
        object_from_membrane_pad_l - desired_object_from_membrane_pad_l
    )
    centered_ee_target_w = np.asarray(ee_position_w, dtype=np.float64) + correction_w
    tangent_error_mm = float(
        np.linalg.norm(object_from_membrane_pad_l[1:3]) * 1000.0
    )
    normal_surface_error_mm = float(
        abs(
            object_from_membrane_pad_l[0]
            - desired_object_from_membrane_pad_l[0]
        )
        * 1000.0
    )
    return (
        centered_ee_target_w,
        membrane_center_w,
        tangent_error_mm,
        normal_surface_error_mm,
    )


def _phase_plan(waypoints: dict[str, np.ndarray]) -> list[dict[str, object]]:
    opened = float(args_cli.gripper_opening_mm)
    closed = float(_v6_args.gripper_closed_mm)
    targets = {
        "SETTLE_READY": (waypoints["home"], opened),
        "APPROACH_PICK": (waypoints["above_pick"], opened),
        "LOWER_TO_GRASP": (waypoints["grasp"], opened),
        "CLOSE_GRIPPER": (waypoints["grasp"], closed),
        "CONFIRM_GRASP": (waypoints["grasp"], closed),
        "LIFT_OBJECT": (waypoints["lift"], closed),
        "HOLD_LIFTED": (waypoints["lift"], closed),
    }
    plan = [
        {"name": name, "target": targets[name][0], "opening": targets[name][1], "frames": frames}
        for name, frames in PHASE_SPECS
    ]
    if sum(int(phase["frames"]) for phase in plan) != SHORT_FRAME_COUNT:
        raise RuntimeError("The v6.1c short phase plan has an inconsistent frame budget")
    return plan


def _cylinder_vertical_half_extent(quat: np.ndarray) -> float:
    rotation = _quat_matrix(quat)
    radius = float(_v6_args.object_radius_mm) * 1.0e-3
    half_height = 0.5 * float(_v6_args.object_height_mm) * 1.0e-3
    return half_height * abs(float(rotation[2, 2])) + radius * math.hypot(
        float(rotation[2, 0]), float(rotation[2, 1])
    )


def _coefficient_of_variation(values: np.ndarray) -> float:
    array = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    mean = float(np.mean(array)) if array.size else 0.0
    return float(np.std(array) / max(mean, EPS))


def _field_centroid(field: np.ndarray) -> np.ndarray | None:
    weights = np.clip(np.asarray(field, dtype=np.float64), 0.0, None)
    total = float(np.sum(weights))
    if total <= EPS:
        return None
    rows, columns = np.indices(weights.shape, dtype=np.float64)
    return np.asarray((np.sum(rows * weights) / total, np.sum(columns * weights) / total))


def _field_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return 1.0 if np.allclose(a, b, rtol=0.0, atol=1.0e-12) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _json_write(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def _atomic_savez(path: Path, values: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
    os.replace(temporary, path)


def _save_record_arrays(output_dir: Path, records: dict[str, list[object]]) -> None:
    array_names = (
        ("frame_id", "frame_id.npy"),
        ("phase_id", "phase_id.npy"),
        ("object_pose_w", "object_pose_w.npy"),
        ("object_velocity_w", "object_velocity_w.npy"),
        ("end_effector_pose_w", "gripper_pose_w.npy"),
        ("pad_pose_w", "pad_pose_w.npy"),
        ("object_relative_pose_to_gripper", "object_pose_gripper_local.npy"),
        ("object_gripper_distance_mm", "object_to_gripper_distance_mm.npy"),
        ("object_lift_height_mm", "object_lift_mm.npy"),
        ("gripper_opening_mm", "gripper_opening_mm.npy"),
        ("physx_left_finger_contact_count", "physx_left_finger_contact_count.npy"),
        ("physx_right_finger_contact_count", "physx_right_finger_contact_count.npy"),
        ("contact_vertex_count", "uipc_pad_contact_count.npy"),
        ("minimum_signed_distance_mm", "minimum_signed_distance_mm.npy"),
        ("max_penetration_mm", "maximum_penetration_mm.npy"),
        ("max_normal_compression_mm", "maximum_normal_compression_mm.npy"),
        ("force_pad_local", "force_pad_local.npy"),
        ("tactile_force_channels", "tactile_force_channels.npy"),
        ("surface_deformation", "surface_displacement_pad_local.npy"),
        ("uipc_substep_wall_time_sec", "uipc_substep_wall_time_sec.npy"),
        ("uipc_frame_wall_time_sec", "uipc_frame_wall_time_sec.npy"),
        ("physx_step_wall_time_sec", "physx_step_wall_time_sec.npy"),
        ("deformation_and_force_wall_time_sec", "deformation_and_force_wall_time_sec.npy"),
        ("tactile_field_wall_time_sec", "tactile_field_wall_time_sec.npy"),
        ("video_capture_wall_time_sec", "video_capture_wall_time_sec.npy"),
        ("data_save_wall_time_sec", "data_save_wall_time_sec.npy"),
        ("formal_frame_wall_time_sec", "formal_frame_wall_time_sec.npy"),
        ("membrane_center_tangent_error_mm", "membrane_center_tangent_error_mm.npy"),
        (
            "membrane_center_normal_surface_error_mm",
            "membrane_center_normal_surface_error_mm.npy",
        ),
        ("physx_uipc_object_position_error_mm", "physx_uipc_object_position_error_mm.npy"),
        ("physx_uipc_object_orientation_error_deg", "physx_uipc_object_orientation_error_deg.npy"),
        ("pad_attachment_position_error_mm", "pad_attachment_position_error_mm.npy"),
        ("pad_attachment_orientation_error_deg", "pad_attachment_orientation_error_deg.npy"),
        ("physx_uipc_relative_position_error_mm", "physx_uipc_relative_position_error_mm.npy"),
        ("physx_uipc_relative_orientation_error_deg", "physx_uipc_relative_orientation_error_deg.npy"),
    )
    for key, filename in array_names:
        dtype = bool if key == "contact_vertex_mask" else None
        _atomic_save(output_dir / filename, np.asarray(records[key], dtype=dtype))
    _json_write(output_dir / "phase_history.json", records["phase"])


def _partial_status(
    records: dict[str, list[object]], *, official_pass: bool = False,
    partial: bool = True, termination_reason: str = ""
) -> dict[str, object]:
    return {
        "official_pass": bool(official_pass),
        "partial_diagnostic_only": bool(partial),
        "completed_frame_count": int(len(records["phase"])),
        "last_completed_phase": str(records["phase"][-1]) if records["phase"] else "",
        "termination_reason": str(termination_reason),
    }


def _checkpoint_payload(records: dict[str, list[object]]) -> dict[str, np.ndarray]:
    keys = (
        "frame_id", "phase_id", "object_pose_w", "object_velocity_w",
        "end_effector_pose_w", "object_relative_pose_to_gripper",
        "object_gripper_distance_mm", "object_lift_height_mm", "gripper_opening_mm",
        "physx_left_finger_contact_count", "physx_right_finger_contact_count",
        "contact_vertex_count", "minimum_signed_distance_mm", "max_penetration_mm",
        "max_normal_compression_mm", "force_pad_local", "tactile_force_channels",
        "surface_deformation", "uipc_substep_wall_time_sec", "uipc_frame_wall_time_sec",
        "physx_step_wall_time_sec", "deformation_and_force_wall_time_sec",
        "tactile_field_wall_time_sec", "data_save_wall_time_sec",
        "video_capture_wall_time_sec",
        "membrane_center_tangent_error_mm",
        "membrane_center_normal_surface_error_mm",
    )
    payload = {key: np.asarray(records[key]) for key in keys}
    payload["phase"] = np.asarray(records["phase"], dtype=str)
    payload["official_pass"] = np.asarray(False)
    payload["partial_diagnostic_only"] = np.asarray(True)
    payload["completed_frame_count"] = np.asarray(len(records["phase"]), dtype=np.int64)
    payload["last_completed_phase"] = np.asarray(
        records["phase"][-1] if records["phase"] else "", dtype=str
    )
    return payload


def _write_checkpoint(
    output_dir: Path, filename: str, records: dict[str, list[object]]
) -> None:
    _atomic_savez(output_dir / filename, _checkpoint_payload(records))


def _write_lightweight_progress(
    output_dir: Path, records: dict[str, list[object]]
) -> None:
    """Atomically persist only the newest diagnostic state.

    This intentionally overwrites a small fixed-size snapshot instead of
    serializing the complete in-memory history.  Full histories are retained
    for phase boundaries, termination, and normal completion.
    """
    if not records["phase"]:
        return
    latest_keys = (
        "frame_id",
        "phase_id",
        "object_pose_w",
        "object_velocity_w",
        "end_effector_pose_w",
        "pad_pose_w",
        "gripper_opening_mm",
        "object_lift_height_mm",
        "object_gripper_distance_mm",
        "contact_vertex_count",
        "minimum_signed_distance_mm",
        "max_penetration_mm",
        "max_front_deformation_mm",
        "max_normal_compression_mm",
        "force_pad_local",
        "tactile_force_channels",
        "uipc_substep_wall_time_sec",
        "uipc_frame_wall_time_sec",
        "physx_step_wall_time_sec",
        "deformation_and_force_wall_time_sec",
        "tactile_field_wall_time_sec",
        "video_capture_wall_time_sec",
        "membrane_center_tangent_error_mm",
        "membrane_center_normal_surface_error_mm",
    )
    payload = {
        "completed_frame_count": np.asarray(len(records["phase"]), dtype=np.int64),
        "phase": np.asarray(records["phase"][-1], dtype=str),
    }
    payload.update({key: np.asarray(records[key][-1]) for key in latest_keys})
    _atomic_savez(output_dir / "checkpoint_lightweight_progress.npz", payload)


def _physx_contact_count(sensor: ContactSensor) -> int:
    sensor.update(0.0, force_recompute=True)
    force_matrix = sensor.data.force_matrix_w
    if force_matrix is None or force_matrix.numel() == 0:
        return 0
    norms = torch.linalg.norm(force_matrix[0, 0], dim=-1)
    return int(torch.count_nonzero(norms > float(_v6_args.physx_contact_force_epsilon_n)).item())


class _HardSubstepWatchdog:
    """Kill a wedged native solve after prior completed frames are already durable."""

    def __init__(
        self, output_dir: Path, records: dict[str, list[object]],
        frame: int, phase: str, substep: int, video_writer=None
    ) -> None:
        self.output_dir = output_dir
        self.records = records
        self.frame = int(frame)
        self.phase = str(phase)
        self.substep = int(substep)
        self.video_writer = video_writer
        self.timer: threading.Timer | None = None

    def __enter__(self) -> "_HardSubstepWatchdog":
        timeout = float(_v6_args.uipc_substep_timeout_sec)
        self.timer = threading.Timer(timeout, self._terminate)
        self.timer.daemon = True
        self.timer.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.timer is not None:
            self.timer.cancel()

    def _terminate(self) -> None:
        reason = (
            f"UIPC substep exceeded {float(_v6_args.uipc_substep_timeout_sec):.3f}s "
            f"at frame {self.frame}, phase {self.phase}, substep {self.substep}"
        )
        try:
            _write_checkpoint(
                self.output_dir, "checkpoint_uipc_substep_timeout.npz", self.records
            )
            _json_write(
                self.output_dir / "partial_diagnostic_status.json",
                _partial_status(self.records, termination_reason=reason),
            )
            _json_write(
                self.output_dir / "failure_snapshot.json",
                {**_partial_status(self.records, termination_reason=reason),
                 "frame": self.frame, "phase": self.phase, "substep": self.substep},
            )
        finally:
            if self.video_writer is not None:
                self.video_writer.release()
            os.kill(os.getpid(), signal.SIGTERM)


def main() -> None:
    _validate_v6_args()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "error.json").unlink(missing_ok=True)
    contract_dir = Path(args_cli.contract_dir).expanduser().resolve()
    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)
    uipc_substep_dt = sim_dt / float(UIPC_SUBSTEPS)

    sim = mainline_v9.sim_utils.SimulationContext(
        mainline_v9.SimulationCfg(
            dt=sim_dt,
            render_interval=1,
            physx=mainline_v9.PhysxCfg(enable_ccd=True),
            physics_material=mainline_v9.sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=float(_v6_args.object_friction),
                dynamic_friction=float(_v6_args.object_friction),
                restitution=0.0,
            ),
        )
    )
    if bool(_v6_args.save_diagnostic_video):
        sim.set_camera_view([0.58, -0.48, 0.42], [0.28, -0.01, 0.13])
    stage = mainline_v9.omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage")
    for prim_path in ("/World", "/World/envs", "/World/envs/env_0"):
        UsdGeom.Xform.Define(stage, prim_path)
    ground_cfg = mainline_v9.sim_utils.GroundPlaneCfg(
        physics_material=mainline_v9.sim_utils.RigidBodyMaterialCfg(
            static_friction=float(_v6_args.object_friction),
            dynamic_friction=float(_v6_args.object_friction),
            restitution=0.0,
        )
    )
    ground_cfg.func(
        "/World/defaultGroundPlane",
        ground_cfg,
        translation=(0.0, 0.0, float(_v6_args.support_height_m)),
    )
    light_cfg = mainline_v9.sim_utils.DomeLightCfg(intensity=2800.0, color=(0.78, 0.78, 0.78))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = mainline_v9.AGILEX_PIPER_HIGH_PD_CFG.replace(
        prim_path=mainline_v9.ROBOT_ROOT
    )
    if str(args_cli.robot_usd_path).strip():
        robot_cfg.spawn.usd_path = str(Path(args_cli.robot_usd_path).expanduser().resolve())
    # Contact-report activation is diagnostic-only; material, friction, solver,
    # joints, and the grasp controller remain exactly as in v6.1.
    robot_cfg.spawn.activate_contact_sensors = True
    robot = mainline_v9.Articulation(robot_cfg)
    object_height_m = float(_v6_args.object_height_mm) * 1.0e-3
    object_radius_m = float(_v6_args.object_radius_mm) * 1.0e-3
    precontact_clearance_m = max(
        2.0 * float(args_cli.uipc_contact_d_hat_mm) * 1.0e-3,
        0.0005,
    )
    object_initial_position = np.asarray(
        (
            float(_v6_args.object_x),
            float(_v6_args.object_y),
            float(_v6_args.support_height_m)
            + 0.5 * object_height_m
            + float(_v6_args.object_support_clearance_mm) * 1.0e-3,
        ),
        dtype=np.float64,
    )
    cylinder = RigidObject(
        RigidObjectCfg(
            prim_path=OBJECT_PATH,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(float(v) for v in object_initial_position),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=mainline_v9.sim_utils.CylinderCfg(
                radius=object_radius_m,
                height=object_height_m,
                axis="Z",
                rigid_props=_rigid_props(),
                mass_props=mainline_v9.sim_utils.MassPropertiesCfg(
                    mass=float(_v6_args.object_mass_kg)
                ),
                collision_props=mainline_v9.sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.001,
                    rest_offset=0.0,
                ),
                physics_material=mainline_v9.sim_utils.RigidBodyMaterialCfg(
                    static_friction=float(_v6_args.object_friction),
                    dynamic_friction=float(_v6_args.object_friction),
                    restitution=0.0,
                ),
                visual_material=mainline_v9.sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.85, 0.34, 0.20), roughness=0.55
                ),
            ),
        )
    )

    mount_link_path = mainline_v9._normalize_mount_link_path(str(args_cli.mount_link_path))
    if mount_link_path != mainline_v9.DEFAULT_MOUNT_LINK_PATH:
        raise ValueError(f"v6.0 requires direct link8 mounting at {mainline_v9.DEFAULT_MOUNT_LINK_PATH}")
    pad_root = f"{mount_link_path}/{mainline_v9.PAD_ASSET_NAME}"
    simulation_root = f"{pad_root}/simulation"
    membrane_mesh_path = f"{simulation_root}/membrane_sim_mesh"
    pad_position_l = (
        float(args_cli.pad_x_mm) * 1.0e-3,
        float(args_cli.pad_y_mm) * 1.0e-3,
        float(args_cli.pad_z_mm) * 1.0e-3,
    )
    pad_quat_l = mainline_v9._quat_from_rpy_deg(
        float(args_cli.pad_roll_deg),
        float(args_cli.pad_pitch_deg),
        float(args_cli.pad_yaw_deg),
    )
    mainline_v9._reference_pad_asset(stage, Path(args_cli.asset_usd), pad_root)
    mainline_v9._set_local_pose(stage, pad_root, pad_position_l, pad_quat_l)
    hidden_visual_paths = mainline_v9._apply_visual_policy(stage, pad_root)
    if not stage.GetPrimAtPath(membrane_mesh_path).IsValid():
        raise RuntimeError(f"Frozen membrane source is missing: {membrane_mesh_path}")

    scene_camera: Camera | None = None
    if bool(_v6_args.save_diagnostic_video):
        scene_camera = Camera(
            CameraCfg(
                prim_path=SCENE_CAMERA_PATH,
                update_period=0.0,
                height=int(args_cli.camera_height),
                width=int(args_cli.camera_width),
                data_types=["rgb"],
                spawn=mainline_v9.sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=1.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.01, 5.0),
                ),
                update_latest_camera_pose=True,
            )
        )
    left_contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=LEFT_FINGER_PATH,
            update_period=0.0,
            history_length=0,
            track_pose=False,
            filter_prim_paths_expr=[OBJECT_PATH],
        )
    )
    right_contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=RIGHT_FINGER_PATH,
            update_period=0.0,
            history_length=0,
            track_pose=False,
            filter_prim_paths_expr=[OBJECT_PATH],
        )
    )

    sim.reset()
    robot.update(0.0)
    cylinder.update(0.0)
    mount_body_idx, mount_body_name = mainline_v9._resolve_mount_body(robot, mount_link_path)
    ee_body_idx, ee_body_name = _resolve_ee_body(robot)
    jacobi_body_idx = ee_body_idx - 1
    finger_joint_ids, finger_joint_signs = mainline_v9._resolve_gripper(robot)
    offset_position = torch.tensor(
        _v6_args.piper_tip_offset, device=sim.device, dtype=torch.float32
    ).reshape(1, 3)
    offset_rotation = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=torch.float32
    )
    ik_controller = DifferentialIKController(
        cfg=DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=False, ik_method="dls"
        ),
        num_envs=1,
        device=sim.device,
    )
    home_position = np.asarray(
        (_v6_args.home_ee_x, _v6_args.home_ee_y, _v6_args.home_ee_z), dtype=np.float64
    )
    if scene_camera is not None:
        camera_eye = torch.tensor([[0.58, -0.48, 0.42]], device=scene_camera.device)
        camera_target = torch.tensor([[0.28, -0.01, 0.13]], device=scene_camera.device)
        scene_camera.set_world_poses_from_view(camera_eye, camera_target)
    # Initialization is the only episode point at which PhysX receives the configured object pose.
    # From here onward the object is never written by this script.
    initialization_object_pose_write_count = 1
    runtime_object_pose_write_count = 0
    for settle_frame in range(int(_v6_args.initial_settle_frames)):
        _apply_ik_action(
            robot,
            ik_controller,
            home_position,
            float(args_cli.gripper_opening_mm),
            ee_body_idx,
            jacobi_body_idx,
            finger_joint_ids,
            finger_joint_signs,
            offset_position,
            offset_rotation,
        )
        sim.step(render=False)
        robot.update(sim_dt)
        cylinder.update(sim_dt)
        if settle_frame % max(1, int(args_cli.log_every)) == 0:
            position, _, velocity, _ = _object_state(cylinder)
            print(
                f"[V6_INIT_PHYSX] frame={settle_frame + 1:04d} "
                f"object_z={position[2]:.6f} speed={np.linalg.norm(velocity):.6g}",
                flush=True,
            )

    mirror_initial_position, mirror_initial_quat, _, _ = _object_state(cylinder)
    warmup_joint_target = robot.data.joint_pos.clone()
    link_position_w, link_quat_w = mainline_v9._body_pose(robot, mount_body_idx)
    initial_pad_position_w, initial_pad_quat_w = mainline_v9._compose_child_pose(
        link_position_w, link_quat_w, pad_position_l, pad_quat_l
    )

    membrane_source_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(membrane_mesh_path))
    source_points_l = np.asarray(membrane_source_mesh.GetPointsAttr().Get(), dtype=np.float64).reshape(-1, 3)
    structured_points_l, structured_tetrahedra, structured_surface_triangles = (
        mainline_v9._structured_box_tet_mesh_l(
            np.min(source_points_l, axis=0),
            np.max(source_points_l, axis=0),
            (1, 22, 26),
        )
    )
    mainline_v9._write_precomputed_tet_data(
        stage,
        membrane_mesh_path,
        structured_points_l,
        structured_tetrahedra,
        structured_surface_triangles,
    )
    uipc_sim = mainline_v9.UipcSim(
        mainline_v9.UipcSimCfg(
            dt=uipc_substep_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=str(Path(args_cli.workspace_dir).expanduser().resolve()),
            sanity_check_enable=bool(args_cli.uipc_sanity_check),
            newton=mainline_v9.UipcSimCfg.Newton(max_iter=int(args_cli.uipc_newton_max_iter)),
            contact=mainline_v9.UipcSimCfg.Contact(
                enable=True,
                enable_friction=True,
                d_hat=float(args_cli.uipc_contact_d_hat_mm) * 1.0e-3,
                default_friction_ratio=float(args_cli.uipc_friction_mu),
                default_contact_resistance=float(args_cli.uipc_contact_resistance_gpa),
            ),
        )
    )
    membrane = mainline_v9.UipcObject(
        mainline_v9.UipcObjectCfg(
            prim_path=simulation_root,
            mesh_cfg=None,
            mass_density=float(args_cli.mass_density),
            constitution_cfg=mainline_v9.UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=float(args_cli.youngs_modulus_mpa),
                poisson_rate=float(args_cli.poisson_rate),
            ),
        ),
        uipc_sim,
    )
    stage_vertices_w = np.asarray(
        membrane.uipc_meshes[0].positions().view().copy().reshape(-1, 3), dtype=np.float32
    )
    rest_tet_vertices_pad_l = mainline_v9._local_from_world_matrix(
        stage_vertices_w, membrane.init_world_transform.detach().cpu().numpy()
    ).astype(np.float32)
    back_tet_indices, front_tet_indices, tet_thickness_m = mainline_v9._face_indices(
        rest_tet_vertices_pad_l
    )
    initial_back_world = mainline_v9._world_from_local(
        rest_tet_vertices_pad_l[back_tet_indices], initial_pad_position_w, initial_pad_quat_w
    )
    attachment_offsets_link_l = mainline_v9._local_from_world(
        initial_back_world, link_position_w, link_quat_w
    ).astype(np.float32)
    mainline_v9._write_precomputed_link_attachment(
        membrane, back_tet_indices, attachment_offsets_link_l
    )
    attachment = mainline_v9.UipcIsaacAttachments(
        mainline_v9.UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=float(args_cli.attachment_strength_ratio),
            body_name=mount_body_name,
            compute_attachment_data=False,
            debug_vis=False,
        ),
        membrane,
        robot,
    )

    mirror_points_l, mirror_triangles, mirror_tetrahedra = _z_cylinder_surface(
        object_radius_m, object_height_m
    )
    mirror_points_w = mainline_v9._world_from_local(
        mirror_points_l, mirror_initial_position, tuple(mirror_initial_quat)
    )
    mainline_v9._write_triangle_mesh(
        stage,
        UIPC_OBJECT_MESH_PATH,
        mirror_points_w,
        mirror_triangles,
        color=(0.24, 0.24, 0.24),
    )
    mainline_v9._write_precomputed_tet_data(
        stage,
        UIPC_OBJECT_MESH_PATH,
        mirror_points_w,
        mirror_tetrahedra,
        mirror_triangles,
    )
    mirror_density = float(_v6_args.object_mass_kg) / max(
        math.pi * object_radius_m * object_radius_m * object_height_m, EPS
    )
    mirror = mainline_v9.UipcObject(
        mainline_v9.UipcObjectCfg(
            prim_path=UIPC_OBJECT_ROOT,
            mesh_cfg=None,
            mass_density=mirror_density,
            constitution_cfg=mainline_v9.UipcObjectCfg.AffineBodyConstitutionCfg(
                m_kappa=float(args_cli.tool_m_kappa_mpa), kinematic=True
            ),
        ),
        uipc_sim,
    )
    mainline_v9._ensure_asset_initialized(membrane)
    mainline_v9._ensure_asset_initialized(attachment)
    mainline_v9._ensure_asset_initialized(mirror)
    attachment._compute_aim_positions()
    mirror_state: dict[str, object] = {
        "target_matrix": np.eye(4, dtype=np.float64),
        "update_count": 0,
        "last_frame": -1,
    }

    manual_uipc_step = uipc_sim.step
    uipc_sim.step = lambda dt=0.0: None
    uipc_sim.setup_sim()
    mainline_v9._write_initial_alignment(
        membrane, rest_tet_vertices_pad_l, initial_pad_position_w, initial_pad_quat_w
    )
    attachment._compute_aim_positions()
    manual_uipc_step()

    def write_mirror_pose_pair(
        previous_matrix: np.ndarray,
        current_matrix: np.ndarray,
        frame_index: int,
    ) -> None:
        _validate_rigid_transform(previous_matrix)
        _validate_rigid_transform(current_matrix)
        mirror.write_kinematic_abd_pose_pair_to_sim(
            previous_matrix, current_matrix, uipc_substep_dt
        )
        mirror_state["previous_matrix"] = np.asarray(previous_matrix, dtype=np.float64)
        mirror_state["target_matrix"] = np.asarray(current_matrix, dtype=np.float64)
        mirror_state["update_count"] = int(mirror_state["update_count"]) + 1
        mirror_state["last_frame"] = int(frame_index)

    UsdGeom.Imageable(stage.GetPrimAtPath(UIPC_OBJECT_ROOT)).MakeInvisible()

    contract_vertex_area = np.asarray(
        np.load(contract_dir / "vertex_area.npy", allow_pickle=False), dtype=np.float64
    ).reshape(-1)
    contract_front_mask = np.asarray(
        np.load(contract_dir / "front_surface_mask.npy", allow_pickle=False), dtype=bool
    ).reshape(-1)
    contract_verdict = json.loads((contract_dir / "verdict.json").read_text())

    provisional_rest = mainline_v9._local_from_world(
        mainline_v9._uipc_surface(membrane), initial_pad_position_w, initial_pad_quat_w
    ).astype(np.float32)
    _, provisional_front_indices, _ = mainline_v9._face_indices(provisional_rest)
    runtime_front_mask = np.zeros(provisional_rest.shape[0], dtype=bool)
    runtime_front_mask[provisional_front_indices] = True
    if contract_vertex_area.shape != (provisional_rest.shape[0],) or not np.array_equal(
        contract_front_mask, runtime_front_mask
    ):
        raise RuntimeError("Runtime membrane surface does not match the frozen 7f contract")
    if not bool(contract_verdict.get("deformation_contract_passed", False)):
        raise RuntimeError("Frozen 7f contract verdict did not pass")

    warmup_force: list[np.ndarray] = []
    warmup_contact: list[int] = []
    warmup_surface: list[np.ndarray] = []
    consecutive_stable = 0
    stability_steps_used = 0
    previous_mirror_position_w = np.asarray(mirror_initial_position, dtype=np.float64).copy()
    previous_mirror_quat_w = np.asarray(mirror_initial_quat, dtype=np.float64).copy()
    previous_pad_position_w = np.asarray(initial_pad_position_w, dtype=np.float64).copy()
    previous_pad_quat_w = np.asarray(initial_pad_quat_w, dtype=np.float64).copy()
    previous_readiness_surface_pad_l = provisional_rest.copy()
    warmup_sync_position: list[float] = []
    warmup_sync_orientation: list[float] = []
    for stability_step in range(int(_v6_args.init_unrecorded_max_frames)):
        stability_steps_used = stability_step + 1
        # The robot has already converged to HOME.  Freeze its joint target during
        # INIT_UNRECORDED so repeated IK updates cannot inject attachment jitter
        # into the 30-frame rest-surface stability gate.
        robot.set_joint_position_target(warmup_joint_target)
        if hasattr(robot, "write_data_to_sim"):
            robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_dt)
        cylinder.update(sim_dt)
        object_position_w, object_quat_w, _, _ = _object_state(cylinder)
        link_position_w, link_quat_w = mainline_v9._body_pose(robot, mount_body_idx)
        pad_position_w, pad_quat_w = mainline_v9._compose_child_pose(
            link_position_w, link_quat_w, pad_position_l, pad_quat_l
        )
        substep_count = int(args_cli.uipc_substeps_per_record)
        frame_sync_position: list[float] = []
        frame_sync_orientation: list[float] = []
        for substep_index in range(substep_count):
            alpha_previous = float(substep_index) / float(substep_count)
            alpha_current = float(substep_index + 1) / float(substep_count)
            object_previous_position, object_previous_quat = _interpolate_pose(
                previous_mirror_position_w, previous_mirror_quat_w,
                object_position_w, object_quat_w, alpha_previous,
            )
            object_current_position, object_current_quat = _interpolate_pose(
                previous_mirror_position_w, previous_mirror_quat_w,
                object_position_w, object_quat_w, alpha_current,
            )
            previous_matrix = _pose_delta_matrix(
                object_previous_position, object_previous_quat,
                mirror_initial_position, mirror_initial_quat,
            )
            target_matrix = _pose_delta_matrix(
                object_current_position, object_current_quat,
                mirror_initial_position, mirror_initial_quat,
            )
            pad_current_position, pad_current_quat = _interpolate_pose(
                previous_pad_position_w, previous_pad_quat_w,
                pad_position_w, pad_quat_w, alpha_current,
            )
            write_mirror_pose_pair(previous_matrix, target_matrix, -(stability_step + 1))
            attachment.aim_positions = mainline_v9._world_from_local(
                rest_tet_vertices_pad_l[back_tet_indices],
                pad_current_position,
                pad_current_quat,
            ).astype(np.float32)
            manual_uipc_step()
            actual_matrix = np.asarray(
                mirror.geo_slot_list[0].geometry().transforms().view()[0], dtype=np.float64
            )
            position_error, orientation_error = _pose_errors(actual_matrix, target_matrix)
            frame_sync_position.append(position_error)
            frame_sync_orientation.append(orientation_error)
        previous_mirror_position_w = np.asarray(object_position_w, dtype=np.float64).copy()
        previous_mirror_quat_w = np.asarray(object_quat_w, dtype=np.float64).copy()
        previous_pad_position_w = np.asarray(pad_position_w, dtype=np.float64).copy()
        previous_pad_quat_w = np.asarray(pad_quat_w, dtype=np.float64).copy()
        surface_w = mainline_v9._uipc_surface(membrane)
        surface_pad_l = mainline_v9._local_from_world(surface_w, pad_position_w, pad_quat_w).astype(np.float32)
        displacement = surface_pad_l.astype(np.float64) - provisional_rest.astype(np.float64)
        force_result = frozen_7g.estimate_deformation_force(
            displacement.reshape(1, *displacement.shape),
            contract_vertex_area,
            contract_front_mask,
            frozen_7g.EstimatorConfig(),
        )
        force_vector = np.asarray(force_result.force_pad_local_tu[0], dtype=np.float64)
        _, contact_diagnostics = _free_cylinder_contact_diagnostics(
            surface_w,
            surface_pad_l,
            provisional_rest,
            provisional_front_indices,
            object_position_w,
            object_quat_w,
            radius_m=object_radius_m,
            height_m=object_height_m,
            contact_distance_m=float(args_cli.uipc_contact_d_hat_mm) * 1.0e-3 * 1.5,
        )
        contact_count = int(contact_diagnostics["contact_vertex_count"])
        frame_displacement_mm = float(
            np.max(np.linalg.norm(surface_pad_l - previous_readiness_surface_pad_l, axis=1))
            * 1000.0
        )
        previous_readiness_surface_pad_l = surface_pad_l.copy()
        stable = bool(
            contact_count == 0
            and float(np.linalg.norm(force_vector)) < float(args_cli.accept_max_warmup_force_tu)
            and frame_displacement_mm < 0.001
            and max(frame_sync_position, default=math.inf) < 0.05
            and max(frame_sync_orientation, default=math.inf) < 0.05
        )
        if stable:
            consecutive_stable += 1
            warmup_force.append(force_vector.copy())
            warmup_contact.append(contact_count)
            warmup_surface.append(surface_pad_l.copy())
            warmup_sync_position.append(max(frame_sync_position))
            warmup_sync_orientation.append(max(frame_sync_orientation))
        else:
            provisional_rest = surface_pad_l.copy()
            consecutive_stable = 0
            warmup_force.clear()
            warmup_contact.clear()
            warmup_surface.clear()
            warmup_sync_position.clear()
            warmup_sync_orientation.clear()
        if stability_step % max(1, int(args_cli.log_every)) == 0:
            print(
                f"[V6_INIT_UIPC] step={stability_step + 1:04d} "
                f"stable={consecutive_stable:02d}/{int(args_cli.warmup_stability_frames)} "
                f"force={np.linalg.norm(force_vector):.6g}TU contact={contact_count} "
                f"motion={frame_displacement_mm:.6g}mm "
                f"sync={max(frame_sync_position):.6g}mm/{max(frame_sync_orientation):.6g}deg",
                flush=True,
            )
        if consecutive_stable >= int(args_cli.warmup_stability_frames):
            break
    if consecutive_stable < int(args_cli.warmup_stability_frames):
        raise RuntimeError("INIT_UNRECORDED did not reach 30 consecutive stable no-contact frames")

    # Capture the formal rest surface only after the stable window, then start fresh records.
    rest_surface_pad_l = surface_pad_l.copy()
    back_surface_indices, front_surface_indices, surface_thickness_m = mainline_v9._face_indices(
        rest_surface_pad_l
    )
    np.save(output_dir / "warmup_surface_pad_local.npy", np.asarray(warmup_surface))
    np.save(output_dir / "warmup_force_pad_local.npy", np.asarray(warmup_force))
    np.save(output_dir / "warmup_contact_vertex_count.npy", np.asarray(warmup_contact))
    np.save(output_dir / "rest_tet_vertices_pad_local.npy", rest_tet_vertices_pad_l)
    np.save(output_dir / "rest_surface_vertices_pad_local.npy", rest_surface_pad_l)
    np.save(output_dir / "front_surface_indices.npy", front_surface_indices)
    np.save(output_dir / "back_surface_indices.npy", back_surface_indices)
    np.save(output_dir / "surface_triangles.npy", mainline_v9._uipc_surface_triangles(membrane))
    np.save(output_dir / "attachment_vertex_indices.npy", back_tet_indices)

    stable_object_position, _, stable_object_velocity, _ = _object_state(cylinder)
    ee_position_w, ee_quat_w = _ee_pose_w(robot, ee_body_idx, offset_position)
    membrane_front_center_pad_l = np.mean(
        rest_surface_pad_l[front_surface_indices], axis=0, dtype=np.float64
    )
    initial_membrane_center_w = np.asarray(pad_position_w, dtype=np.float64) + _quat_matrix(
        pad_quat_w
    ) @ membrane_front_center_pad_l
    initial_object_from_membrane_pad_l = _quat_matrix(pad_quat_w).T @ (
        stable_object_position - initial_membrane_center_w
    )
    contact_normal_sign = 1.0 if initial_object_from_membrane_pad_l[0] >= 0.0 else -1.0
    (
        centered_grasp_position,
        _,
        membrane_center_tangent_error_mm,
        membrane_center_normal_surface_error_mm,
    ) = _membrane_centered_grasp_target(
        stable_object_position,
        ee_position_w,
        pad_position_w,
        pad_quat_w,
        membrane_front_center_pad_l,
        object_radius_m=object_radius_m,
        precontact_clearance_m=precontact_clearance_m,
        contact_normal_sign=contact_normal_sign,
    )
    waypoints = _build_waypoints(stable_object_position, centered_grasp_position)
    phases = _phase_plan(waypoints)
    estimator_config = frozen_7g.EstimatorConfig()
    field_plan = tactile_field_v9.build_gaussian_splat_plan(
        rest_surface_pad_l[front_surface_indices, 1:3],
        height=int(args_cli.field_height),
        width=int(args_cli.field_width),
        sigma_cells=float(args_cli.gaussian_sigma_cells),
        truncate_sigma=float(args_cli.gaussian_truncate_sigma),
    )
    field_gains = {
        "normal_gain_tu_per_m3": float(estimator_config.normal_gain_tu_per_m3),
        "tangent_y_gain_tu_per_m3": float(estimator_config.tangent_y_gain_tu_per_m3),
        "tangent_z_gain_tu_per_m3": float(estimator_config.tangent_z_gain_tu_per_m3),
    }
    configured_attachment_indices = np.asarray(attachment.attachment_points_idx, dtype=np.int64)
    back_constrained_coverage = float(
        np.intersect1d(configured_attachment_indices, back_tet_indices).size
        / max(1, back_tet_indices.size)
    )
    front_constrained_fraction = float(
        np.intersect1d(configured_attachment_indices, front_tet_indices).size
        / max(1, front_tet_indices.size)
    )

    records: dict[str, list[object]] = {
        key: []
        for key in (
            "phase",
            "frame_id",
            "phase_id",
            "robot_joint_position",
            "robot_joint_velocity",
            "gripper_opening_mm",
            "end_effector_pose_w",
            "pad_pose_w",
            "object_pose_w",
            "object_velocity_w",
            "object_angular_velocity_w",
            "object_lift_height_mm",
            "object_gripper_distance_mm",
            "object_relative_pose_to_gripper",
            "physx_uipc_object_position_error_mm",
            "physx_uipc_object_orientation_error_deg",
            "pad_attachment_position_error_mm",
            "pad_attachment_orientation_error_deg",
            "physx_uipc_relative_position_error_mm",
            "physx_uipc_relative_orientation_error_deg",
            "uipc_mirror_pose_w",
            "uipc_mirror_substep_pose_w",
            "uipc_attachment_substep_pose_w",
            "uipc_mirror_substep_displacement_mm",
            "uipc_mirror_substep_rotation_deg",
            "uipc_mirror_generalized_velocity_norm",
            "uipc_mirror_q_prev_q_delta_norm",
            "uipc_surface_w",
            "surface_deformation",
            "contact_vertex_mask",
            "contact_vertex_count",
            "max_penetration_mm",
            "max_front_deformation_mm",
            "max_normal_compression_mm",
            "back_target_error_mm",
            "support_clearance_mm",
            "commanded_opening_mm",
            "cycle_index",
            "physx_left_finger_contact_count",
            "physx_right_finger_contact_count",
            "minimum_signed_distance_mm",
            "force_pad_local",
            "tactile_force_channels",
            "tactile_force_field",
            "uipc_substep_wall_time_sec",
            "uipc_frame_wall_time_sec",
            "physx_step_wall_time_sec",
            "deformation_and_force_wall_time_sec",
            "tactile_field_wall_time_sec",
            "video_capture_wall_time_sec",
            "data_save_wall_time_sec",
            "formal_frame_wall_time_sec",
            "membrane_center_tangent_error_mm",
            "membrane_center_normal_surface_error_mm",
        )
    }
    total_formal_frames = sum(int(phase["frames"]) for phase in phases)
    if total_formal_frames != SHORT_FRAME_COUNT:
        raise RuntimeError(
            f"Expected {SHORT_FRAME_COUNT} short formal frames, got {total_formal_frames}"
        )
    maximum_formal_frames = (
        total_formal_frames
        - dict(PHASE_SPECS)["LOWER_TO_GRASP"]
        + int(_v6_args.max_grasp_centering_frames)
    )
    scene_video_path = output_dir / "short_lift_hold_scene.mp4"
    scene_writer = None
    if bool(_v6_args.save_diagnostic_video):
        if scene_camera is None:
            raise RuntimeError("Diagnostic video requested without a scene camera")
        scene_writer = cv2.VideoWriter(
            str(scene_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args_cli.video_fps),
            (int(args_cli.camera_width), int(args_cli.camera_height)),
        )
        if not scene_writer.isOpened():
            raise RuntimeError(f"Could not open scene video: {scene_video_path}")
        for _ in range(max(1, int(args_cli.camera_warmup_renders))):
            sim.render()
            scene_camera.update(sim_dt)
    _json_write(output_dir / "partial_diagnostic_status.json", _partial_status(records))

    formal_frame = 0
    previous_target = waypoints["home"].copy()
    previous_opening = float(args_cli.gripper_opening_mm)
    frozen_precontact_target: np.ndarray | None = None
    consecutive_zero_pad_contact = 0
    confirmed_relative_pose: np.ndarray | None = None
    confirmed_world_relative_z: float | None = None
    termination_reason = ""
    try:
        for phase in phases:
            phase_name = str(phase["name"])
            target = np.asarray(phase["target"], dtype=np.float64)
            target_opening = float(phase["opening"])
            frame_count = int(phase["frames"])
            if phase_name in ("LIFT_OBJECT", "HOLD_LIFTED"):
                if frozen_precontact_target is None:
                    raise RuntimeError("Cannot start lift before freezing the pre-contact target")
                target = frozen_precontact_target.copy()
                target[2] += float(_v6_args.lift_distance_mm) * 1.0e-3
            print(f"[V6_PHASE] {phase_name}", flush=True)
            phase_frame = 0
            alignment_stable_frames = 0
            phase_frame_limit = (
                int(_v6_args.max_grasp_centering_frames)
                if phase_name == "LOWER_TO_GRASP"
                else frame_count
            )
            while phase_frame < phase_frame_limit:
                if (
                    phase_name == "LOWER_TO_GRASP"
                    and phase_frame >= frame_count
                    and alignment_stable_frames >= GRASP_CENTER_STABLE_FRAMES
                ):
                    break
                frame_wall_start = time.perf_counter()
                if not simulation_app.is_running():
                    raise RuntimeError("Simulation application stopped during the formal trajectory")
                alpha = _smoothstep01(float(phase_frame + 1) / float(frame_count))
                target_position = previous_target + (target - previous_target) * alpha
                commanded_opening = previous_opening + (target_opening - previous_opening) * alpha
                if phase_name == "LOWER_TO_GRASP" and frozen_precontact_target is None:
                    centered_live_target, _, _, _ = _membrane_centered_grasp_target(
                        object_position_w,
                        ee_position_w,
                        pad_position_w,
                        pad_quat_w,
                        membrane_front_center_pad_l,
                        object_radius_m=object_radius_m,
                        precontact_clearance_m=precontact_clearance_m,
                        contact_normal_sign=contact_normal_sign,
                    )
                    target_position = previous_target + (
                        centered_live_target - previous_target
                    ) * alpha
                elif phase_name in ("LOWER_TO_GRASP", "CLOSE_GRIPPER", "CONFIRM_GRASP"):
                    if frozen_precontact_target is None:
                        raise RuntimeError(
                            "Cannot close or confirm before freezing the pre-contact target"
                        )
                    target_position = frozen_precontact_target.copy()

                # 1) commands; 2) PhysX step; 3) read current robot, Pad, and free object.
                _apply_ik_action(
                    robot,
                    ik_controller,
                    target_position,
                    commanded_opening,
                    ee_body_idx,
                    jacobi_body_idx,
                    finger_joint_ids,
                    finger_joint_signs,
                    offset_position,
                    offset_rotation,
                )
                physx_start = time.perf_counter()
                sim.step(render=False)
                robot.update(sim_dt)
                cylinder.update(sim_dt)
                physx_left_contact_count = _physx_contact_count(left_contact_sensor)
                physx_right_contact_count = _physx_contact_count(right_contact_sensor)
                physx_wall_time = time.perf_counter() - physx_start
                object_position_w, object_quat_w, object_velocity_w, object_angular_velocity_w = _object_state(cylinder)
                ee_position_w, ee_quat_w = _ee_pose_w(robot, ee_body_idx, offset_position)
                link_position_w, link_quat_w = mainline_v9._body_pose(robot, mount_body_idx)
                pad_position_w, pad_quat_w = mainline_v9._compose_child_pose(
                    link_position_w, link_quat_w, pad_position_l, pad_quat_l
                )
                (
                    _,
                    _,
                    membrane_center_tangent_error_mm,
                    membrane_center_normal_surface_error_mm,
                ) = _membrane_centered_grasp_target(
                    object_position_w,
                    ee_position_w,
                    pad_position_w,
                    pad_quat_w,
                    membrane_front_center_pad_l,
                    object_radius_m=object_radius_m,
                    precontact_clearance_m=precontact_clearance_m,
                    contact_normal_sign=contact_normal_sign,
                )
                if phase_name == "LOWER_TO_GRASP":
                    center_aligned_this_frame = bool(
                        membrane_center_tangent_error_mm <= GRASP_CENTER_TOLERANCE_MM
                        and membrane_center_normal_surface_error_mm
                        <= GRASP_CENTER_TOLERANCE_MM
                    )
                    alignment_stable_frames = (
                        alignment_stable_frames + 1
                        if center_aligned_this_frame
                        else 0
                    )
                    if (
                        frozen_precontact_target is None
                        and alignment_stable_frames >= GRASP_CENTER_STABLE_FRAMES
                    ):
                        # Preserve the command that reached the clearance target.
                        # From now through CONFIRM_GRASP the arm does not track the
                        # free object; only the fingers establish contact.
                        frozen_precontact_target = target_position.copy()

                # 4) same-frame PhysX pose -> UIPC mirror; 5) attachment; 6) UIPC substeps.
                substep_count = int(args_cli.uipc_substeps_per_record)
                frame_object_position_errors: list[float] = []
                frame_object_orientation_errors: list[float] = []
                frame_attachment_position_errors: list[float] = []
                frame_attachment_orientation_errors: list[float] = []
                frame_relative_position_errors: list[float] = []
                frame_relative_orientation_errors: list[float] = []
                frame_mirror_poses: list[np.ndarray] = []
                frame_attachment_poses: list[np.ndarray] = []
                frame_displacements: list[float] = []
                frame_rotations: list[float] = []
                frame_velocity_norms: list[float] = []
                frame_q_delta_norms: list[float] = []
                frame_substep_wall_times: list[float] = []
                uipc_frame_start = time.perf_counter()
                for substep_index in range(substep_count):
                    alpha_previous = float(substep_index) / float(substep_count)
                    alpha_current = float(substep_index + 1) / float(substep_count)
                    object_previous_position, object_previous_quat = _interpolate_pose(
                        previous_mirror_position_w, previous_mirror_quat_w,
                        object_position_w, object_quat_w, alpha_previous,
                    )
                    interpolated_position, interpolated_quat = _interpolate_pose(
                        previous_mirror_position_w, previous_mirror_quat_w,
                        object_position_w, object_quat_w, alpha_current,
                    )
                    pad_interpolated_position, pad_interpolated_quat = _interpolate_pose(
                        previous_pad_position_w, previous_pad_quat_w,
                        pad_position_w, pad_quat_w, alpha_current,
                    )
                    previous_matrix = _pose_delta_matrix(
                        object_previous_position, object_previous_quat,
                        mirror_initial_position, mirror_initial_quat,
                    )
                    target_matrix = _pose_delta_matrix(
                        interpolated_position,
                        interpolated_quat,
                        mirror_initial_position,
                        mirror_initial_quat,
                    )
                    write_mirror_pose_pair(previous_matrix, target_matrix, formal_frame)
                    written_abd_state = mirror.read_kinematic_abd_state_from_sim()
                    attachment.aim_positions = mainline_v9._world_from_local(
                        rest_tet_vertices_pad_l[back_tet_indices],
                        pad_interpolated_position,
                        pad_interpolated_quat,
                    ).astype(np.float32)
                    substep_wall_start = time.perf_counter()
                    with _HardSubstepWatchdog(
                        output_dir,
                        records,
                        formal_frame,
                        phase_name,
                        substep_index,
                        scene_writer,
                    ):
                        manual_uipc_step()
                    frame_substep_wall_times.append(time.perf_counter() - substep_wall_start)

                    actual_delta = np.asarray(
                        mirror.geo_slot_list[0].geometry().transforms().view()[0],
                        dtype=np.float64,
                    )
                    initial_object_transform = _pose_matrix(
                        mirror_initial_position, mirror_initial_quat
                    )
                    actual_object_transform = actual_delta @ initial_object_transform
                    expected_object_transform = _pose_matrix(
                        interpolated_position, interpolated_quat
                    )
                    expected_pad_transform = _pose_matrix(
                        pad_interpolated_position, pad_interpolated_quat
                    )
                    actual_attachment_transform = _recover_rigid_transform(
                        rest_tet_vertices_pad_l[back_tet_indices], attachment.aim_positions
                    )
                    object_position_error, object_orientation_error = _pose_errors(
                        actual_object_transform, expected_object_transform
                    )
                    attachment_position_error, attachment_orientation_error = _pose_errors(
                        actual_attachment_transform, expected_pad_transform
                    )
                    relative_position_error, relative_orientation_error = _pose_errors(
                        _relative_transform(actual_attachment_transform, actual_object_transform),
                        _relative_transform(expected_pad_transform, expected_object_transform),
                    )
                    errors = (
                        object_position_error, object_orientation_error,
                        attachment_position_error, attachment_orientation_error,
                        relative_position_error, relative_orientation_error,
                    )
                    if any(error > 0.05 for error in errors):
                        failure_snapshot = {
                            "official_pass": False,
                            "frame": formal_frame,
                            "substep": substep_index,
                            "phase": phase_name,
                            "object_position_error_mm": object_position_error,
                            "object_orientation_error_deg": object_orientation_error,
                            "attachment_position_error_mm": attachment_position_error,
                            "attachment_orientation_error_deg": attachment_orientation_error,
                            "relative_position_error_mm": relative_position_error,
                            "relative_orientation_error_deg": relative_orientation_error,
                        }
                        _json_write(output_dir / "failure_snapshot.json", failure_snapshot)
                        raise DiagnosticTermination(
                            "V6.1c substep synchronization exceeded 0.05 mm or 0.05 deg"
                        )
                    q_previous = _transform_to_q(previous_matrix)
                    q_current = _transform_to_q(target_matrix)
                    if not (
                        np.allclose(written_abd_state[0], q_previous, rtol=0.0, atol=1.0e-12)
                        and np.allclose(written_abd_state[1], q_current, rtol=0.0, atol=1.0e-12)
                    ):
                        raise DiagnosticTermination(
                            "Backend ABD q_prev/q state does not match pose-pair targets"
                        )
                    q_delta_norm = float(np.linalg.norm(written_abd_state[4]))
                    frame_object_position_errors.append(object_position_error)
                    frame_object_orientation_errors.append(object_orientation_error)
                    frame_attachment_position_errors.append(attachment_position_error)
                    frame_attachment_orientation_errors.append(attachment_orientation_error)
                    frame_relative_position_errors.append(relative_position_error)
                    frame_relative_orientation_errors.append(relative_orientation_error)
                    frame_mirror_poses.append(_transform_pose_w(actual_object_transform))
                    frame_attachment_poses.append(_transform_pose_w(actual_attachment_transform))
                    frame_displacements.append(float(np.linalg.norm(
                        interpolated_position - object_previous_position
                    ) * 1000.0))
                    frame_rotations.append(_rotation_error_deg(
                        _quat_matrix(interpolated_quat), _quat_matrix(object_previous_quat)
                    ))
                    frame_q_delta_norms.append(q_delta_norm)
                    frame_velocity_norms.append(float(np.linalg.norm(written_abd_state[5])))
                uipc_frame_wall_time = time.perf_counter() - uipc_frame_start
                previous_mirror_position_w = np.asarray(
                    object_position_w, dtype=np.float64
                ).copy()
                previous_mirror_quat_w = np.asarray(object_quat_w, dtype=np.float64).copy()
                previous_pad_position_w = np.asarray(pad_position_w, dtype=np.float64).copy()
                previous_pad_quat_w = np.asarray(pad_quat_w, dtype=np.float64).copy()

                # 7-12) unique membrane -> Pad local -> frozen deformation/force/field inputs -> record.
                deformation_start = time.perf_counter()
                vertices_w = mainline_v9._uipc_vertices(membrane)
                surface_w = mainline_v9._uipc_surface(membrane)
                surface_pad_l = mainline_v9._local_from_world(
                    surface_w, pad_position_w, pad_quat_w
                )
                surface_deformation = (surface_pad_l - rest_surface_pad_l).astype(np.float32)
                force_result = frozen_7g.estimate_deformation_force(
                    surface_deformation,
                    contract_vertex_area,
                    contract_front_mask,
                    estimator_config,
                )
                force_pad_local = np.asarray(
                    force_result.force_pad_local_tu[0], dtype=np.float64
                )
                tactile_force_channels = np.asarray(
                    force_result.tactile_force_channels_tu[0], dtype=np.float64
                )
                deformation_and_force_wall_time = time.perf_counter() - deformation_start

                field_start = time.perf_counter()
                _, tactile_vertex_force = tactile_field_v9.tactile_vertex_contributions(
                    force_result.vertex_deformation_volume_contribution_m3,
                    **field_gains,
                )
                tactile_force_field = tactile_field_v9.splat_vertex_values(
                    tactile_vertex_force[:, front_surface_indices], field_plan
                )[0]
                tactile_field_wall_time = time.perf_counter() - field_start
                contact_mask, contact_diagnostics = _free_cylinder_contact_diagnostics(
                    surface_w,
                    surface_pad_l,
                    rest_surface_pad_l,
                    front_surface_indices,
                    object_position_w,
                    object_quat_w,
                    radius_m=object_radius_m,
                    height_m=object_height_m,
                    contact_distance_m=float(args_cli.uipc_contact_d_hat_mm) * 1.0e-3 * 1.5,
                )
                position_sync_error_mm = frame_object_position_errors[-1]
                orientation_sync_error_deg = frame_object_orientation_errors[-1]
                back_target_error_mm = np.linalg.norm(
                    vertices_w[back_tet_indices] - attachment.aim_positions, axis=1
                ) * 1000.0
                video_capture_wall_time = 0.0
                if scene_writer is not None:
                    if scene_camera is None:
                        raise RuntimeError("Diagnostic video writer has no scene camera")
                    # The UIPC mirror is intentionally invisible; syncing its mesh
                    # only adds a GPU/CPU transfer and cannot affect the video.
                    mainline_v9._sync_render_surface(membrane, surface_w)
                    video_capture_start = time.perf_counter()
                    sim.render()
                    scene_camera.update(sim_dt)
                    rgb = mainline_v9._to_uint8_rgb(scene_camera.data.output["rgb"])
                    scene_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    video_capture_wall_time = time.perf_counter() - video_capture_start

                object_lift_height_mm = float(
                    (object_position_w[2] - stable_object_position[2]) * 1000.0
                )
                object_gripper_distance_mm = float(
                    np.linalg.norm(object_position_w - ee_position_w) * 1000.0
                )
                support_clearance_mm = float(
                    (
                        object_position_w[2]
                        - _cylinder_vertical_half_extent(object_quat_w)
                        - float(_v6_args.support_height_m)
                    )
                    * 1000.0
                )
                object_relative_pose = _relative_pose(
                    object_position_w, object_quat_w, ee_position_w, ee_quat_w
                )
                records["frame_id"].append(formal_frame)
                records["phase_id"].append(PHASE_TO_ID[phase_name])
                records["phase"].append(phase_name)
                records["robot_joint_position"].append(
                    robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
                )
                records["robot_joint_velocity"].append(
                    robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
                )
                records["gripper_opening_mm"].append(mainline_v9._read_gripper_opening_mm(robot))
                records["end_effector_pose_w"].append(
                    np.asarray([*ee_position_w, *ee_quat_w], dtype=np.float32)
                )
                records["pad_pose_w"].append(
                    np.asarray([*pad_position_w, *pad_quat_w], dtype=np.float32)
                )
                records["object_pose_w"].append(
                    np.asarray([*object_position_w, *object_quat_w], dtype=np.float32)
                )
                records["object_velocity_w"].append(object_velocity_w.astype(np.float32))
                records["object_angular_velocity_w"].append(
                    object_angular_velocity_w.astype(np.float32)
                )
                records["object_lift_height_mm"].append(object_lift_height_mm)
                records["object_gripper_distance_mm"].append(object_gripper_distance_mm)
                records["object_relative_pose_to_gripper"].append(object_relative_pose)
                records["physx_uipc_object_position_error_mm"].append(position_sync_error_mm)
                records["physx_uipc_object_orientation_error_deg"].append(
                    np.asarray(frame_object_orientation_errors, dtype=np.float32)
                )
                records["physx_uipc_object_position_error_mm"][-1] = np.asarray(
                    frame_object_position_errors, dtype=np.float32
                )
                records["pad_attachment_position_error_mm"].append(
                    np.asarray(frame_attachment_position_errors, dtype=np.float32)
                )
                records["pad_attachment_orientation_error_deg"].append(
                    np.asarray(frame_attachment_orientation_errors, dtype=np.float32)
                )
                records["physx_uipc_relative_position_error_mm"].append(
                    np.asarray(frame_relative_position_errors, dtype=np.float32)
                )
                records["physx_uipc_relative_orientation_error_deg"].append(
                    np.asarray(frame_relative_orientation_errors, dtype=np.float32)
                )
                records["uipc_mirror_pose_w"].append(frame_mirror_poses[-1])
                records["uipc_mirror_substep_pose_w"].append(
                    np.asarray(frame_mirror_poses, dtype=np.float32)
                )
                records["uipc_attachment_substep_pose_w"].append(
                    np.asarray(frame_attachment_poses, dtype=np.float32)
                )
                records["uipc_mirror_substep_displacement_mm"].append(
                    np.asarray(frame_displacements, dtype=np.float32)
                )
                records["uipc_mirror_substep_rotation_deg"].append(
                    np.asarray(frame_rotations, dtype=np.float32)
                )
                records["uipc_mirror_generalized_velocity_norm"].append(
                    np.asarray(frame_velocity_norms, dtype=np.float32)
                )
                records["uipc_mirror_q_prev_q_delta_norm"].append(
                    np.asarray(frame_q_delta_norms, dtype=np.float32)
                )
                records["uipc_surface_w"].append(surface_w.astype(np.float32))
                records["surface_deformation"].append(surface_deformation)
                records["contact_vertex_mask"].append(contact_mask)
                records["contact_vertex_count"].append(
                    int(contact_diagnostics["contact_vertex_count"])
                )
                records["max_penetration_mm"].append(
                    float(contact_diagnostics["max_penetration_mm"])
                )
                records["max_front_deformation_mm"].append(
                    float(contact_diagnostics["max_front_deformation_mm"])
                )
                records["max_normal_compression_mm"].append(
                    float(contact_diagnostics["max_normal_compression_mm"])
                )
                records["back_target_error_mm"].append(back_target_error_mm.astype(np.float32))
                records["support_clearance_mm"].append(support_clearance_mm)
                records["commanded_opening_mm"].append(commanded_opening)
                records["cycle_index"].append(int(_v6_args.episode_index))
                records["physx_left_finger_contact_count"].append(
                    physx_left_contact_count
                )
                records["physx_right_finger_contact_count"].append(
                    physx_right_contact_count
                )
                records["minimum_signed_distance_mm"].append(
                    float(contact_diagnostics["min_signed_gap_mm"])
                )
                records["force_pad_local"].append(force_pad_local)
                records["tactile_force_channels"].append(tactile_force_channels)
                records["tactile_force_field"].append(tactile_force_field)
                records["uipc_substep_wall_time_sec"].append(
                    np.asarray(frame_substep_wall_times, dtype=np.float64)
                )
                records["uipc_frame_wall_time_sec"].append(uipc_frame_wall_time)
                records["physx_step_wall_time_sec"].append(physx_wall_time)
                records["deformation_and_force_wall_time_sec"].append(
                    deformation_and_force_wall_time
                )
                records["tactile_field_wall_time_sec"].append(tactile_field_wall_time)
                records["video_capture_wall_time_sec"].append(video_capture_wall_time)
                records["data_save_wall_time_sec"].append(0.0)
                records["formal_frame_wall_time_sec"].append(0.0)
                records["membrane_center_tangent_error_mm"].append(
                    membrane_center_tangent_error_mm
                )
                records["membrane_center_normal_surface_error_mm"].append(
                    membrane_center_normal_surface_error_mm
                )

                records["formal_frame_wall_time_sec"][-1] = (
                    time.perf_counter() - frame_wall_start
                )
                if (formal_frame + 1) % LIGHTWEIGHT_CHECKPOINT_INTERVAL_FRAMES == 0:
                    save_start = time.perf_counter()
                    _write_lightweight_progress(output_dir, records)
                    records["data_save_wall_time_sec"][-1] += (
                        time.perf_counter() - save_start
                    )
                    records["formal_frame_wall_time_sec"][-1] = (
                        time.perf_counter() - frame_wall_start
                    )
                    _json_write(
                        output_dir / "partial_diagnostic_status.json",
                        _partial_status(records),
                    )

                pad_contact_count = int(contact_diagnostics["contact_vertex_count"])
                if phase_name in ("LIFT_OBJECT", "HOLD_LIFTED"):
                    consecutive_zero_pad_contact = (
                        consecutive_zero_pad_contact + 1 if pad_contact_count == 0 else 0
                    )
                    if consecutive_zero_pad_contact >= 3:
                        termination_reason = (
                            "UIPC Pad contact_count was zero for three consecutive "
                            f"{phase_name} frames"
                        )
                    if (
                        confirmed_relative_pose is not None
                        and confirmed_world_relative_z is not None
                    ):
                        relative_drift_mm = float(
                            np.linalg.norm(
                                object_relative_pose[:3] - confirmed_relative_pose[:3]
                            )
                            * 1000.0
                        )
                        downward_relative_motion_mm = float(
                            (
                                object_position_w[2]
                                - ee_position_w[2]
                                - confirmed_world_relative_z
                            )
                            * 1000.0
                        )
                        if (
                            relative_drift_mm
                            > float(_v6_args.terminate_obvious_slip_mm)
                            or downward_relative_motion_mm
                            < -float(_v6_args.terminate_obvious_slip_mm)
                        ):
                            termination_reason = (
                                "Object obviously slipped relative to the gripper: "
                                f"drift={relative_drift_mm:.3f}mm, "
                                f"vertical_relative={downward_relative_motion_mm:.3f}mm"
                            )
                if float(contact_diagnostics["max_penetration_mm"]) > float(
                    _v6_args.terminate_max_penetration_mm
                ):
                    termination_reason = (
                        "Penetration limit exceeded: "
                        f"{float(contact_diagnostics['max_penetration_mm']):.6f}mm > "
                        f"{float(_v6_args.terminate_max_penetration_mm):.6f}mm"
                    )
                if not np.all(np.isfinite(tactile_force_channels)):
                    termination_reason = "Non-finite tactile force channel detected"
                if termination_reason:
                    _write_checkpoint(output_dir, "checkpoint_early_termination.npz", records)
                    _json_write(
                        output_dir / "failure_snapshot.json",
                        _partial_status(records, termination_reason=termination_reason),
                    )
                    raise DiagnosticTermination(termination_reason)

                if formal_frame % max(1, int(args_cli.log_every)) == 0:
                    print(
                        f"[V6_RECORD] frame={formal_frame + 1:04d}/<={maximum_formal_frames} "
                        f"phase={phase_name} lift={object_lift_height_mm:+.3f}mm "
                        f"distance={object_gripper_distance_mm:.3f}mm "
                        f"contact={int(contact_diagnostics['contact_vertex_count'])} "
                        f"center={membrane_center_tangent_error_mm:.3f}mm/"
                        f"{membrane_center_normal_surface_error_mm:.3f}mm "
                        f"sync={position_sync_error_mm:.6f}mm/{orientation_sync_error_deg:.6f}deg",
                        flush=True,
                    )
                formal_frame += 1
                phase_frame += 1
            if phase_name == "LOWER_TO_GRASP":
                center_aligned = bool(
                    alignment_stable_frames >= GRASP_CENTER_STABLE_FRAMES
                )
                if not center_aligned:
                    reason = (
                        "Refusing to close before membrane/object center alignment: "
                        f"tangent={membrane_center_tangent_error_mm:.3f}mm, "
                        f"normal_surface={membrane_center_normal_surface_error_mm:.3f}mm, "
                        f"tolerance={GRASP_CENTER_TOLERANCE_MM:.3f}mm, "
                        f"stable={alignment_stable_frames}/{GRASP_CENTER_STABLE_FRAMES}, "
                        f"used={phase_frame}/{phase_frame_limit} frames"
                    )
                    _json_write(
                        output_dir / "failure_snapshot.json",
                        {
                            **_partial_status(records, termination_reason=reason),
                            "membrane_center_tangent_error_mm": membrane_center_tangent_error_mm,
                            "membrane_center_normal_surface_error_mm": (
                                membrane_center_normal_surface_error_mm
                            ),
                        },
                    )
                    raise DiagnosticTermination(reason)
                print(
                    "[V6_GRASP_CENTER_ALIGNED] "
                    f"tangent={membrane_center_tangent_error_mm:.3f}mm "
                    f"normal_surface={membrane_center_normal_surface_error_mm:.3f}mm "
                    f"stable={alignment_stable_frames}/{GRASP_CENTER_STABLE_FRAMES} "
                    f"frames={phase_frame}",
                    flush=True,
                )
            if phase_name == "CONFIRM_GRASP":
                confirm_slice = slice(-10, None)
                confirm_displacement = np.asarray(
                    records["surface_deformation"][confirm_slice], dtype=np.float64
                )
                confirm_force = frozen_7g.estimate_deformation_force(
                    confirm_displacement,
                    contract_vertex_area,
                    contract_front_mask,
                    frozen_7g.EstimatorConfig(),
                )
                confirm_fz = np.asarray(
                    confirm_force.tactile_force_channels_tu, dtype=np.float64
                )[:, 2]
                confirm_opening = np.asarray(
                    records["gripper_opening_mm"][confirm_slice], dtype=np.float64
                )
                confirm_contact = np.asarray(
                    records["contact_vertex_count"][confirm_slice], dtype=np.int64
                )
                confirm_distance = np.asarray(
                    records["object_gripper_distance_mm"][confirm_slice], dtype=np.float64
                )
                confirm_sync = np.asarray(
                    records["physx_uipc_relative_position_error_mm"][confirm_slice],
                    dtype=np.float64,
                )
                grasp_ready = bool(
                    confirm_opening.size == 10
                    and np.all(confirm_opening > float(_v6_args.gripper_closed_mm) + 0.5)
                    and np.all(confirm_contact > 0)
                    and np.ptp(confirm_distance) <= 2.0
                    and np.all(confirm_fz > 0.0)
                    and np.max(confirm_sync, initial=0.0) <= 0.05
                )
                if not grasp_ready:
                    _json_write(
                        output_dir / "failure_snapshot.json",
                        {
                            "official_pass": False,
                            "phase": phase_name,
                            "reason": "CONFIRM_GRASP did not satisfy 10 consecutive readiness frames",
                            "completed_formal_frames": int(formal_frame),
                            "partial_diagnostics_saved": True,
                            "last_10": {
                                "measured_opening_mm": confirm_opening.tolist(),
                                "contact_vertex_count": confirm_contact.tolist(),
                                "object_gripper_distance_mm": confirm_distance.tolist(),
                                "tactile_fz_tu": confirm_fz.tolist(),
                                "relative_sync_position_error_mm": confirm_sync.tolist(),
                                "max_normal_compression_mm": np.asarray(
                                    records["max_normal_compression_mm"][-10:], dtype=np.float64
                                ).tolist(),
                            },
                            "conditions": {
                                "opening_stopped_above_closed_command": bool(np.all(
                                    confirm_opening > float(_v6_args.gripper_closed_mm) + 0.5
                                )),
                                "uipc_contact_present": bool(np.all(confirm_contact > 0)),
                                "distance_stable_within_2mm": bool(np.ptp(confirm_distance) <= 2.0),
                                "frozen_7g_tactile_fz_positive": bool(np.all(confirm_fz > 0.0)),
                                "relative_sync_within_0_05mm": bool(
                                    np.max(confirm_sync, initial=0.0) <= 0.05
                                ),
                            },
                        },
                    )
                    raise DiagnosticTermination(
                        "CONFIRM_GRASP did not satisfy 10 consecutive readiness frames"
                    )
                confirmed_relative_pose = np.asarray(
                    records["object_relative_pose_to_gripper"][-1], dtype=np.float64
                )
                confirmed_world_relative_z = float(
                    np.asarray(records["object_pose_w"][-1], dtype=np.float64)[2]
                    - np.asarray(records["end_effector_pose_w"][-1], dtype=np.float64)[2]
                )
            if phase_name in ("LOWER_TO_GRASP", "CLOSE_GRIPPER", "CONFIRM_GRASP"):
                if frozen_precontact_target is None:
                    raise RuntimeError("Pre-contact target was not frozen before grasp closure")
                previous_target = frozen_precontact_target.copy()
            else:
                previous_target = target.copy()
            previous_opening = target_opening
            phase_save_start = time.perf_counter()
            _write_checkpoint(
                output_dir,
                f"checkpoint_{phase_name.lower()}.npz",
                records,
            )
            records["data_save_wall_time_sec"][-1] += time.perf_counter() - phase_save_start
            records["formal_frame_wall_time_sec"][-1] = (
                time.perf_counter() - frame_wall_start
            )
            _json_write(
                output_dir / "partial_diagnostic_status.json",
                _partial_status(records),
            )
    except KeyboardInterrupt:
        termination_reason = "KeyboardInterrupt requested by user"
        _save_record_arrays(output_dir, records)
        _write_checkpoint(output_dir, "checkpoint_user_interrupt.npz", records)
        _json_write(
            output_dir / "partial_diagnostic_status.json",
            _partial_status(records, termination_reason=termination_reason),
        )
    except DiagnosticTermination as exc:
        termination_reason = str(exc)
        _save_record_arrays(output_dir, records)
        _write_checkpoint(output_dir, "checkpoint_early_termination.npz", records)
        _json_write(
            output_dir / "partial_diagnostic_status.json",
            _partial_status(records, termination_reason=termination_reason),
        )
    except BaseException as exc:
        termination_reason = f"{type(exc).__name__}: {exc}"
        _save_record_arrays(output_dir, records)
        _write_checkpoint(output_dir, "checkpoint_exception.npz", records)
        _json_write(
            output_dir / "partial_diagnostic_status.json",
            _partial_status(records, termination_reason=termination_reason),
        )
        raise
    finally:
        if scene_writer is not None:
            scene_writer.release()

    completed_frame_count = len(records["phase"])
    completed_full_short_run = bool(
        not termination_reason
        and completed_frame_count == formal_frame
        and records["phase"]
        and records["phase"][-1] == "HOLD_LIFTED"
    )
    if runtime_object_pose_write_count != 0:
        raise RuntimeError("A forbidden runtime PhysX object pose write was recorded")

    _finish_short_outputs(
        output_dir=output_dir,
        records=records,
        completed_full_short_run=completed_full_short_run,
        termination_reason=termination_reason,
    )
    simulation_app.close()


def _finish_short_outputs(
    *,
    output_dir: Path,
    records: dict[str, list[object]],
    completed_full_short_run: bool,
    termination_reason: str,
) -> None:
    """Write the short acceptance, contact-loss diagnosis, and timing report."""
    _save_record_arrays(output_dir, records)
    tactile_fields = np.asarray(records["tactile_force_field"], dtype=np.float64)
    _atomic_save(output_dir / "tactile_force_field.npy", tactile_fields)
    if tactile_fields.ndim == 4 and tactile_fields.shape[-1] == 3:
        _atomic_save(output_dir / "tactile_fx_field.npy", tactile_fields[..., 0])
        _atomic_save(output_dir / "tactile_fy_field.npy", tactile_fields[..., 1])
        _atomic_save(output_dir / "tactile_fz_field.npy", tactile_fields[..., 2])
    phases = np.asarray(records["phase"], dtype=str)
    frame_ids = np.asarray(records["frame_id"], dtype=np.int64)
    object_pose = np.asarray(records["object_pose_w"], dtype=np.float64)
    object_velocity = np.asarray(records["object_velocity_w"], dtype=np.float64)
    gripper_pose = np.asarray(records["end_effector_pose_w"], dtype=np.float64)
    object_relative = np.asarray(
        records["object_relative_pose_to_gripper"], dtype=np.float64
    )
    object_distance = np.asarray(
        records["object_gripper_distance_mm"], dtype=np.float64
    )
    object_lift = np.asarray(records["object_lift_height_mm"], dtype=np.float64)
    left_contact = np.asarray(
        records["physx_left_finger_contact_count"], dtype=np.int64
    )
    right_contact = np.asarray(
        records["physx_right_finger_contact_count"], dtype=np.int64
    )
    pad_contact = np.asarray(records["contact_vertex_count"], dtype=np.int64)
    minimum_distance = np.asarray(
        records["minimum_signed_distance_mm"], dtype=np.float64
    )
    compression = np.asarray(records["max_normal_compression_mm"], dtype=np.float64)
    tactile = np.asarray(records["tactile_force_channels"], dtype=np.float64)
    substep_times = np.asarray(
        records["uipc_substep_wall_time_sec"], dtype=np.float64
    )
    uipc_frame_times = np.asarray(
        records["uipc_frame_wall_time_sec"], dtype=np.float64
    )
    physx_times = np.asarray(records["physx_step_wall_time_sec"], dtype=np.float64)
    deformation_times = np.asarray(
        records["deformation_and_force_wall_time_sec"], dtype=np.float64
    )
    field_times = np.asarray(records["tactile_field_wall_time_sec"], dtype=np.float64)
    video_times = np.asarray(records["video_capture_wall_time_sec"], dtype=np.float64)
    save_times = np.asarray(records["data_save_wall_time_sec"], dtype=np.float64)
    total_frame_times = np.asarray(
        records["formal_frame_wall_time_sec"], dtype=np.float64
    )

    def indices(name: str) -> np.ndarray:
        return np.flatnonzero(phases == name)

    def safe_mean(values: np.ndarray) -> float | None:
        return float(np.mean(values)) if values.size else None

    confirm_indices = indices("CONFIRM_GRASP")
    lift_indices = indices("LIFT_OBJECT")
    hold_indices = indices("HOLD_LIFTED")
    close_indices = indices("CLOSE_GRIPPER")
    loss_domain = np.concatenate((lift_indices, hold_indices))
    zero_pad_indices = loss_domain[pad_contact[loss_domain] == 0] if loss_domain.size else np.empty(0, dtype=np.int64)
    reference_index = int(confirm_indices[-1]) if confirm_indices.size else -1

    whole_gripper_slip_conditions = {
        "object_to_gripper_distance_sustained_increase": False,
        "object_rises_slower_than_gripper": False,
        "object_moves_down_relative_to_gripper": False,
        "both_physx_finger_contacts_disappear": False,
    }
    other_side_conditions = {
        "object_still_follows_gripper": False,
        "other_side_physx_contact_remains": False,
        "uipc_pad_contact_disappears": bool(zero_pad_indices.size),
        "lateral_offset_or_rotation": False,
    }
    relative_drift_mm: float | None = None
    relative_rotation_deg: float | None = None
    if reference_index >= 0 and loss_domain.size:
        tail = loss_domain[-min(5, loss_domain.size) :]
        distance_delta = object_distance[loss_domain] - object_distance[reference_index]
        distance_steps = np.diff(object_distance[np.concatenate(([reference_index], loss_domain))])
        relative_world_z = object_pose[:, 2] - gripper_pose[:, 2]
        dt = 1.0 / max(float(args_cli.sim_hz), EPS)
        object_vz = object_velocity[tail, 2]
        gripper_vz = np.gradient(gripper_pose[:, 2], dt)[tail]
        whole_gripper_slip_conditions = {
            "object_to_gripper_distance_sustained_increase": bool(
                distance_delta[-1] > 2.0
                and np.mean(distance_steps > 0.0) >= 0.60
            ),
            "object_rises_slower_than_gripper": bool(
                np.mean(object_vz) < np.mean(gripper_vz) - 0.005
            ),
            "object_moves_down_relative_to_gripper": bool(
                relative_world_z[loss_domain[-1]]
                < relative_world_z[reference_index] - 0.002
            ),
            "both_physx_finger_contacts_disappear": bool(
                np.all(left_contact[tail] == 0) and np.all(right_contact[tail] == 0)
            ),
        }
        relative_drift_mm = float(
            np.linalg.norm(
                object_relative[loss_domain[-1], :3] - object_relative[reference_index, :3]
            )
            * 1000.0
        )
        relative_rotation_deg = mainline_v9._quat_angle_error_deg(
            tuple(object_relative[reference_index, 3:7]),
            tuple(object_relative[loss_domain[-1], 3:7]),
        )
        lateral_shift_mm = float(
            np.linalg.norm(
                object_relative[loss_domain[-1], :2] - object_relative[reference_index, :2]
            )
            * 1000.0
        )
        other_side_conditions = {
            "object_still_follows_gripper": bool(relative_drift_mm <= 2.0),
            # The frozen UIPC Pad is mounted on link8/right; link7/left is the other side.
            "other_side_physx_contact_remains": bool(
                zero_pad_indices.size and np.any(left_contact[zero_pad_indices] > 0)
            ),
            "uipc_pad_contact_disappears": bool(zero_pad_indices.size),
            "lateral_offset_or_rotation": bool(
                lateral_shift_mm > 0.5 or relative_rotation_deg > 1.0
            ),
        }

    geometry_mismatch_indices = np.flatnonzero(
        (minimum_distance <= 0.0)
        & (compression > 0.0)
        & (tactile[:, 2] > 0.0)
        & (pad_contact == 0)
    ) if tactile.size else np.empty(0, dtype=np.int64)
    whole_gripper_slip = bool(all(whole_gripper_slip_conditions.values()))
    other_side_loss = bool(all(other_side_conditions.values()))
    diagnostic_mismatch = bool(geometry_mismatch_indices.size)
    if diagnostic_mismatch:
        classification = "geometric_contact_present_but_uipc_contact_diagnostic_zero"
        next_action = "repair_contact_diagnostic_only"
    elif whole_gripper_slip:
        classification = "object_slipped_from_entire_gripper"
        next_action = "repair_grasp_control_only"
    elif other_side_loss:
        classification = "object_held_by_other_finger_but_left_uipc_pad"
        next_action = "repair_grasp_alignment_and_symmetry_only"
    elif not zero_pad_indices.size:
        classification = "no_uipc_pad_contact_loss_observed"
        next_action = "none"
    else:
        classification = "uipc_pad_contact_loss_indeterminate_from_completed_frames"
        next_action = "inspect_saved_frame_window"

    slowest_substeps: list[dict[str, object]] = []
    if substep_times.size:
        flat_order = np.argsort(substep_times.reshape(-1))[::-1][
            : min(10, substep_times.size)
        ]
        for flat_index in flat_order:
            frame_index, substep_index = np.unravel_index(flat_index, substep_times.shape)
            previous_contact = pad_contact[frame_index - 1] if frame_index > 0 else 0
            labels: list[str] = []
            phase_frame = int(np.count_nonzero(phases[: frame_index + 1] == phases[frame_index])) - 1
            if pad_contact[frame_index] > 0 and previous_contact == 0:
                labels.append("first_contact")
            if phases[frame_index] == "LIFT_OBJECT" and phase_frame == 0:
                labels.append("lift_start")
            if previous_contact > 0 and pad_contact[frame_index] < 0.5 * previous_contact:
                labels.append("rapid_contact_reduction")
            if pad_contact[frame_index] == 0 and previous_contact > 0:
                labels.append("separation")
            slowest_substeps.append(
                {
                    "frame_id": int(frame_ids[frame_index]),
                    "phase": str(phases[frame_index]),
                    "phase_frame": phase_frame,
                    "substep": int(substep_index),
                    "wall_time_sec": float(substep_times[frame_index, substep_index]),
                    "contact_count": int(pad_contact[frame_index]),
                    "event_labels": labels,
                }
            )

    measured_total = float(np.sum(total_frame_times))
    performance = {
        "completed_frame_count": int(phases.size),
        "mean_no_contact_frame_sec": safe_mean(total_frame_times[pad_contact == 0]),
        "mean_close_frame_sec": safe_mean(total_frame_times[close_indices]),
        "mean_lift_frame_sec": safe_mean(total_frame_times[lift_indices]),
        "mean_hold_frame_sec": safe_mean(total_frame_times[hold_indices]),
        "mean_uipc_frame_sec": safe_mean(uipc_frame_times),
        "mean_physx_step_sec": safe_mean(physx_times),
        "mean_deformation_and_force_sec": safe_mean(deformation_times),
        "mean_v9_tactile_field_sec": safe_mean(field_times),
        "mean_video_capture_sec": safe_mean(video_times),
        "mean_data_save_sec": safe_mean(save_times),
        "measured_formal_frame_total_sec": measured_total,
        "uipc_fraction_of_measured_formal_time": (
            float(np.sum(uipc_frame_times) / measured_total) if measured_total > 0.0 else None
        ),
        "slowest_10_uipc_substeps": slowest_substeps,
    }

    confirm_tail = confirm_indices[-10:]
    sync_position = np.asarray(
        records["physx_uipc_object_position_error_mm"], dtype=np.float64
    )
    sync_orientation = np.asarray(
        records["physx_uipc_object_orientation_error_deg"], dtype=np.float64
    )
    evaluation_domain = np.concatenate((lift_indices, hold_indices))
    maximum_relative_drift_mm: float | None = None
    if reference_index >= 0 and evaluation_domain.size:
        maximum_relative_drift_mm = float(
            np.max(
                np.linalg.norm(
                    object_relative[evaluation_domain, :3]
                    - object_relative[reference_index, :3],
                    axis=1,
                )
            )
            * 1000.0
        )
    hold_fz = tactile[hold_indices, 2] if tactile.size else np.empty(0)
    hold_median_fz = float(np.median(hold_fz)) if hold_fz.size else 0.0
    pass_checks = {
        "confirm_grasp_10_consecutive_contact_frames": bool(
            confirm_tail.size == 10 and np.all(pad_contact[confirm_tail] > 0)
        ),
        "sync_position_error_at_most_0_05_mm": bool(
            sync_position.size and np.max(sync_position) <= 0.05
        ),
        "sync_orientation_error_at_most_0_05_deg": bool(
            sync_orientation.size and np.max(sync_orientation) <= 0.05
        ),
        "real_object_lift_at_least_30_mm": bool(
            lift_indices.size and np.max(object_lift[lift_indices]) >= 30.0
        ),
        "lift_pad_contact_nonzero_ratio_at_least_95_percent": bool(
            lift_indices.size == dict(PHASE_SPECS)["LIFT_OBJECT"]
            and np.mean(pad_contact[lift_indices] > 0) >= 0.95
        ),
        "hold_has_contact_in_at_least_19_of_20_frames": bool(
            hold_indices.size == 20 and np.count_nonzero(pad_contact[hold_indices] > 0) >= 19
        ),
        "object_relative_gripper_drift_at_most_2_mm": bool(
            maximum_relative_drift_mm is not None
            and maximum_relative_drift_mm <= 2.0
        ),
        "hold_fz_continuously_nonzero_and_not_divergent": bool(
            hold_fz.size == 20
            and np.all(np.isfinite(hold_fz))
            and np.all(hold_fz > 0.0)
            and float(np.max(hold_fz)) <= max(10.0 * hold_median_fz, EPS)
        ),
    }
    official_pass = bool(completed_full_short_run and all(pass_checks.values()))
    partial = not bool(completed_full_short_run)
    contact_diagnosis = {
        "classification": classification,
        "recommended_next_action": next_action,
        "termination_reason": termination_reason,
        "first_zero_uipc_pad_frame": (
            int(frame_ids[zero_pad_indices[0]]) if zero_pad_indices.size else None
        ),
        "whole_gripper_slip_conditions": whole_gripper_slip_conditions,
        "other_side_only_conditions": other_side_conditions,
        "geometry_diagnostic_mismatch_frame_ids": frame_ids[
            geometry_mismatch_indices
        ].tolist(),
        "final_relative_position_drift_mm": relative_drift_mm,
        "final_relative_orientation_drift_deg": relative_rotation_deg,
        "physx_finger_mapping": {
            "left": LEFT_FINGER_PATH,
            "right": RIGHT_FINGER_PATH,
            "uipc_pad_mounted_finger": PAD_FINGER_PATH,
            "count_definition": "number_of_filtered_object_force_entries_above_epsilon",
        },
    }
    verdict = {
        "official_pass": official_pass,
        "partial_diagnostic_only": partial,
        "completed_frame_count": int(phases.size),
        "last_completed_phase": str(phases[-1]) if phases.size else "",
        "completed_full_short_run": bool(completed_full_short_run),
        "termination_reason": termination_reason,
        "checks": pass_checks,
        "observed": {
            "maximum_object_lift_mm": float(np.max(object_lift, initial=0.0)),
            "lift_pad_contact_nonzero_ratio": (
                float(np.mean(pad_contact[lift_indices] > 0)) if lift_indices.size else None
            ),
            "hold_contact_frame_count": int(
                np.count_nonzero(pad_contact[hold_indices] > 0)
            ),
            "maximum_object_relative_gripper_drift_mm": maximum_relative_drift_mm,
        },
    }
    tactile_force_video_metrics: dict[str, object] | None = None
    tactile_force_video_paths: dict[str, str] = {}
    if bool(_v6_args.save_tactile_force_video):
        if tactile_fields.ndim != 4 or tactile_fields.shape[-1] != 3:
            raise RuntimeError("Cannot encode tactile videos from an invalid Fx/Fy/Fz field")
        if tactile_fields.shape[0] != phases.size:
            raise RuntimeError("Tactile-field and phase histories must have the same frame count")
        tactile_video_start = time.perf_counter()
        tactile_force_video_metrics = tactile_field_v9.render_tactile_videos(
            output_dir=output_dir,
            tactile_field=tactile_fields,
            shear_magnitude=np.linalg.norm(tactile_fields[..., :2], axis=3),
            phases=phases,
            cycles=np.asarray(records["cycle_index"], dtype=np.int64),
            fps=float(args_cli.video_fps),
        )
        tactile_force_video_metrics["postprocess_wall_time_sec"] = (
            time.perf_counter() - tactile_video_start
        )
        tactile_force_video_paths = {
            name: str(output_dir / name) for name in tactile_field_v9.CORE_VIDEO_NAMES
        }
    metadata = {
        "version": "v6.1c_contact_timing_fix",
        "phase_order": [name for name, _ in PHASE_SPECS],
        "nominal_planned_frame_count": SHORT_FRAME_COUNT,
        "maximum_planned_frame_count_with_centering": (
            SHORT_FRAME_COUNT
            - dict(PHASE_SPECS)["LOWER_TO_GRASP"]
            + int(_v6_args.max_grasp_centering_frames)
        ),
        "actual_completed_frame_count": int(phases.size),
        "grasp_centering": {
            "tolerance_mm": GRASP_CENTER_TOLERANCE_MM,
            "required_consecutive_frames": GRASP_CENTER_STABLE_FRAMES,
            "maximum_frames": int(_v6_args.max_grasp_centering_frames),
            "gripper_remains_open_until_aligned": True,
            "precontact_clearance_m": max(
                2.0 * float(args_cli.uipc_contact_d_hat_mm) * 1.0e-3,
                0.0005,
            ),
            "end_effector_frozen_after_lower_alignment": True,
        },
        "uipc_substeps_per_physx_frame": UIPC_SUBSTEPS,
        "timing": {
            "physx_dt_sec": 1.0 / max(float(args_cli.sim_hz), EPS),
            "uipc_substep_dt_sec": (
                1.0 / max(float(args_cli.sim_hz), EPS) / float(UIPC_SUBSTEPS)
            ),
            "abd_pose_history_dt_sec": (
                1.0 / max(float(args_cli.sim_hz), EPS) / float(UIPC_SUBSTEPS)
            ),
        },
        "uipc_substep_timeout_sec": float(_v6_args.uipc_substep_timeout_sec),
        "persistence": {
            "lightweight_checkpoint_interval_frames": LIGHTWEIGHT_CHECKPOINT_INTERVAL_FRAMES,
            "full_checkpoint": "at each completed phase, termination, and normal completion",
        },
        "pad_mount": "direct_link8_reference",
        "membrane_source": "UIPC_Pad/simulation/membrane_sim_mesh",
        "deformation_contract": "frozen_v5_new_7f",
        "force_model": "frozen_v5_new_7g",
        "field_model": "frozen_v5_new_9",
        "force_unit": "TU",
        "tactile_force_channel_order": ["Fx", "Fy", "Fz"],
        "video_generation": bool(_v6_args.save_diagnostic_video),
        "scene_video": (
            str(output_dir / "short_lift_hold_scene.mp4")
            if bool(_v6_args.save_diagnostic_video)
            else None
        ),
        "scene_video_frame_count": (
            int(phases.size) if bool(_v6_args.save_diagnostic_video) else 0
        ),
        "tactile_force_video_generation": bool(_v6_args.save_tactile_force_video),
        "tactile_force_videos": tactile_force_video_paths,
        "tactile_force_video_metrics": tactile_force_video_metrics,
        "repeat_count": 1,
        "frozen_hashes_before": FROZEN_HASHES_BEFORE,
        "frozen_hashes_after": _capture_frozen_hashes(),
    }
    _json_write(output_dir / "contact_loss_diagnosis.json", contact_diagnosis)
    _json_write(output_dir / "performance_statistics.json", performance)
    _json_write(output_dir / "short_acceptance.json", verdict)
    _json_write(output_dir / "metadata.json", metadata)
    _json_write(output_dir / "verdict.json", verdict)
    _json_write(
        output_dir / "partial_diagnostic_status.json",
        _partial_status(
            records,
            official_pass=official_pass,
            partial=partial,
            termination_reason=termination_reason,
        ),
    )
    print(json.dumps(verdict, indent=2), flush=True)


def _maximum_relative_error(
    measured: np.ndarray, reference: np.ndarray
) -> dict[str, float | int]:
    measured_array = np.asarray(measured, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    absolute = np.abs(measured_array - reference_array)
    nonzero = np.abs(reference_array) >= 1.0e-6
    relative = np.zeros_like(absolute)
    relative[nonzero] = absolute[nonzero] / np.abs(reference_array[nonzero])
    return {
        "maximum_absolute_error_tu": float(np.max(absolute, initial=0.0)),
        "maximum_nonzero_relative_error": float(np.max(relative, initial=0.0)),
        "maximum_near_zero_absolute_error_tu": float(
            np.max(absolute[~nonzero], initial=0.0)
        ),
        "nonzero_value_count": int(np.count_nonzero(nonzero)),
    }


def _maximum_centroid_drift(fields: np.ndarray) -> float:
    centroids = [value for value in (_field_centroid(field) for field in fields) if value is not None]
    if len(centroids) < 2:
        return 0.0
    return float(
        max(
            np.linalg.norm(first - second)
            for first_index, first in enumerate(centroids)
            for second in centroids[first_index + 1 :]
        )
    )


def _release_ratio(final_value: np.ndarray, peak_value: np.ndarray, tolerance: float) -> np.ndarray:
    result = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        if float(peak_value[axis]) > float(tolerance):
            result[axis] = float(final_value[axis] / peak_value[axis])
        elif float(final_value[axis]) >= float(tolerance):
            result[axis] = float("inf")
    return result


def _finish_outputs(
    *,
    output_dir: Path,
    contract_dir: Path,
    records: dict[str, list[object]],
    contract_verdict: dict[str, object],
    warmup_force: np.ndarray,
    warmup_contact: np.ndarray,
    stable_object_position: np.ndarray,
    stable_object_velocity: np.ndarray,
    waypoints: dict[str, np.ndarray],
    pad_root: str,
    membrane_mesh_path: str,
    mount_body_name: str,
    ee_body_name: str,
    hidden_visual_paths: list[str],
    back_tet_indices: np.ndarray,
    front_tet_indices: np.ndarray,
    tet_thickness_m: float,
    surface_thickness_m: float,
    back_constrained_coverage: float,
    front_constrained_fraction: float,
    attachment,
    mirror_state: dict[str, object],
    initialization_object_pose_write_count: int,
    runtime_object_pose_write_count: int,
    stability_steps_used: int,
    scene_video_path: Path,
) -> None:
    frozen_force_dir = output_dir / "frozen_7g_force"
    frozen_args = frozen_7g.build_parser().parse_args(
        [
            "--contract_dir",
            str(contract_dir),
            "--displacement_path",
            str(output_dir / "surface_deformation.npy"),
            "--output_dir",
            str(frozen_force_dir),
        ]
    )
    estimator_verdict = frozen_7g.run_cli(frozen_args)
    force_output_names = (
        "normal_compression.npy",
        "shear_displacement.npy",
        "contact_activation_weight.npy",
        "vertex_deformation_volume_contribution.npy",
        "normal_deformation_volume.npy",
        "shear_deformation_volume.npy",
        "force_pad_local.npy",
        "tactile_force_channels.npy",
    )
    for filename in force_output_names:
        np.save(
            output_dir / filename,
            np.load(frozen_force_dir / filename, allow_pickle=False),
        )

    force_pad_local = np.asarray(
        np.load(output_dir / "force_pad_local.npy", allow_pickle=False), dtype=np.float64
    )
    tactile_total = np.asarray(
        np.load(output_dir / "tactile_force_channels.npy", allow_pickle=False),
        dtype=np.float64,
    )
    vertex_volume = np.asarray(
        np.load(
            frozen_force_dir / "vertex_deformation_volume_contribution.npy",
            allow_pickle=False,
        ),
        dtype=np.float64,
    )
    activation = np.asarray(
        np.load(frozen_force_dir / "contact_activation_weight.npy", allow_pickle=False),
        dtype=np.float64,
    )
    front_surface_indices = np.asarray(
        np.load(output_dir / "front_surface_indices.npy", allow_pickle=False), dtype=np.int64
    )
    rest_surface = np.asarray(
        np.load(output_dir / "rest_surface_vertices_pad_local.npy", allow_pickle=False),
        dtype=np.float64,
    )
    gains = tactile_field_v9._estimator_gains(output_dir)
    pad_vertices_full, tactile_vertices_full = tactile_field_v9.tactile_vertex_contributions(
        vertex_volume, **gains
    )
    pad_vertices = pad_vertices_full[:, front_surface_indices]
    tactile_vertices = tactile_vertices_full[:, front_surface_indices]
    activation_vertices = activation[:, front_surface_indices]
    front_yz = rest_surface[front_surface_indices, 1:3]
    plan = tactile_field_v9.build_gaussian_splat_plan(
        front_yz,
        height=int(args_cli.field_height),
        width=int(args_cli.field_width),
        sigma_cells=float(args_cli.gaussian_sigma_cells),
        truncate_sigma=float(args_cli.gaussian_truncate_sigma),
    )
    tactile_field = tactile_field_v9.splat_vertex_values(tactile_vertices, plan)
    activation_field = tactile_field_v9.splat_vertex_values(activation_vertices, plan)
    shear_magnitude = np.linalg.norm(tactile_field[..., :2], axis=3)
    field_arrays = {
        "force_pad_local_vertices.npy": pad_vertices,
        "tactile_force_vertices.npy": tactile_vertices,
        "tactile_force_field_tu.npy": tactile_field,
        "tactile_fx_field_tu.npy": tactile_field[..., 0],
        "tactile_fy_field_tu.npy": tactile_field[..., 1],
        "tactile_fz_field_tu.npy": tactile_field[..., 2],
        "tactile_shear_magnitude_tu.npy": shear_magnitude,
        "contact_activation_field.npy": activation_field,
        "front_surface_yz.npy": front_yz,
        "field_grid_y_m.npy": plan.grid_y_m,
        "field_grid_z_m.npy": plan.grid_z_m,
    }
    for filename, value in field_arrays.items():
        np.save(output_dir / filename, value)
    # Canonical v6.1 filenames; values are direct aliases of frozen 7f/7g/v9 outputs.
    np.save(output_dir / "surface_displacement_pad_local.npy", np.asarray(records["surface_deformation"]))
    np.save(output_dir / "normal_deformation.npy", np.load(output_dir / "normal_compression.npy", allow_pickle=False))
    np.save(output_dir / "vertex_area.npy", np.load(contract_dir / "vertex_area.npy", allow_pickle=False))
    np.save(output_dir / "front_surface_mask.npy", np.load(contract_dir / "front_surface_mask.npy", allow_pickle=False))
    np.save(output_dir / "tactile_force_field.npy", tactile_field)
    np.save(output_dir / "tactile_fx_field.npy", tactile_field[..., 0])
    np.save(output_dir / "tactile_fy_field.npy", tactile_field[..., 1])
    np.save(output_dir / "tactile_fz_field.npy", tactile_field[..., 2])

    phases = np.asarray(records["phase"], dtype=str)
    cycles = np.asarray(records["cycle_index"], dtype=np.int64)
    phase_frames: dict[str, dict[str, int]] = {}
    for name, expected_count in PHASE_SPECS:
        indices = np.flatnonzero(phases == name)
        if indices.size != expected_count:
            raise RuntimeError(
                f"Phase {name} has {indices.size} frames, expected {expected_count}"
            )
        phase_frames[name] = {
            "start": int(indices[0]), "end": int(indices[-1]), "count": int(indices.size)
        }
    _json_write(output_dir / "phase_frames.json", phase_frames)
    # Video is deliberately outside the official acceptance path.  This keeps a
    # failed episode from producing a video that could be mistaken for formal data.
    video_metrics = {"generated": False, "reason": "not_part_of_v6.1_acceptance"}
    scene_video_metrics = (
        tactile_field_v9._decode_video(scene_video_path)
        if scene_video_path.is_file()
        else {"opened": False, "decoded_frame_count": 0, "reported_frame_count": 0}
    )

    object_pose = np.asarray(records["object_pose_w"], dtype=np.float64)
    object_velocity = np.asarray(records["object_velocity_w"], dtype=np.float64)
    object_lift = np.asarray(records["object_lift_height_mm"], dtype=np.float64)
    object_distance = np.asarray(records["object_gripper_distance_mm"], dtype=np.float64)
    object_relative = np.asarray(records["object_relative_pose_to_gripper"], dtype=np.float64)
    support_clearance = np.asarray(records["support_clearance_mm"], dtype=np.float64)
    measured_opening = np.asarray(records["gripper_opening_mm"], dtype=np.float64)
    commanded_opening = np.asarray(records["commanded_opening_mm"], dtype=np.float64)
    contact_count = np.asarray(records["contact_vertex_count"], dtype=np.int64)
    max_penetration = np.asarray(records["max_penetration_mm"], dtype=np.float64)
    max_front_deformation = np.asarray(records["max_front_deformation_mm"], dtype=np.float64)
    sync_position = np.asarray(
        records["physx_uipc_object_position_error_mm"], dtype=np.float64
    )
    sync_orientation = np.asarray(
        records["physx_uipc_object_orientation_error_deg"], dtype=np.float64
    )
    back_target_error = np.asarray(records["back_target_error_mm"], dtype=np.float64)
    field_total = np.sum(tactile_field, axis=(1, 2), dtype=np.float64)
    vertex_total = np.sum(tactile_vertices, axis=1, dtype=np.float64)
    active_cells = np.count_nonzero(activation_field > 1.0e-12, axis=(1, 2))
    force_magnitude = np.linalg.norm(force_pad_local, axis=1)
    np.save(output_dir / "field_integral_force.npy", field_total)
    np.save(output_dir / "field_conservation_error.npy", field_total - tactile_total)
    zero_tolerance = float(args_cli.accept_max_warmup_force_tu)

    approach_indices = np.flatnonzero(
        np.isin(phases, ("SETTLE_READY", "HOME", "APPROACH_PICK", "LOWER_TO_GRASP"))
    )
    close_indices = np.flatnonzero(np.isin(phases, ("CLOSE_GRIPPER", "CONFIRM_GRASP")))
    closing_motion_indices = np.flatnonzero(phases == "CLOSE_GRIPPER")
    check_indices = np.flatnonzero(phases == "LIFT_OBJECT")
    hold_indices = np.flatnonzero(phases == "HOLD_LIFTED")
    support_indices = np.flatnonzero(phases == "CONFIRM_SUPPORT")
    recovery_indices = np.flatnonzero(phases == "RETREAT_AND_RECOVER")
    if any(
        indices.size == 0
        for indices in (
            approach_indices,
            close_indices,
            closing_motion_indices,
            check_indices,
            hold_indices,
            support_indices,
            recovery_indices,
        )
    ):
        raise RuntimeError("Formal phase history is incomplete")

    closure = float(args_cli.gripper_opening_mm) - commanded_opening[closing_motion_indices]
    close_fz_spearman = mainline_v9._spearman(
        closure, tactile_total[closing_motion_indices, 2]
    )
    approach_nonzero_ratio = float(
        np.mean(force_magnitude[approach_indices] >= zero_tolerance)
    )
    hold_fz = tactile_total[hold_indices, 2]
    hold_fz_cv = _coefficient_of_variation(hold_fz)
    hold_relative_position = object_relative[hold_indices, :3]
    hold_position_drift_mm = float(
        np.max(
            np.linalg.norm(
                hold_relative_position - hold_relative_position[0].reshape(1, 3), axis=1
            )
        )
        * 1000.0
    )
    hold_relative_quat = object_relative[hold_indices, 3:7]
    hold_orientation_drift_deg = float(
        max(
            mainline_v9._quat_angle_error_deg(
                tuple(hold_relative_quat[0]), tuple(quat)
            )
            for quat in hold_relative_quat
        )
    )
    hold_fz_fields = tactile_field[hold_indices, ..., 2]
    hold_centroid_drift = _maximum_centroid_drift(hold_fz_fields)
    third = max(1, hold_indices.size // 3)
    hold_start_field = np.mean(hold_fz_fields[:third], axis=0)
    hold_end_field = np.mean(hold_fz_fields[-third:], axis=0)
    hold_field_correlation = _field_correlation(hold_start_field, hold_end_field)

    gravity_tangent_ratios: list[float] = []
    gravity_tactile_directions: list[np.ndarray] = []
    for pad_pose in np.asarray(records["pad_pose_w"], dtype=np.float64)[hold_indices]:
        gravity_pad_l = _quat_matrix(pad_pose[3:7]).T @ np.asarray((0.0, 0.0, -1.0))
        gravity_tangent_ratios.append(float(np.linalg.norm(gravity_pad_l[1:3])))
        gravity_tactile_directions.append(
            np.asarray((gravity_pad_l[1], -gravity_pad_l[2]), dtype=np.float64)
        )
    gravity_tangent_ratio = float(np.mean(gravity_tangent_ratios))
    mean_gravity_tactile_direction = np.mean(gravity_tactile_directions, axis=0)
    mean_tactile_shear = np.mean(tactile_total[hold_indices, :2], axis=0)
    tangent_direction_cosine = float(
        np.dot(mean_tactile_shear, mean_gravity_tactile_direction)
        / max(
            np.linalg.norm(mean_tactile_shear)
            * np.linalg.norm(mean_gravity_tactile_direction),
            EPS,
        )
    )
    warmup_tactile_shear = np.column_stack((warmup_force[:, 1], -warmup_force[:, 2]))
    warmup_tangent_noise = float(
        np.sqrt(np.mean(np.sum(warmup_tactile_shear * warmup_tactile_shear, axis=1)))
    )
    mean_tangent_tu = float(np.mean(np.linalg.norm(tactile_total[hold_indices, :2], axis=1)))
    tangent_gate_applicable = gravity_tangent_ratio > 0.20
    tangent_gate_passed = bool(
        not tangent_gate_applicable
        or (
            mean_tangent_tu > 3.0 * warmup_tangent_noise
            and tangent_direction_cosine > 0.5
        )
    )

    peak_axis = np.max(np.abs(tactile_total), axis=0)
    final_axis = np.max(np.abs(tactile_total[recovery_indices[-min(3, recovery_indices.size) :]]), axis=0)
    final_axis_ratio = _release_ratio(final_axis, peak_axis, zero_tolerance)
    final_deformation = np.asarray(records["surface_deformation"][-1], dtype=np.float64)
    final_deformation_mm = float(
        np.max(np.linalg.norm(final_deformation[front_surface_indices], axis=1)) * 1000.0
    )

    vertex_conservation = _maximum_relative_error(vertex_total, tactile_total)
    field_conservation = _maximum_relative_error(field_total, vertex_total)
    frozen_hashes_after = _capture_frozen_hashes()
    frozen_hashes_unchanged = FROZEN_HASHES_BEFORE == frozen_hashes_after
    max_sync_position = float(np.max(sync_position))
    max_sync_orientation = float(np.max(sync_orientation))
    max_back_target_error = float(np.max(back_target_error))
    warmup_force_magnitude = np.linalg.norm(warmup_force, axis=1)
    maximum_object_lift = float(np.max(object_lift[check_indices]))
    minimum_object_lift = float(np.min(object_lift[check_indices]))
    maximum_check_distance = float(np.max(object_distance[check_indices]))
    minimum_check_vertical_velocity = float(np.min(object_velocity[check_indices, 2]))
    minimum_check_support_clearance = float(np.min(support_clearance[check_indices]))
    support_tail = support_indices[-min(5, support_indices.size) :]
    support_confirmed = bool(
        np.max(np.abs(support_clearance[support_tail]))
        <= float(_v6_args.support_contact_tolerance_mm)
        and np.max(np.abs(object_velocity[support_tail, 2])) < 0.02
    )

    attachment_sync_position = np.asarray(
        records["pad_attachment_position_error_mm"], dtype=np.float64
    )
    attachment_sync_orientation = np.asarray(
        records["pad_attachment_orientation_error_deg"], dtype=np.float64
    )
    relative_sync_position = np.asarray(
        records["physx_uipc_relative_position_error_mm"], dtype=np.float64
    )
    relative_sync_orientation = np.asarray(
        records["physx_uipc_relative_orientation_error_deg"], dtype=np.float64
    )
    no_contact_indices = approach_indices[contact_count[approach_indices] == 0]
    hold_contact = contact_count[hold_indices] > 0
    maximum_consecutive_hold_loss = 0
    current_loss = 0
    for has_contact in hold_contact:
        current_loss = 0 if has_contact else current_loss + 1
        maximum_consecutive_hold_loss = max(maximum_consecutive_hold_loss, current_loss)
    recovery_tail = recovery_indices[-20:]
    recovery_force_norm = np.linalg.norm(tactile_total[recovery_tail], axis=1)
    field_integral_norm = np.linalg.norm(field_total, axis=1)
    recovery_field_norm = field_integral_norm[recovery_tail]
    deformation_norm_mm = np.max(
        np.linalg.norm(np.asarray(records["surface_deformation"], dtype=np.float64), axis=2),
        axis=1,
    ) * 1000.0
    recovery_force_limit = max(0.001, 0.02 * float(np.max(force_magnitude, initial=0.0)))
    recovery_field_limit = max(0.001, 0.02 * float(np.max(field_integral_norm, initial=0.0)))
    recovery_deformation_limit = max(0.005, 0.02 * float(np.max(deformation_norm_mm, initial=0.0)))
    finite_arrays = (
        object_pose, object_velocity, tactile_total, tactile_field, max_penetration,
        sync_position, sync_orientation, attachment_sync_position,
        attachment_sync_orientation, relative_sync_position, relative_sync_orientation,
    )

    gate_checks = {
        "gate_1_frozen_input_integrity": bool(
            frozen_hashes_unchanged
            and contract_verdict.get("deformation_contract_passed", False)
            and estimator_verdict.get("deformation_based_force_estimator_passed", False)
        ),
        "gate_2_formal_record_readiness": bool(
            warmup_force.shape[0] == 30
            and np.max(warmup_contact, initial=0) == 0
            and np.max(warmup_force_magnitude, initial=0.0) < 0.001
            and contact_count[0] == 0
            and force_magnitude[0] < 0.001
        ),
        "gate_3_physx_dynamics_authority": bool(
            initialization_object_pose_write_count >= 1
            and runtime_object_pose_write_count == 0
        ),
        "gate_4_object_mirror_absolute_sync": bool(
            max_sync_position <= 0.05 and max_sync_orientation <= 0.05
        ),
        "gate_5_pad_attachment_sync": bool(
            np.max(attachment_sync_position, initial=0.0) <= 0.05
            and np.max(attachment_sync_orientation, initial=0.0) <= 0.05
        ),
        "gate_6_pad_object_relative_sync": bool(
            np.max(relative_sync_position, initial=0.0) <= 0.05
            and np.max(relative_sync_orientation, initial=0.0) <= 0.05
        ),
        "gate_7_contact_geometry_safety": bool(
            np.max(max_penetration, initial=0.0) <= 0.15
            and all(np.isfinite(value).all() for value in finite_arrays)
            and float(field_conservation["maximum_nonzero_relative_error"]) < 0.01
        ),
        "gate_8_precontact_zero_tactile": bool(
            no_contact_indices.size > 0
            and np.max(force_magnitude[no_contact_indices], initial=0.0) < 0.001
            and np.max(field_integral_norm[no_contact_indices], initial=0.0) < 0.001
            and np.count_nonzero(active_cells[no_contact_indices]) == 0
        ),
        "gate_9_real_grasp_and_lift": bool(
            maximum_object_lift > 30.0
            and maximum_check_distance < float(_v6_args.accept_gripper_distance_mm)
            and np.max(contact_count[close_indices], initial=0) > 0
            and runtime_object_pose_write_count == 0
        ),
        "gate_10_lifted_hold": bool(
            hold_indices.size == 50
            and np.count_nonzero(hold_contact) >= 48
            and maximum_consecutive_hold_loss <= 2
            and hold_position_drift_mm
            <= float(_v6_args.accept_max_hold_position_drift_mm)
            and np.isfinite(hold_fz).all()
        ),
        "gate_11_normal_return": bool(
            support_confirmed
            and np.all(commanded_opening[support_indices] <= float(_v6_args.gripper_closed_mm) + 1.0e-6)
        ),
        "gate_12_release_recovery": bool(
            recovery_tail.size == 20
            and np.all(contact_count[recovery_tail] == 0)
            and np.max(recovery_force_norm, initial=0.0) <= recovery_force_limit
            and np.max(recovery_field_norm, initial=0.0) <= recovery_field_limit
            and np.max(deformation_norm_mm[recovery_tail], initial=0.0) <= recovery_deformation_limit
            and abs(float(object_velocity[-1, 2])) < 0.02
        ),
    }
    single_episode_passed = bool(all(gate_checks.values()))

    free_object_metrics = {
        "initial_object_position_w": stable_object_position.tolist(),
        "initial_object_velocity_w": stable_object_velocity.tolist(),
        "maximum_lift_height_mm_during_check": maximum_object_lift,
        "minimum_lift_height_mm_during_check": minimum_object_lift,
        "maximum_object_gripper_distance_mm_during_check": maximum_check_distance,
        "minimum_vertical_velocity_m_s_during_check": minimum_check_vertical_velocity,
        "minimum_support_clearance_mm_during_check": minimum_check_support_clearance,
        "support_confirmed_before_release": support_confirmed,
        "maximum_physx_uipc_position_error_mm": max_sync_position,
        "maximum_physx_uipc_orientation_error_deg": max_sync_orientation,
        "runtime_object_pose_write_count": int(runtime_object_pose_write_count),
    }
    tactile_metrics = {
        "warmup_stable_frame_count": int(warmup_force.shape[0]),
        "warmup_maximum_force_magnitude_tu": float(
            np.max(warmup_force_magnitude, initial=0.0)
        ),
        "approach_nonzero_frame_ratio": approach_nonzero_ratio,
        "close_fz_spearman": close_fz_spearman,
        "hold_fz_cv": hold_fz_cv,
        "hold_object_relative_position_drift_mm": hold_position_drift_mm,
        "hold_object_relative_orientation_drift_deg": hold_orientation_drift_deg,
        "hold_field_centroid_drift_cells": hold_centroid_drift,
        "hold_field_start_end_correlation": hold_field_correlation,
        "gravity_tangent_projection_ratio": gravity_tangent_ratio,
        "tangent_gate_applicable": tangent_gate_applicable,
        "mean_tangent_tu": mean_tangent_tu,
        "warmup_tangent_noise_tu": warmup_tangent_noise,
        "tangent_direction_cosine": tangent_direction_cosine,
        "final_axis_to_peak_ratio": final_axis_ratio.tolist(),
        "final_membrane_deformation_mm": final_deformation_mm,
        "field_active_cells": active_cells.tolist(),
        "vertex_sum_vs_channels": vertex_conservation,
        "field_sum_vs_vertices": field_conservation,
        "video": video_metrics,
        "scene_video": scene_video_metrics,
    }

    metadata = {
        "version": "v6.1_free_object_full_cycle_tactile_validation",
        "script_version": "v6_1_free_object_full_cycle_tactile_validation",
        "robot_control_source": "previous_validated_piper_grasp_flow",
        "grasp_control_reused_only": True,
        "pad_mount": "direct_link8_reference",
        "membrane_source": "UIPC_Pad/simulation/membrane_sim_mesh",
        "pad_root": pad_root,
        "membrane_mesh_path": membrane_mesh_path,
        "pad_mount_parameters": {
            "pad_x_mm": float(args_cli.pad_x_mm),
            "pad_y_mm": float(args_cli.pad_y_mm),
            "pad_z_mm": float(args_cli.pad_z_mm),
            "pad_roll_deg": float(args_cli.pad_roll_deg),
            "pad_pitch_deg": float(args_cli.pad_pitch_deg),
            "pad_yaw_deg": float(args_cli.pad_yaw_deg),
        },
        "runtime_duplicate_membrane": False,
        "intermediate_mount_frame_used": False,
        "rigid_object_dynamics_source": "PhysX",
        "uipc_object_pose_source": "mirrored_from_physx",
        "coupling_mode": "one_way_physx_to_uipc",
        "physx_dynamics_authority": True,
        "uipc_contact_feedback_to_physx": False,
        "uipc_force_feedback_to_physx": False,
        "uipc_mirror_driver": "kinematic_abd_pose_pair",
        "uipc_mirror_pose_history_preserved": True,
        "pose_interpolation_per_uipc_substep": True,
        "pad_attachment_interpolation_per_uipc_substep": True,
        "uipc_substeps_per_physx_frame": 3,
        "runtime_physx_object_pose_writes": int(runtime_object_pose_write_count),
        "runtime_physx_object_velocity_writes": 0,
        "fixed_joint_used": False,
        "baseline_subtraction_used": False,
        "deformation_contract": "frozen_v5_new_7f",
        "official_force_model": "frozen_v5_new_7g",
        "official_field_model": "frozen_v5_new_9",
        "force_source": "uipc_membrane_surface_deformation_reduced_order",
        "force_unit": "TU",
        "newton_calibrated": False,
        "damping_used": False,
        "force_direction": "object_on_sensor",
        "tactile_channel_axes": {
            "Fx": "pad_local_positive_Y",
            "Fy": "pad_local_negative_Z",
            "Fz": "pad_local_negative_X",
        },
        "contact_geometry_role": "diagnostic_only",
        "native_uipc_contact_force_used": False,
        "camera_rgb_force_role": "none",
        "runtime_object_pose_overwrite": False,
        "grasp_assist_used": False,
        "object": {
            "path": OBJECT_PATH,
            "shape": "cylinder",
            "axis": "Z",
            "radius_mm": float(_v6_args.object_radius_mm),
            "height_mm": float(_v6_args.object_height_mm),
            "mass_kg": float(_v6_args.object_mass_kg),
            "dynamic": True,
            "kinematic": False,
            "gravity_enabled": True,
            "ccd_enabled": True,
            "fixed_joint": False,
            "initialization_pose_write_count": int(initialization_object_pose_write_count),
            "official_trajectory_runtime_pose_write_count": int(runtime_object_pose_write_count),
        },
        "uipc_mirror": {
            "root": UIPC_OBJECT_ROOT,
            "role": "contact_geometry_mirror_only",
            "driver": "kinematic_abd_pose_pair",
            "precomputed_mesh_vertices": 66,
            "precomputed_mesh_tetrahedra": 96,
            "runtime_full_mirror_vertex_write": False,
            "pose_interpolation_per_uipc_substep": True,
            "sync_fail_fast_position_mm": 0.05,
            "sync_fail_fast_orientation_deg": 0.05,
            "runtime_physx_object_pose_write": False,
            "same_frame_update_before_uipc_solve": True,
            "pose_sync_update_count": int(mirror_state["update_count"]),
            "pose_sync_last_frame": int(mirror_state["last_frame"]),
        },
        "membrane": {
            "structured_cells": [1, 22, 26],
            "youngs_modulus_mpa": 0.05,
            "poisson_ratio": 0.49,
            "density": 1050.0,
            "attachment_vertex_count": int(back_tet_indices.size),
            "front_tet_vertex_count": int(front_tet_indices.size),
            "back_constrained_coverage": back_constrained_coverage,
            "front_constrained_fraction": front_constrained_fraction,
            "tet_thickness_mm": float(tet_thickness_m * 1000.0),
            "surface_thickness_mm": float(surface_thickness_m * 1000.0),
        },
        "attachment": {
            "body_name": mount_body_name,
            "maximum_target_error_mm": max_back_target_error,
            "animation_update_count": int(attachment.animation_update_count),
            "animation_last_frame": int(attachment.animation_last_frame),
        },
        "ik": {
            "end_effector_body": ee_body_name,
            "tip_offset_m": [float(v) for v in _v6_args.piper_tip_offset],
            "command_type": "position",
            "method": "dls",
        },
        "waypoints_w": {name: value.tolist() for name, value in waypoints.items()},
        "phase_order": [name for name, _ in PHASE_SPECS],
        "episode_index": int(_v6_args.episode_index),
        "run_uuid": RUN_UUID,
        "process_id": int(os.getpid()),
        "official_frame_count": int(phases.size),
        "formal_frame_count": int(phases.size),
        "readiness_warmup_recorded": False,
        "warmup_stability_steps_used": int(stability_steps_used),
        "uipc_unrecorded_prewarm_steps": int(stability_steps_used),
        "rest_surface_capture_protocol": (
            "INIT_UNRECORDED -> no-contact UIPC prewarm -> 30 consecutive stable frames "
            "-> capture rest surface -> clear initialization cache -> formal record"
        ),
        "field": {
            "size": [81, 65],
            "mapping": "force_conserving_gaussian_splat",
            "sigma_cells": 1.25,
            "fixed_global_video_range": True,
        },
        "hidden_visual_paths": hidden_visual_paths,
        "frozen_hashes_before": FROZEN_HASHES_BEFORE,
        "frozen_hashes_after": frozen_hashes_after,
        "gate_1_12_pass": single_episode_passed,
        "gate_13_five_episode_pass": False,
        "outputs": {
            "scene_video": str(scene_video_path),
            "tactile_video": str(output_dir / "tactile_fxyz_composite_sequence.mp4"),
        },
    }
    verdict = {
        "official_pass": single_episode_passed,
        "v6_1_gate_1_12_passed": single_episode_passed,
        "v6_1_gate_13_five_episode_passed": False,
        "checks": gate_checks,
        "gate_13_repeatability": {
            "evaluated": False,
            "reason": (
                "Run five independent application processes with episode_index 0..4, then "
                "aggregate their output directories; a single process is not reported as "
                "five independent resets."
            ),
            "required_episode_count": 5,
        },
        "observed": {
            "free_object": free_object_metrics,
            "tactile": tactile_metrics,
        },
        "force_source": "uipc_membrane_surface_deformation_reduced_order",
        "force_unit": "TU",
        "newton_calibrated": False,
        "damping_used": False,
    }
    _json_write(output_dir / "free_object_grasp_metrics.json", free_object_metrics)
    _json_write(output_dir / "tactile_grasp_metrics.json", tactile_metrics)
    _json_write(output_dir / "field_conservation_metrics.json", {
        "vertex_sum_vs_tactile_force_channels": vertex_conservation,
        "field_sum_vs_vertex_sum": field_conservation,
    })
    _json_write(output_dir / "metadata.json", metadata)
    _json_write(output_dir / "verdict.json", verdict)
    _json_write(output_dir / "summary.json", {**metadata, "verdict": verdict})
    _json_write(output_dir / "gate_results.json", verdict)
    _json_write(output_dir / "episode_summary.json", {**metadata, "verdict": verdict})
    _json_write(output_dir / "frozen_input_hashes.json", frozen_hashes_after)
    if not single_episode_passed:
        scene_video_path.unlink(missing_ok=True)
        (output_dir / "tactile_fxyz_composite_sequence.mp4").unlink(missing_ok=True)
        _json_write(
            output_dir / "failure_snapshot.json",
            {
                "official_pass": False,
                "failed_gates": [name for name, passed in gate_checks.items() if not passed],
                "completed_formal_frames": int(phases.size),
            },
        )
    print(json.dumps(verdict, indent=2), flush=True)
    if bool(args_cli.fail_on_verdict_fail) and not single_episode_passed:
        raise RuntimeError(f"v6.1 single-episode acceptance failed: {gate_checks}")
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        try:
            error_output_dir = Path(args_cli.output_dir).expanduser().resolve()
            error_output_dir.mkdir(parents=True, exist_ok=True)
            _json_write(
                error_output_dir / "error.json",
                {
                    "script_version": "v6_1c_contact_timing_fix",
                    "official_pass": False,
                    "partial_diagnostic_only": True,
                    "error": traceback.format_exc(),
                },
            )
            if not (error_output_dir / "failure_snapshot.json").exists():
                _json_write(
                    error_output_dir / "failure_snapshot.json",
                    {
                        "official_pass": False,
                        "error": traceback.format_exc(),
                    },
                )
        except BaseException:
            pass
        simulation_app.close()
        raise
