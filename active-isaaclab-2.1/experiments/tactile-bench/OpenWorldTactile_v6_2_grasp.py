from __future__ import annotations

"""V6.2 Pad-local UIPC membrane with closed-loop PhysX rigid-body coupling.

PhysX is the only owner of the Piper and free cylinder state.  UIPC owns the
single membrane at ``link8/UIPC_Pad/simulation/membrane_sim_mesh``.  UIPC keeps
that membrane fixed in the Pad frame and receives the PhysX cylinder only as a
kinematic boundary at ``T_pad_object = inverse(T_world_pad) T_world_object``.
Each 60 Hz record interval contains eight alternating PhysX/UIPC coupling
substeps.  UIPC's native Pad-local contact reaction is rotated back to world,
under-relaxed, and returned to the PhysX cylinder on the next coupling substep.
The cylinder is never pose-written during formal motion.

The gripper follows ordinary finite-stiffness joint targets; no gap or tactile
feedback changes its command.  Frozen 7g remains a separate TU-valued tactile
estimator and is never used as a PhysX force.
"""

import argparse
import atexit
import datetime as datetime_module
import itertools
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

import numpy as np


def _install_timestamped_terminal() -> None:
    """Prefix every stdout/stderr line, including native-library logs, with wall time."""
    stdout_fd = int(sys.stdout.fileno())
    stderr_fd = int(sys.stderr.fileno())
    sys.stdout.flush()
    sys.stderr.flush()
    original_fd = os.dup(stdout_fd)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, stdout_fd)
    os.dup2(write_fd, stderr_fd)
    os.close(write_fd)

    def pump() -> None:
        pending = b""
        try:
            while True:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    break
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    timestamp = datetime_module.datetime.now().astimezone().isoformat(
                        sep=" ", timespec="milliseconds"
                    )
                    os.write(original_fd, f"[{timestamp}] ".encode() + line + b"\n")
            if pending:
                timestamp = datetime_module.datetime.now().astimezone().isoformat(
                    sep=" ", timespec="milliseconds"
                )
                os.write(original_fd, f"[{timestamp}] ".encode() + pending + b"\n")
        finally:
            os.close(read_fd)

    terminal_thread = threading.Thread(
        target=pump, name="v62-timestamped-terminal", daemon=True
    )
    terminal_thread.start()

    def restore() -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(original_fd, stdout_fd)
        os.dup2(original_fd, stderr_fd)
        terminal_thread.join(timeout=2.0)
        os.close(original_fd)

    atexit.register(restore)


_install_timestamped_terminal()


_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--object_radius_mm", type=float, default=15.0)
_parser.add_argument("--object_height_mm", type=float, default=105.0)
_parser.add_argument("--object_mass_kg", type=float, default=0.018)
_parser.add_argument("--object_x", type=float, default=0.34)
_parser.add_argument("--object_y", type=float, default=-0.02)
_parser.add_argument("--object_support_clearance_mm", type=float, default=0.25)
_parser.add_argument("--object_friction", type=float, default=2.0)
_parser.add_argument("--gripper_closed_mm", type=float, default=17.5)
_parser.add_argument("--home_ee_x", type=float, default=0.28)
_parser.add_argument("--home_ee_y", type=float, default=0.0)
_parser.add_argument("--home_ee_z", type=float, default=0.20)
_parser.add_argument("--approach_z", type=float, default=0.20)
_parser.add_argument("--lift_distance_mm", type=float, default=40.0)
_parser.add_argument(
    "--lift_frames",
    type=int,
    default=240,
    help="Smoothstep lift duration in 60 Hz record frames.",
)
_parser.add_argument("--piper_gripper_body", type=str, default="gripper_base")
_parser.add_argument("--piper_tip_offset", type=float, nargs=3, default=[0.0, 0.0, 0.1358])
_parser.add_argument("--initial_settle_frames", type=int, default=90)
_parser.add_argument("--uipc_warmup_frames", type=int, default=30)
_parser.add_argument("--close_frames", type=int, default=90)
_parser.add_argument("--contact_force_epsilon_n", type=float, default=1.0e-6)
_parser.add_argument(
    "--uipc_feedback_relaxation",
    type=float,
    default=1.0,
    help="Per-coupling-substep under-relaxation factor for raw UIPC reaction feedback.",
)
_parser.add_argument(
    "--uipc_feedback_force_limit_n",
    type=float,
    default=0.25,
    help=(
        "Norm limit for the delayed UIPC force applied to PhysX. The raw UIPC "
        "reaction remains unmodified in the recorded reaction arrays."
    ),
)
_parser.add_argument("--gripper_drive_stiffness", type=float, default=200.0)
_parser.add_argument("--gripper_drive_damping", type=float, default=8.0)
_parser.add_argument("--gripper_effort_limit_n", type=float, default=6.0)
_parser.add_argument("--gripper_closing_velocity_m_s", type=float, default=0.03)
_parser.add_argument(
    "--uipc_substeps_per_record",
    type=int,
    default=8,
    help=(
        "Alternating PhysX/UIPC coupling substeps per recorded 60 Hz interval. "
        "At least eight are required so the external cylinder cannot cross "
        "the 0.1 mm contact layer in one coupling step."
    ),
)
_parser.add_argument("--uipc_substep_timeout_sec", type=float, default=60.0)
_parser.add_argument(
    "--slow_frame_threshold_sec",
    "--maximum_frame_time_sec",
    dest="slow_frame_threshold_sec",
    type=float,
    default=0.5,
    help="Emit detailed diagnostics when a warmup or formal frame exceeds this wall time.",
)
_parser.add_argument(
    "--trace_uipc_steps",
    action="store_true",
    help="Print begin/end timing for every native UIPC substep.",
)
_parser.add_argument(
    "--max_formal_frames",
    type=int,
    default=0,
    help="Diagnostic cap; zero runs the complete motion program.",
)
_parser.add_argument(
    "--loop_forever",
    action="store_true",
    help="Repeat the motion program until the Isaac application is closed or Ctrl+C is pressed.",
)
_v62_args, _v9_argv = _parser.parse_known_args()

# V6.2 has no camera path.  Viewport rendering remains available through the
# inherited --render_viewport option without enabling a camera sensor.
_v9_argv = [
    value
    for value in _v9_argv
    if value not in ("--save_camera_rgb", "--no_save_camera_rgb")
]
_v9_argv.append("--no_save_camera_rgb")

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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from pxr import PhysxSchema, UsdGeom, UsdPhysics


args_cli = mainline_v9.args_cli
simulation_app = mainline_v9.simulation_app
frozen_7g = mainline_v9.frozen_7g

OBJECT_PATH = "/World/envs/env_0/GraspCylinder"
UIPC_BOUNDARY_ROOT = "/World/UIPC_v62_ExternalBoundaries/GraspCylinder"
UIPC_BOUNDARY_MESH_PATH = f"{UIPC_BOUNDARY_ROOT}/mesh"
OPPOSING_PAD_PATH = f"{mainline_v9.ROBOT_ROOT}/openworldtactile_case_left/openworldtactile_pad_visual"
OPPOSING_CONTACT_BODY_PATH = f"{mainline_v9.ROBOT_ROOT}/openworldtactile_case_left"
_OWT_MAINLINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OWT_ROBOT_USD = (
    _OWT_MAINLINE_ROOT
    / "packages"
    / "assets"
    / "openworldtactile_assets"
    / "openworldtactile_assets"
    / "data"
    / "Robots"
    / "AgileX"
    / "Piper"
    / "piper_openworldtactile.usda"
)
EPS = 1.0e-12
UIPC_SUBSTEPS = int(_v62_args.uipc_substeps_per_record)
# The inherited v9 parser has the legacy default of three.  Keep its namespace
# truthful for any shared helper that inspects this setting.
args_cli.uipc_substeps_per_record = UIPC_SUBSTEPS
STRUCTURED_CELLS = (1, 22, 26)
MOTION_STAGE_SPECS = (
    ("settle", 15),
    ("approach", 30),
    ("lower", 50),
    ("close", 90),
    ("hold", 20),
    ("lift", 60),
    ("hold_lifted", 20),
    ("release", 30),
    ("retreat", 30),
    ("recovery", 20),
)


def _validate_args() -> None:
    for name in (
        "object_radius_mm",
        "object_height_mm",
        "object_mass_kg",
        "object_friction",
        "contact_force_epsilon_n",
        "uipc_substep_timeout_sec",
        "uipc_feedback_force_limit_n",
        "slow_frame_threshold_sec",
        "gripper_drive_stiffness",
        "gripper_drive_damping",
        "gripper_effort_limit_n",
        "gripper_closing_velocity_m_s",
    ):
        if float(getattr(_v62_args, name)) <= 0.0:
            raise ValueError(f"--{name} must be > 0")
    for name in (
        "initial_settle_frames",
        "uipc_warmup_frames",
        "close_frames",
        "lift_frames",
    ):
        if int(getattr(_v62_args, name)) <= 0:
            raise ValueError(f"--{name} must be > 0")
    if int(_v62_args.max_formal_frames) < 0:
        raise ValueError("--max_formal_frames must be >= 0")
    if not 0.0 < float(_v62_args.uipc_feedback_relaxation) <= 1.0:
        raise ValueError("--uipc_feedback_relaxation must be in (0, 1]")
    if not 0.0 <= float(_v62_args.gripper_closed_mm) < float(args_cli.gripper_opening_mm):
        raise ValueError("Require 0 <= --gripper_closed_mm < --gripper_opening_mm")
    if len(_v62_args.piper_tip_offset) != 3:
        raise ValueError("--piper_tip_offset requires three values")
    if UIPC_SUBSTEPS < 8:
        raise ValueError(
            "V6.2 collision-safe runtime requires --uipc_substeps_per_record >= 8"
        )
    if float(args_cli.sim_hz) <= 0.0:
        raise ValueError("--sim_hz must be > 0")


def _smoothstep01(value: float) -> float:
    alpha = min(max(float(value), 0.0), 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _limit_vector_norm(vector: np.ndarray, limit: float) -> tuple[np.ndarray, float]:
    """Limit a vector norm and return the multiplicative scale that was applied."""
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    maximum = float(limit)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("Vector norm limit must be finite and positive")
    if not math.isfinite(norm):
        raise ValueError("Cannot limit a vector containing NaN or Inf")
    scale = min(1.0, maximum / max(norm, 1.0e-12))
    return value * scale, scale


def _project_force_to_contact_cone(
    force_w: np.ndarray,
    pad_quat_w: np.ndarray,
    friction_mu: float,
) -> tuple[np.ndarray, float]:
    """Project link8's feedback force onto its unilateral Coulomb cone.

    The grasp cylinder is always on the positive Pad-X side of the membrane.
    A negative Pad-X component would therefore attract the cylinder into the
    membrane, which a unilateral contact cannot do.  UIPC can emit such a
    transient while an explicit moving boundary crosses the barrier layer;
    feeding it back one substep later turns that transient into penetration.

    The returned scale is also applied to the reaction torque.  It is one for
    an already admissible force and zero when the contact reaction is wholly
    rejected.
    """
    value = np.asarray(force_w, dtype=np.float64).reshape(3)
    mu = float(friction_mu)
    if not np.all(np.isfinite(value)):
        raise ValueError("Cannot project a force containing NaN or Inf")
    if not math.isfinite(mu) or mu < 0.0:
        raise ValueError("Contact friction must be finite and non-negative")
    pad_normal_w = _quat_matrix(pad_quat_w)[:, 0]
    force_norm = float(np.linalg.norm(value))
    if force_norm <= 1.0e-12:
        return value.copy(), 1.0
    normal_force = float(np.dot(value, pad_normal_w))
    if normal_force <= 0.0:
        return np.zeros(3, dtype=np.float64), 0.0
    normal_part = normal_force * pad_normal_w
    tangent_part = value - normal_part
    tangent_norm = float(np.linalg.norm(tangent_part))
    tangent_limit = mu * normal_force
    tangent_scale = min(1.0, tangent_limit / max(tangent_norm, 1.0e-12))
    projected = normal_part + tangent_scale * tangent_part
    wrench_scale = min(
        1.0,
        float(np.linalg.norm(projected)) / force_norm,
    )
    return projected, wrench_scale


def _symmetric_nearest_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Return the symmetric maximum nearest-neighbor distance between two point sets."""
    a = np.asarray(points_a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(points_b, dtype=np.float64).reshape(-1, 3)
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("Point sets must be non-empty")
    distance_squared = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    return float(
        math.sqrt(
            max(
                float(np.max(np.min(distance_squared, axis=1))),
                float(np.max(np.min(distance_squared, axis=0))),
            )
        )
    )


def _quat_matrix(quat_wxyz: np.ndarray | tuple[float, float, float, float]) -> np.ndarray:
    return mainline_v9._quat_to_matrix(tuple(float(value) for value in quat_wxyz))


def _pose_matrix(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
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


def _position_from_delta_transform(
    delta_transform: np.ndarray,
    initial_position: np.ndarray,
) -> np.ndarray:
    """Reconstruct a body's current position from its rest-to-current transform."""
    transform = np.asarray(delta_transform, dtype=np.float64)
    position = np.asarray(initial_position, dtype=np.float64).reshape(3)
    if transform.shape != (4, 4):
        raise ValueError(f"Delta transform must have shape (4, 4), got {transform.shape}")
    if not np.all(np.isfinite(transform)) or not np.all(np.isfinite(position)):
        raise ValueError("Delta transform and initial position must be finite")
    position_h = transform @ np.asarray([*position, 1.0], dtype=np.float64)
    if not np.isfinite(position_h).all() or abs(float(position_h[3])) <= 1.0e-12:
        raise RuntimeError("UIPC boundary transform produced an invalid homogeneous position")
    return position_h[:3] / position_h[3]


def _points_from_pad_local(
    points_pad_l: np.ndarray,
    pad_position_w: np.ndarray,
    pad_quat_w: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_pad_l, dtype=np.float64).reshape(-1, 3)
    return points @ _quat_matrix(pad_quat_w).T + np.asarray(
        pad_position_w, dtype=np.float64
    ).reshape(1, 3)


def _vectors_from_pad_local(
    vectors_pad_l: np.ndarray,
    pad_quat_w: np.ndarray,
) -> np.ndarray:
    """Rotate free vectors from Pad local to world without adding translation."""
    vectors = np.asarray(vectors_pad_l, dtype=np.float64).reshape(-1, 3)
    return vectors @ _quat_matrix(pad_quat_w).T


def _resultant_wrench_from_contact_gradient(
    global_vertex_indices: np.ndarray,
    contact_gradient_pad_l: np.ndarray,
    *,
    boundary_global_vertex_offset: int,
    boundary_vertex_positions_pad_l: np.ndarray,
    object_center_pad_l: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reduce native Pad-local contact gradients to a Pad-local object wrench."""
    indices = np.asarray(global_vertex_indices, dtype=np.int64).reshape(-1)
    gradient = np.asarray(contact_gradient_pad_l, dtype=np.float64)
    vertices = np.asarray(boundary_vertex_positions_pad_l, dtype=np.float64).reshape(
        -1, 3
    )
    center = np.asarray(object_center_pad_l, dtype=np.float64).reshape(3)
    if gradient.size == 0 or indices.size == 0:
        return np.zeros(3), np.zeros(3), 0
    gradient = gradient.reshape(gradient.shape[0], -1)
    if gradient.shape[1] < 3 or gradient.shape[0] != indices.size:
        raise ValueError("UIPC contact gradient must match indices and have three components")
    if not np.all(np.isfinite(gradient[:, :3])) or not np.all(np.isfinite(vertices)):
        raise ValueError("UIPC reaction input contains NaN or Inf")
    local_indices = indices - int(boundary_global_vertex_offset)
    selected = (local_indices >= 0) & (local_indices < vertices.shape[0])
    if not np.any(selected):
        return np.zeros(3), np.zeros(3), 0
    selected_local = local_indices[selected]
    # UIPC exports dE_contact/dx; physical force is -dE_contact/dx.
    vertex_forces_pad_l = -gradient[selected, :3]
    force_pad_l = np.sum(vertex_forces_pad_l, axis=0, dtype=np.float64)
    moment_arms_pad_l = vertices[selected_local] - center.reshape(1, 3)
    torque_pad_l = np.sum(
        np.cross(moment_arms_pad_l, vertex_forces_pad_l), axis=0, dtype=np.float64
    )
    return force_pad_l, torque_pad_l, int(np.count_nonzero(selected))


def _object_pose_in_pad_frame(
    object_position_w: np.ndarray,
    object_quat_w: np.ndarray,
    pad_position_w: np.ndarray,
    pad_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``T_pad_object`` without making UIPC track the Pad in world space."""
    object_position_pad_l = mainline_v9._local_from_world(
        np.asarray(object_position_w, dtype=np.float64).reshape(1, 3),
        np.asarray(pad_position_w, dtype=np.float64),
        tuple(float(value) for value in pad_quat_w),
    )[0]
    object_quat_pad_l = mainline_v9._quat_multiply(
        mainline_v9._quat_conjugate(tuple(float(value) for value in pad_quat_w)),
        tuple(float(value) for value in object_quat_w),
    )
    return (
        np.asarray(object_position_pad_l, dtype=np.float64),
        np.asarray(object_quat_pad_l, dtype=np.float64),
    )


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


def _map_contract_vertices_to_runtime(
    contract_rest_pad_l: np.ndarray,
    runtime_rest_pad_l: np.ndarray,
    *,
    maximum_distance_m: float = 0.05e-3,
) -> tuple[np.ndarray, float]:
    """Map frozen 7f arrays onto the current compact-surface vertex ordering.

    UIPC compact-surface extraction may permute otherwise identical structured
    vertices when another contact object is added to the world.  The 7f force
    contract is geometric, so an exact one-to-one rest-position mapping is the
    correct bridge; assuming equal array indices is not.
    """
    contract = np.asarray(contract_rest_pad_l, dtype=np.float64)
    runtime = np.asarray(runtime_rest_pad_l, dtype=np.float64)
    if contract.ndim != 2 or contract.shape[1] != 3 or runtime.shape != contract.shape:
        raise ValueError(
            f"Contract/runtime rest surfaces must share [N,3], got {contract.shape} and {runtime.shape}"
        )
    if not np.all(np.isfinite(contract)) or not np.all(np.isfinite(runtime)):
        raise ValueError("Contract/runtime rest surfaces contain NaN or Inf")
    runtime_to_contract = np.empty(runtime.shape[0], dtype=np.int64)
    nearest_distance = np.empty(runtime.shape[0], dtype=np.float64)
    for begin in range(0, runtime.shape[0], 128):
        end = min(begin + 128, runtime.shape[0])
        delta = runtime[begin:end, None, :] - contract[None, :, :]
        distance_squared = np.sum(delta * delta, axis=2, dtype=np.float64)
        nearest = np.argmin(distance_squared, axis=1)
        runtime_to_contract[begin:end] = nearest
        nearest_distance[begin:end] = np.sqrt(
            distance_squared[np.arange(end - begin), nearest]
        )
    if np.unique(runtime_to_contract).size != runtime.shape[0]:
        raise RuntimeError("7f-to-runtime vertex mapping is not one-to-one")
    maximum_distance = float(np.max(nearest_distance, initial=0.0))
    if maximum_distance > float(maximum_distance_m):
        raise RuntimeError(
            "7f/runtime rest surfaces are geometrically incompatible: "
            f"maximum nearest distance {maximum_distance * 1000.0:.6f} mm"
        )
    return runtime_to_contract, maximum_distance


def _z_cylinder_surface(
    radius_m: float, height_m: float, segments: int = 32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segment_count = max(12, int(segments))
    half_height = 0.5 * float(height_m)
    points: list[tuple[float, float, float]] = []
    for z_value in (-half_height, half_height):
        for index in range(segment_count):
            theta = 2.0 * math.pi * float(index) / float(segment_count)
            points.append(
                (
                    float(radius_m) * math.cos(theta),
                    float(radius_m) * math.sin(theta),
                    z_value,
                )
            )
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


def _make_finite_drive_piper():
    robot_cfg = mainline_v9.AGILEX_PIPER_HIGH_PD_CFG.replace(
        prim_path=mainline_v9.ROBOT_ROOT
    )
    if str(args_cli.robot_usd_path).strip():
        robot_cfg.spawn.usd_path = str(
            Path(args_cli.robot_usd_path).expanduser().resolve()
        )
    else:
        robot_cfg.spawn.usd_path = str(DEFAULT_OWT_ROBOT_USD.resolve())
    # Required by the filtered contact diagnostic on the fixed link7 OpenWorldTactile
    # body.  This does not change collision response; it only exposes the
    # PhysX impulse/force that was already being solved.
    robot_cfg.spawn.activate_contact_sensors = True
    gripper = robot_cfg.actuators["piper_gripper"]
    gripper.stiffness = float(_v62_args.gripper_drive_stiffness)
    gripper.damping = float(_v62_args.gripper_drive_damping)
    gripper.effort_limit_sim = float(_v62_args.gripper_effort_limit_n)
    gripper.velocity_limit_sim = float(_v62_args.gripper_closing_velocity_m_s)
    return mainline_v9.Articulation(robot_cfg)


def _enable_opposing_link7_pad_collision(stage) -> str:
    """Turn the authored link7 OpenWorldTactile pad visual into the rigid opposing contact pad."""
    prim = stage.GetPrimAtPath(OPPOSING_PAD_PATH)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Cube):
        raise RuntimeError(
            "V6.2 requires the link7 OpenWorldTactile opposing pad from piper_openworldtactile.usda: "
            f"{OPPOSING_PAD_PATH}"
        )
    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr(True)
    prim.CreateAttribute("openworldtactile:contact_role", mainline_v9.Sdf.ValueTypeNames.String).Set(
        "rigid_opposing_pad_for_link8_uipc_membrane"
    )
    return str(prim.GetPath())


def _enable_link8_membrane_backing_collision(stage, pad_root: str) -> str:
    """Enable the authored rigid base behind the 0.5 mm UIPC membrane."""
    backing_path = f"{pad_root}/rigid_base/mesh"
    prim = stage.GetPrimAtPath(backing_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(
            "V6.2 requires the authored UIPC Pad rigid backing mesh: "
            f"{backing_path}"
        )
    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr(True)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision_api.GetApproximationAttr().Set("convexHull")
    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx_collision_api.CreateContactOffsetAttr().Set(0.00025)
    physx_collision_api.CreateRestOffsetAttr().Set(0.0)
    prim.CreateAttribute("openworldtactile:contact_role", mainline_v9.Sdf.ValueTypeNames.String).Set(
        "rigid_backing_behind_link8_uipc_membrane"
    )
    return backing_path


def _resolve_ee_body(robot) -> tuple[int, str]:
    indices, names = robot.find_bodies(str(_v62_args.piper_gripper_body))
    if len(indices) != 1:
        raise RuntimeError(
            f"Expected one body matching {_v62_args.piper_gripper_body!r}, got {list(names)}"
        )
    return int(indices[0]), str(names[0])


def _compute_frame_pose(robot, body_idx: int, offset_pos: torch.Tensor, offset_rot: torch.Tensor):
    body_pos_w = robot.data.body_link_pos_w[:, body_idx]
    body_quat_w = robot.data.body_link_quat_w[:, body_idx]
    position_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        body_pos_w,
        body_quat_w,
    )
    return math_utils.combine_frame_transforms(position_b, quat_b, offset_pos, offset_rot)


def _compute_frame_jacobian(
    robot, jacobi_body_idx: int, offset_pos: torch.Tensor, offset_rot: torch.Tensor
):
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, :].clone()
    base_rotation = math_utils.matrix_from_quat(math_utils.quat_inv(robot.data.root_link_quat_w))
    jacobian[:, :3, :] = torch.bmm(base_rotation, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rotation, jacobian[:, 3:, :])
    jacobian[:, :3, :] += torch.bmm(
        -math_utils.skew_symmetric_matrix(offset_pos), jacobian[:, 3:, :]
    )
    jacobian[:, 3:, :] = torch.bmm(
        math_utils.matrix_from_quat(offset_rot), jacobian[:, 3:, :]
    )
    return jacobian


def _world_pose_to_base(
    robot, target_position_w: np.ndarray, target_quat_w: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    device = robot.data.root_link_pos_w.device
    position_w = torch.as_tensor(
        target_position_w, device=device, dtype=torch.float32
    ).reshape(1, 3)
    quat_w = torch.as_tensor(
        target_quat_w, device=device, dtype=torch.float32
    ).reshape(1, 4)
    return math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        position_w,
        quat_w,
    )


def _quat_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    rotation = torch.as_tensor(
        np.asarray(matrix, dtype=np.float64).reshape(1, 3, 3), dtype=torch.float64
    )
    return (
        math_utils.quat_from_matrix(rotation)[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def _vertical_cylinder_aligned_ee_quat(
    current_pad_quat_w: np.ndarray, current_ee_quat_w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align Pad local Y with world Z and make its contact normal horizontal."""
    current_pad_rotation = _quat_matrix(current_pad_quat_w)
    current_ee_rotation = _quat_matrix(current_ee_quat_w)
    world_z = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)

    desired_pad_x = current_pad_rotation[:, 0] - np.dot(
        current_pad_rotation[:, 0], world_z
    ) * world_z
    desired_pad_x /= np.linalg.norm(desired_pad_x)
    tangent_sign = 1.0 if np.dot(current_pad_rotation[:, 1], world_z) >= 0.0 else -1.0
    desired_pad_y = tangent_sign * world_z
    desired_pad_z = np.cross(desired_pad_x, desired_pad_y)
    desired_pad_z /= np.linalg.norm(desired_pad_z)
    desired_pad_y = np.cross(desired_pad_z, desired_pad_x)
    desired_pad_rotation = np.column_stack(
        (desired_pad_x, desired_pad_y, desired_pad_z)
    )

    ee_to_pad_rotation = current_ee_rotation.T @ current_pad_rotation
    desired_ee_rotation = desired_pad_rotation @ ee_to_pad_rotation.T
    return (
        _quat_from_rotation_matrix(desired_ee_rotation),
        _quat_from_rotation_matrix(desired_pad_rotation),
    )


def _apply_ik_action(
    robot,
    controller,
    target_position_w: np.ndarray,
    target_quat_w: np.ndarray,
    opening_mm: float,
    body_idx: int,
    jacobi_body_idx: int,
    finger_joint_ids: list[int],
    finger_joint_signs: torch.Tensor,
    offset_pos: torch.Tensor,
    offset_rot: torch.Tensor,
) -> None:
    position_b, quat_b = _compute_frame_pose(robot, body_idx, offset_pos, offset_rot)
    target_position_b, target_quat_b = _world_pose_to_base(
        robot, target_position_w, target_quat_w
    )
    controller.set_command(
        torch.cat((target_position_b, target_quat_b), dim=1), position_b, quat_b
    )
    jacobian = _compute_frame_jacobian(
        robot, jacobi_body_idx, offset_pos, offset_rot
    )
    desired = controller.compute(
        position_b, quat_b, jacobian, robot.data.joint_pos
    ).clone()
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


def _object_state(cylinder: RigidObject) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = cylinder.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)
    quat = cylinder.data.root_link_quat_w[0].detach().cpu().numpy().astype(np.float64)
    velocity = cylinder.data.root_link_vel_w[0].detach().cpu().numpy().astype(np.float64)
    return position, quat, velocity


def _append_motion_segment(
    program: list[tuple[str, np.ndarray, float]],
    stage: str,
    start_position: np.ndarray,
    target_position: np.ndarray,
    start_opening_mm: float,
    target_opening_mm: float,
    frame_count: int,
) -> None:
    for frame in range(int(frame_count)):
        alpha = _smoothstep01(float(frame + 1) / float(frame_count))
        position = np.asarray(start_position, dtype=np.float64) + alpha * (
            np.asarray(target_position, dtype=np.float64)
            - np.asarray(start_position, dtype=np.float64)
        )
        opening = float(start_opening_mm) + alpha * (
            float(target_opening_mm) - float(start_opening_mm)
        )
        program.append((str(stage), position, opening))


def _build_motion_program(
    home: np.ndarray,
    above: np.ndarray,
    grasp: np.ndarray,
    lift: np.ndarray,
    *,
    opened_mm: float,
    closed_mm: float,
    close_frames: int,
    lift_frames: int,
) -> list[tuple[str, np.ndarray, float]]:
    counts = dict(MOTION_STAGE_SPECS)
    program: list[tuple[str, np.ndarray, float]] = []
    _append_motion_segment(program, "settle", home, home, opened_mm, opened_mm, counts["settle"])
    _append_motion_segment(program, "approach", home, above, opened_mm, opened_mm, counts["approach"])
    _append_motion_segment(program, "lower", above, grasp, opened_mm, opened_mm, counts["lower"])
    _append_motion_segment(
        program,
        "close",
        grasp,
        grasp,
        opened_mm,
        closed_mm,
        int(close_frames),
    )
    _append_motion_segment(program, "hold", grasp, grasp, closed_mm, closed_mm, counts["hold"])
    _append_motion_segment(
        program,
        "lift",
        grasp,
        lift,
        closed_mm,
        closed_mm,
        int(lift_frames),
    )
    _append_motion_segment(
        program, "hold_lifted", lift, lift, closed_mm, closed_mm, counts["hold_lifted"]
    )
    _append_motion_segment(program, "release", lift, lift, closed_mm, opened_mm, counts["release"])
    _append_motion_segment(program, "retreat", lift, home, opened_mm, opened_mm, counts["retreat"])
    _append_motion_segment(program, "recovery", home, home, opened_mm, opened_mm, counts["recovery"])
    return program


def _atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def _json_write(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


class _UipcStepWatchdog:
    """Terminate a native solve that would otherwise look like an endless GUI freeze."""

    def __init__(
        self,
        output_dir: Path,
        *,
        stage: str,
        frame: int,
        substep: int,
    ) -> None:
        self.output_dir = output_dir
        self.stage = str(stage)
        self.frame = int(frame)
        self.substep = int(substep)
        self.timer: threading.Timer | None = None

    def __enter__(self) -> "_UipcStepWatchdog":
        self.timer = threading.Timer(
            float(_v62_args.uipc_substep_timeout_sec), self._terminate
        )
        self.timer.daemon = True
        self.timer.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.timer is not None:
            self.timer.cancel()

    def _terminate(self) -> None:
        reason = (
            f"UIPC native step exceeded {_v62_args.uipc_substep_timeout_sec:.3f}s "
            f"at stage={self.stage}, frame={self.frame}, substep={self.substep}"
        )
        try:
            print(f"[V62_UIPC_TIMEOUT] {reason}", flush=True)
            _json_write(
                self.output_dir / "uipc_timeout.json",
                {
                    "error": reason,
                    "stage": self.stage,
                    "frame": self.frame,
                    "substep": self.substep,
                },
            )
        finally:
            os.kill(os.getpid(), signal.SIGTERM)


def _run_native_uipc_step(
    manual_step,
    output_dir: Path,
    *,
    stage: str,
    frame: int,
    substep: int,
) -> float:
    trace = bool(_v62_args.trace_uipc_steps)
    if trace:
        print(
            f"[V62_UIPC_BEGIN] stage={stage} frame={frame} substep={substep}",
            flush=True,
        )
    start = time.perf_counter()
    with _UipcStepWatchdog(
        output_dir, stage=stage, frame=frame, substep=substep
    ):
        manual_step()
    elapsed = time.perf_counter() - start
    if trace or elapsed >= 1.0:
        print(
            f"[V62_UIPC_END] stage={stage} frame={frame} substep={substep} "
            f"elapsed={elapsed:.6f}s",
            flush=True,
        )
    return elapsed


def _rebase_uipc_mesh_to_pad_local(
    uipc_object,
    vertex_positions_pad_l: np.ndarray,
) -> None:
    """Make Pad-local coordinates the UIPC rest frame before scene setup."""
    expected = np.asarray(vertex_positions_pad_l, dtype=np.float64).reshape(-1, 3)
    position_view = uipc_object.uipc_meshes[0].positions().view()
    positions = np.asarray(position_view).reshape(-1, 3)
    if positions.shape != expected.shape:
        raise ValueError(
            f"UIPC mesh/rebase shape mismatch: {positions.shape} != {expected.shape}"
        )
    positions[...] = expected.astype(positions.dtype, copy=False)
    actual = np.asarray(uipc_object.uipc_meshes[0].positions().view()).reshape(-1, 3)
    if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-8):
        raise RuntimeError("Failed to rebase UIPC membrane mesh into Pad-local coordinates")


class _PadLocalBackFaceAttachment:
    """Constrain membrane back vertices to fixed Pad-local rest positions."""

    def __init__(
        self,
        membrane,
        constraint_indices: np.ndarray,
        rest_positions_pad_l: np.ndarray,
        *,
        strength_ratio: float,
    ) -> None:
        from uipc.constitution import SoftPositionConstraint

        self._membrane = membrane
        self._indices = np.asarray(constraint_indices, dtype=np.int64).reshape(-1)
        self._targets_pad_l = np.asarray(
            rest_positions_pad_l, dtype=np.float32
        ).reshape(-1, 3)
        if self._indices.size == 0 or self._targets_pad_l.shape != (
            self._indices.size,
            3,
        ):
            raise ValueError("UIPC back-face attachment must be a non-empty [N,3] array")
        SoftPositionConstraint().apply_to(
            self._membrane.uipc_meshes[0], float(strength_ratio)
        )
        self._installed = False

    def install(self) -> None:
        """Install immutable Pad-local aim positions after scene-object creation."""
        if self._installed:
            return
        from uipc import builtin, view

        if len(self._membrane.uipc_scene_objects) != 1:
            raise RuntimeError("Expected exactly one UIPC membrane scene object")

        def animate_tet(info) -> None:
            geometry = info.geo_slots()[0].geometry()
            constrained = geometry.vertices().find(builtin.is_constrained)
            aim_position = geometry.vertices().find(builtin.aim_position)
            view(constrained)[self._indices] = 1
            view(aim_position)[self._indices] = self._targets_pad_l.reshape(-1, 3, 1)

        self._membrane.uipc_sim.scene.animator().insert(
            self._membrane.uipc_scene_objects[0], animate_tet
        )
        self._installed = True


class _ExternalRigidCollisionBoundary:
    """Kinematic collider driven by the authoritative ``T_pad_object`` pose."""

    def __init__(
        self,
        boundary,
        initial_position_pad_l: np.ndarray,
        initial_quat_pad_l: np.ndarray,
    ) -> None:
        self._boundary = boundary
        self._initial_position_pad_l = np.asarray(
            initial_position_pad_l, dtype=np.float64
        ).reshape(3)
        self._initial_quat_pad_l = np.asarray(
            initial_quat_pad_l, dtype=np.float64
        ).reshape(4)
        self._previous_transform = np.eye(4, dtype=np.float64)

    def synchronize(
        self,
        object_position_pad_l: np.ndarray,
        object_quat_pad_l: np.ndarray,
        dt: float,
    ) -> None:
        current_transform = _pose_delta_matrix(
            object_position_pad_l,
            object_quat_pad_l,
            self._initial_position_pad_l,
            self._initial_quat_pad_l,
        )
        self._boundary.write_external_rigid_boundary_pose_to_sim(
            self._previous_transform,
            current_transform,
            float(dt),
        )
        self._previous_transform = current_transform

    def actual_position_pad_l(self) -> np.ndarray:
        """Read the solver-retrieved ABD pose instead of echoing the write target."""
        geometry = self._boundary.geo_slot_list[0].geometry()
        transforms = np.asarray(geometry.transforms().view(), dtype=np.float64)
        if transforms.size != 16:
            raise RuntimeError(
                "Expected one UIPC transform for the external cylinder, "
                f"got array shape {transforms.shape}"
            )
        return _position_from_delta_transform(
            transforms.reshape(4, 4),
            self._initial_position_pad_l,
        )


def _uipc_global_vertex_offset(uipc_object) -> int:
    from uipc import builtin, view

    geometry = uipc_object.geo_slot_list[0].geometry()
    attribute = geometry.meta().find(builtin.global_vertex_offset)
    if attribute is None:
        raise RuntimeError("UIPC collision boundary has no global vertex offset")
    return int(np.asarray(view(attribute)).reshape(-1)[0])


def _uipc_object_surface(uipc_object) -> np.ndarray:
    """Return this object's solved compact surface from UIPC SceneIO."""
    uipc_sim = uipc_object.uipc_sim
    all_surface_points = np.asarray(
        uipc_sim.sio.simplicial_surface(2).positions().view(), dtype=np.float32
    ).reshape(-1, 3)
    object_index = int(uipc_object.obj_id) - 1
    offsets = uipc_sim._surf_vertex_offsets
    if object_index < 0 or object_index + 1 >= len(offsets):
        raise RuntimeError(
            f"UIPC object id {uipc_object.obj_id} has no SceneIO surface offset"
        )
    start = int(offsets[object_index])
    end = int(offsets[object_index + 1])
    if not 0 <= start <= end <= all_surface_points.shape[0]:
        raise RuntimeError(
            "Invalid UIPC SceneIO surface range: "
            f"object_id={uipc_object.obj_id} range=({start}, {end}) "
            f"global_count={all_surface_points.shape[0]}"
        )
    return all_surface_points[start:end].copy()


def _require_uipc_contact_feature(uipc_sim):
    from uipc.core import ContactSystemFeature

    feature = uipc_sim.world.features().find(ContactSystemFeature)
    if feature is None:
        raise RuntimeError("UIPC ContactSystemFeature is required for PhysX reaction coupling")
    return feature


def _read_uipc_boundary_reaction(
    contact_feature,
    *,
    boundary_global_vertex_offset: int,
    boundary_vertex_positions_pad_l: np.ndarray,
    object_center_pad_l: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    from uipc import view
    from uipc.geometry import Geometry

    gradient_geometry = Geometry()
    contact_feature.contact_gradient(gradient_geometry)
    instances = gradient_geometry.instances()
    index_attribute = instances.find("i")
    gradient_attribute = instances.find("grad")
    if index_attribute is None or gradient_attribute is None:
        raise RuntimeError("UIPC contact gradient is missing i/grad attributes")
    indices = np.asarray(view(index_attribute), dtype=np.int64).reshape(-1)
    gradients = np.asarray(view(gradient_attribute), dtype=np.float64)
    if gradients.size == 0:
        gradients = np.zeros((0, 3), dtype=np.float64)
    else:
        gradients = gradients.reshape(gradients.shape[0], -1)
    boundary_positions_pad_l = np.asarray(
        boundary_vertex_positions_pad_l, dtype=np.float64
    ).reshape(-1, 3)
    incremental_force, incremental_torque, vertex_count = (
        _resultant_wrench_from_contact_gradient(
        indices,
        gradients,
        boundary_global_vertex_offset=boundary_global_vertex_offset,
        boundary_vertex_positions_pad_l=boundary_positions_pad_l,
        object_center_pad_l=object_center_pad_l,
        )
    )
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("UIPC reaction conversion requires a finite dt > 0")
    # libuipc assembles IPC barrier energy with kappa * dt^2.  The exported
    # contact gradient is therefore an incremental-potential gradient, not a
    # force in newtons.  Divide by dt^2 before coupling the wrench to PhysX.
    inverse_dt_squared = 1.0 / (float(dt) * float(dt))
    return (
        incremental_force * inverse_dt_squared,
        incremental_torque * inverse_dt_squared,
        vertex_count,
    )


def _sync_pad_local_membrane_render(
    membrane,
    surface_pad_l: np.ndarray,
    pad_position_w: np.ndarray,
    pad_quat_w: np.ndarray,
) -> None:
    """Render the Pad-local solve at the live Pad world pose."""
    surface_w = _points_from_pad_local(
        surface_pad_l, pad_position_w, pad_quat_w
    ).astype(np.float32)
    mainline_v9._sync_render_surface(membrane, surface_w)


def _apply_physx_coupling_wrenches(
    cylinder: RigidObject,
    robot,
    mount_body_idx: int,
    force_w: np.ndarray,
    torque_w: np.ndarray,
) -> None:
    """Apply the UIPC object wrench and its equal/opposite link8 reaction."""
    device = cylinder.data.root_link_pos_w.device
    dtype = cylinder.data.root_link_pos_w.dtype
    force = torch.as_tensor(force_w, device=device, dtype=dtype).reshape(1, 1, 3)
    torque = torch.as_tensor(torque_w, device=device, dtype=dtype).reshape(1, 1, 3)
    cylinder.set_external_force_and_torque(
        forces=force,
        torques=torque,
        is_global=True,
    )

    # The UIPC torque is reduced about the cylinder center.  Transfer the
    # equal/opposite wrench to the link8 center of mass so Newton's third law
    # also reaches the gripper articulation and its finite-effort drive.
    object_center_w = cylinder.data.root_com_pos_w[0]
    link_center_w = robot.data.body_com_pos_w[0, int(mount_body_idx)]
    center_offset_w = object_center_w - link_center_w
    link_force = -force
    link_torque = -torque - torch.linalg.cross(
        center_offset_w.reshape(1, 1, 3), force, dim=2
    )
    robot.set_external_force_and_torque(
        forces=link_force,
        torques=link_torque,
        body_ids=[int(mount_body_idx)],
        is_global=True,
    )
    cylinder.write_data_to_sim()
    robot.write_data_to_sim()


def _save_dataset(
    output_dir: Path,
    records: dict[str, list[object]],
    *,
    rest_surface_pad_l: np.ndarray,
    contract_vertex_area: np.ndarray,
    contract_front_mask: np.ndarray,
    surface_triangles: np.ndarray,
    metadata: dict[str, object],
) -> None:
    triangles = np.asarray(surface_triangles, dtype=np.int64).reshape(-1, 3)
    front_triangles = triangles[
        np.all(np.asarray(contract_front_mask, dtype=bool)[triangles], axis=1)
    ]
    arrays = {
        "frame_id.npy": np.asarray(records["frame_id"], dtype=np.int64),
        "motion_stage.npy": np.asarray(records["motion_stage"], dtype=str),
        "surface_displacement_pad_local.npy": np.asarray(
            records["surface_displacement_pad_local"], dtype=np.float32
        ),
        "force_pad_local.npy": np.asarray(records["force_pad_local"], dtype=np.float64),
        "tactile_force_channels.npy": np.asarray(
            records["tactile_force_channels"], dtype=np.float64
        ),
        "contact_active.npy": np.asarray(records["contact_active"], dtype=bool),
        "minimum_signed_gap_mm.npy": np.asarray(
            records["minimum_signed_gap_mm"], dtype=np.float64
        ),
        "maximum_normal_deformation_mm.npy": np.asarray(
            records["maximum_normal_deformation_mm"], dtype=np.float64
        ),
        "object_pose_w.npy": np.asarray(records["object_pose_w"], dtype=np.float32),
        "object_pose_pad_local.npy": np.asarray(
            records["object_pose_pad_local"], dtype=np.float32
        ),
        "object_pose_opposing_pad_local.npy": np.asarray(
            records["object_pose_opposing_pad_local"], dtype=np.float32
        ),
        "object_velocity_w.npy": np.asarray(records["object_velocity_w"], dtype=np.float32),
        "pad_pose_w.npy": np.asarray(records["pad_pose_w"], dtype=np.float32),
        "opposing_pad_pose_w.npy": np.asarray(
            records["opposing_pad_pose_w"], dtype=np.float32
        ),
        "end_effector_pose_w.npy": np.asarray(
            records["end_effector_pose_w"], dtype=np.float32
        ),
        "gripper_opening_mm.npy": np.asarray(
            records["gripper_opening_mm"], dtype=np.float32
        ),
        "commanded_gripper_opening_mm.npy": np.asarray(
            records["commanded_gripper_opening_mm"], dtype=np.float32
        ),
        "uipc_reaction_force_w.npy": np.asarray(
            records["uipc_reaction_force_w"], dtype=np.float64
        ),
        "uipc_reaction_torque_w.npy": np.asarray(
            records["uipc_reaction_torque_w"], dtype=np.float64
        ),
        "applied_uipc_force_w.npy": np.asarray(
            records["applied_uipc_force_w"], dtype=np.float64
        ),
        "applied_uipc_torque_w.npy": np.asarray(
            records["applied_uipc_torque_w"], dtype=np.float64
        ),
        "uipc_reaction_force_substeps_w.npy": np.asarray(
            records["uipc_reaction_force_substeps_w"], dtype=np.float64
        ),
        "uipc_reaction_torque_substeps_w.npy": np.asarray(
            records["uipc_reaction_torque_substeps_w"], dtype=np.float64
        ),
        "uipc_admissible_force_substeps_w.npy": np.asarray(
            records["uipc_admissible_force_substeps_w"], dtype=np.float64
        ),
        "uipc_admissible_torque_substeps_w.npy": np.asarray(
            records["uipc_admissible_torque_substeps_w"], dtype=np.float64
        ),
        "applied_uipc_force_substeps_w.npy": np.asarray(
            records["applied_uipc_force_substeps_w"], dtype=np.float64
        ),
        "applied_uipc_torque_substeps_w.npy": np.asarray(
            records["applied_uipc_torque_substeps_w"], dtype=np.float64
        ),
        "opposing_contact_force_w.npy": np.asarray(
            records["opposing_contact_force_w"], dtype=np.float64
        ),
        "opposing_contact_force_substeps_w.npy": np.asarray(
            records["opposing_contact_force_substeps_w"], dtype=np.float64
        ),
        "backing_contact_force_w.npy": np.asarray(
            records["backing_contact_force_w"], dtype=np.float64
        ),
        "backing_contact_force_substeps_w.npy": np.asarray(
            records["backing_contact_force_substeps_w"], dtype=np.float64
        ),
        "uipc_feedback_force_scale_substeps.npy": np.asarray(
            records["uipc_feedback_force_scale_substeps"], dtype=np.float64
        ),
        "uipc_feedback_torque_scale_substeps.npy": np.asarray(
            records["uipc_feedback_torque_scale_substeps"], dtype=np.float64
        ),
        "uipc_contact_cone_scale_substeps.npy": np.asarray(
            records["uipc_contact_cone_scale_substeps"], dtype=np.float64
        ),
        "uipc_boundary_surface_sync_error_mm.npy": np.asarray(
            records["uipc_boundary_surface_sync_error_mm"], dtype=np.float64
        ),
        "uipc_reaction_vertex_count.npy": np.asarray(
            records["uipc_reaction_vertex_count"], dtype=np.int64
        ),
        "uipc_step_time_sec.npy": np.asarray(
            records["uipc_step_time_sec"], dtype=np.float64
        ),
        "frame_wall_time_sec.npy": np.asarray(
            records["frame_wall_time_sec"], dtype=np.float64
        ),
        "uipc_substep_time_sec.npy": np.asarray(
            records["uipc_substep_time_sec"], dtype=np.float64
        ),
        "rest_surface_vertices_pad_local.npy": np.asarray(
            rest_surface_pad_l, dtype=np.float32
        ),
        "vertex_area.npy": np.asarray(contract_vertex_area, dtype=np.float64),
        "front_surface_mask.npy": np.asarray(contract_front_mask, dtype=bool),
        "surface_triangles.npy": triangles,
        "front_surface_triangles.npy": front_triangles,
    }
    for filename, value in arrays.items():
        _atomic_save(output_dir / filename, value)
    _json_write(output_dir / "metadata.json", metadata)


RECORD_KEYS = (
    "frame_id",
    "motion_stage",
    "surface_displacement_pad_local",
    "force_pad_local",
    "tactile_force_channels",
    "contact_active",
    "minimum_signed_gap_mm",
    "maximum_normal_deformation_mm",
    "object_pose_w",
    "object_pose_pad_local",
    "object_pose_opposing_pad_local",
    "object_velocity_w",
    "pad_pose_w",
    "opposing_pad_pose_w",
    "end_effector_pose_w",
    "gripper_opening_mm",
    "commanded_gripper_opening_mm",
    "uipc_reaction_force_w",
    "uipc_reaction_torque_w",
    "applied_uipc_force_w",
    "applied_uipc_torque_w",
    "uipc_reaction_force_substeps_w",
    "uipc_reaction_torque_substeps_w",
    "uipc_admissible_force_substeps_w",
    "uipc_admissible_torque_substeps_w",
    "applied_uipc_force_substeps_w",
    "applied_uipc_torque_substeps_w",
    "opposing_contact_force_w",
    "opposing_contact_force_substeps_w",
    "backing_contact_force_w",
    "backing_contact_force_substeps_w",
    "uipc_feedback_force_scale_substeps",
    "uipc_feedback_torque_scale_substeps",
    "uipc_contact_cone_scale_substeps",
    "uipc_boundary_surface_sync_error_mm",
    "uipc_reaction_vertex_count",
    "uipc_step_time_sec",
    "frame_wall_time_sec",
    "uipc_substep_time_sec",
)


def _new_records() -> dict[str, list[object]]:
    """Keep only the current cycle in RAM during interactive looping."""
    return {key: [] for key in RECORD_KEYS}


def main() -> None:
    print("[V62_START] validating arguments", flush=True)
    _validate_args()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "error.json").unlink(missing_ok=True)
    (output_dir / "uipc_timeout.json").unlink(missing_ok=True)
    contract_dir = Path(args_cli.contract_dir).expanduser().resolve()
    workspace_dir = Path(args_cli.workspace_dir).expanduser().resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    required_contract_files = (
        "vertex_area.npy",
        "front_surface_mask.npy",
        "rest_surface_pad_local.npy",
        "verdict.json",
    )
    missing_contract_files = [
        name for name in required_contract_files if not (contract_dir / name).is_file()
    ]
    if missing_contract_files:
        raise FileNotFoundError(
            f"Frozen 7f contract is incomplete in {contract_dir}: {missing_contract_files}"
        )
    frozen_contract_vertex_area = np.asarray(
        np.load(contract_dir / "vertex_area.npy", allow_pickle=False), dtype=np.float64
    ).reshape(-1)
    frozen_contract_front_mask = np.asarray(
        np.load(contract_dir / "front_surface_mask.npy", allow_pickle=False), dtype=bool
    ).reshape(-1)
    frozen_contract_rest_pad_l = np.asarray(
        np.load(contract_dir / "rest_surface_pad_local.npy", allow_pickle=False),
        dtype=np.float64,
    )
    contract_verdict = json.loads((contract_dir / "verdict.json").read_text())
    if not bool(contract_verdict.get("deformation_contract_passed", False)):
        raise RuntimeError("Frozen 7f deformation contract did not pass")
    print(
        f"[V62_START] output={output_dir} contract_vertices={frozen_contract_vertex_area.size} "
        f"viewport={bool(args_cli.render_viewport)}",
        flush=True,
    )

    record_dt = 1.0 / float(args_cli.sim_hz)
    coupling_substep_dt = record_dt / float(UIPC_SUBSTEPS)
    object_radius_m = float(_v62_args.object_radius_mm) * 1.0e-3
    object_height_m = float(_v62_args.object_height_mm) * 1.0e-3
    feedback_force_limit_n = float(_v62_args.uipc_feedback_force_limit_n)
    feedback_torque_limit_nm = feedback_force_limit_n * object_radius_m

    print("[V62_INIT] creating PhysX scene", flush=True)
    sim = mainline_v9.sim_utils.SimulationContext(
        mainline_v9.SimulationCfg(
            dt=coupling_substep_dt,
            render_interval=1,
            physx=mainline_v9.PhysxCfg(enable_ccd=True),
            physics_material=mainline_v9.sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=float(_v62_args.object_friction),
                dynamic_friction=float(_v62_args.object_friction),
                restitution=0.0,
            ),
        )
    )
    if bool(args_cli.render_viewport):
        sim.set_camera_view([0.58, -0.48, 0.42], [0.28, -0.01, 0.13])
    stage = mainline_v9.omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage")
    for prim_path in ("/World", "/World/envs", "/World/envs/env_0"):
        UsdGeom.Xform.Define(stage, prim_path)

    ground_cfg = mainline_v9.sim_utils.GroundPlaneCfg(
        physics_material=mainline_v9.sim_utils.RigidBodyMaterialCfg(
            static_friction=float(_v62_args.object_friction),
            dynamic_friction=float(_v62_args.object_friction),
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg, translation=(0.0, 0.0, 0.0))
    light_cfg = mainline_v9.sim_utils.DomeLightCfg(
        intensity=2800.0, color=(0.78, 0.78, 0.78)
    )
    light_cfg.func("/World/Light", light_cfg)

    robot = _make_finite_drive_piper()
    opposing_pad_path = _enable_opposing_link7_pad_collision(stage)
    object_initial_position = np.asarray(
        (
            float(_v62_args.object_x),
            float(_v62_args.object_y),
            0.5 * object_height_m
            + float(_v62_args.object_support_clearance_mm) * 1.0e-3,
        ),
        dtype=np.float64,
    )
    cylinder = RigidObject(
        RigidObjectCfg(
            prim_path=OBJECT_PATH,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(float(value) for value in object_initial_position),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=mainline_v9.sim_utils.CylinderCfg(
                radius=object_radius_m,
                height=object_height_m,
                axis="Z",
                rigid_props=_rigid_props(),
                mass_props=mainline_v9.sim_utils.MassPropertiesCfg(
                    mass=float(_v62_args.object_mass_kg)
                ),
                collision_props=mainline_v9.sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.001,
                    rest_offset=0.0,
                ),
                physics_material=mainline_v9.sim_utils.RigidBodyMaterialCfg(
                    static_friction=float(_v62_args.object_friction),
                    dynamic_friction=float(_v62_args.object_friction),
                    restitution=0.0,
                ),
                visual_material=mainline_v9.sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.85, 0.34, 0.20), roughness=0.55
                ),
            ),
        )
    )
    opposing_contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=OPPOSING_CONTACT_BODY_PATH,
            update_period=0.0,
            history_length=0,
            debug_vis=False,
            track_pose=True,
            filter_prim_paths_expr=[OBJECT_PATH],
        )
    )
    opposing_pad_prim = stage.GetPrimAtPath(opposing_pad_path)
    opposing_pad_position_body_l = np.asarray(
        opposing_pad_prim.GetAttribute("xformOp:translate").Get(), dtype=np.float64
    )
    opposing_pad_scale_m = np.asarray(
        opposing_pad_prim.GetAttribute("xformOp:scale").Get(), dtype=np.float64
    )
    # The cube has translate+non-uniform-scale only.  Extracting a quaternion
    # from its scaled world matrix contaminates the rotation; relative to the
    # openworldtactile_case_left rigid body its orientation is exactly identity.
    opposing_pad_quat_body_l = (1.0, 0.0, 0.0, 0.0)

    mount_link_path = mainline_v9._normalize_mount_link_path(str(args_cli.mount_link_path))
    if mount_link_path != mainline_v9.DEFAULT_MOUNT_LINK_PATH:
        raise ValueError(
            f"V6.2 requires direct link8 mounting at {mainline_v9.DEFAULT_MOUNT_LINK_PATH}"
        )
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
    mainline_v9._apply_visual_policy(stage, pad_root)
    if not stage.GetPrimAtPath(membrane_mesh_path).IsValid():
        raise RuntimeError(f"Frozen membrane source is missing: {membrane_mesh_path}")
    backing_path = _enable_link8_membrane_backing_collision(stage, pad_root)
    backing_contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=mount_link_path,
            update_period=0.0,
            history_length=0,
            debug_vis=False,
            track_pose=False,
            filter_prim_paths_expr=[OBJECT_PATH],
        )
    )

    print("[V62_INIT] resetting PhysX scene", flush=True)
    sim.reset()
    print("[V62_INIT] PhysX scene ready", flush=True)
    robot.update(0.0)
    cylinder.update(0.0)
    opposing_contact_sensor.update(0.0, force_recompute=True)
    backing_contact_sensor.update(0.0, force_recompute=True)
    print(
        "[V62_OPPOSING_CONTACT_SETUP] "
        f"body={OPPOSING_CONTACT_BODY_PATH} filter={OBJECT_PATH} "
        f"pad_position_body_l_m={np.asarray(opposing_pad_position_body_l).tolist()} "
        f"pad_quat_body_l_wxyz={np.asarray(opposing_pad_quat_body_l).tolist()} "
        f"pad_size_m={opposing_pad_scale_m.tolist()}",
        flush=True,
    )
    print(
        "[V62_BACKING_CONTACT_SETUP] "
        f"body={mount_link_path} collider={backing_path} filter={OBJECT_PATH} "
        "membrane_front_to_backing_m=0.0005 approximation=convexHull "
        "contact_offset_m=0.00025 rest_offset_m=0.0",
        flush=True,
    )
    mount_body_idx, _ = mainline_v9._resolve_mount_body(robot, mount_link_path)
    ee_body_idx, _ = _resolve_ee_body(robot)
    jacobi_body_idx = ee_body_idx - 1
    finger_joint_ids, finger_joint_signs = mainline_v9._resolve_gripper(robot)
    offset_position = torch.tensor(
        _v62_args.piper_tip_offset, device=sim.device, dtype=torch.float32
    ).reshape(1, 3)
    offset_rotation = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=torch.float32
    )
    ik_controller = DifferentialIKController(
        cfg=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        num_envs=1,
        device=sim.device,
    )
    home_position = np.asarray(
        (_v62_args.home_ee_x, _v62_args.home_ee_y, _v62_args.home_ee_z),
        dtype=np.float64,
    )
    prealignment_link_position_w, prealignment_link_quat_w = mainline_v9._body_pose(
        robot, mount_body_idx
    )
    _, prealignment_pad_quat_w = mainline_v9._compose_child_pose(
        prealignment_link_position_w,
        prealignment_link_quat_w,
        pad_position_l,
        pad_quat_l,
    )
    _, prealignment_ee_quat_w = _ee_pose_w(robot, ee_body_idx, offset_position)
    desired_ee_quat_w, desired_pad_quat_w = _vertical_cylinder_aligned_ee_quat(
        prealignment_pad_quat_w, prealignment_ee_quat_w
    )
    print(
        "[V62_GRASP_ORIENTATION_TARGET] "
        f"desired_ee_quat_wxyz={desired_ee_quat_w.tolist()} "
        f"desired_pad_quat_wxyz={desired_pad_quat_w.tolist()} "
        "pad_y_parallel_to_world_z=true pad_x_perpendicular_to_world_z=true",
        flush=True,
    )

    for settle_frame in range(int(_v62_args.initial_settle_frames)):
        frame_wall_start = time.perf_counter()
        if settle_frame % max(1, int(args_cli.log_every)) == 0:
            print(
                f"[V62_PHYSX_BEGIN] frame={settle_frame + 1}/"
                f"{int(_v62_args.initial_settle_frames)}",
                flush=True,
            )
        for coupling_substep in range(UIPC_SUBSTEPS):
            _apply_ik_action(
                robot,
                ik_controller,
                home_position,
                desired_ee_quat_w,
                float(args_cli.gripper_opening_mm),
                ee_body_idx,
                jacobi_body_idx,
                finger_joint_ids,
                finger_joint_signs,
                offset_position,
                offset_rotation,
            )
            sim.step(
                render=bool(args_cli.render_viewport)
                and coupling_substep + 1 == UIPC_SUBSTEPS
            )
            robot.update(coupling_substep_dt)
            cylinder.update(coupling_substep_dt)
        frame_wall_elapsed = time.perf_counter() - frame_wall_start
        if frame_wall_elapsed > float(_v62_args.slow_frame_threshold_sec):
            print(
                f"[V62_SLOW_FRAME] phase=physx_warmup frame={settle_frame + 1} "
                f"elapsed={frame_wall_elapsed:.6f}s "
                f"threshold={float(_v62_args.slow_frame_threshold_sec):.6f}s "
                "action=continue_and_diagnose",
                flush=True,
            )
        if settle_frame % max(1, int(args_cli.log_every)) == 0:
            print(
                f"[V62_PHYSX_WARMUP] frame={settle_frame + 1} "
                f"frame_wall={frame_wall_elapsed:.6f}s",
                flush=True,
            )

    warmup_joint_target = robot.data.joint_pos.clone()
    link_position_w, link_quat_w = mainline_v9._body_pose(robot, mount_body_idx)
    initial_pad_position_w, initial_pad_quat_w = mainline_v9._compose_child_pose(
        link_position_w, link_quat_w, pad_position_l, pad_quat_l
    )
    aligned_pad_rotation_w = _quat_matrix(initial_pad_quat_w)
    pad_normal_vertical_component = float(abs(aligned_pad_rotation_w[2, 0]))
    pad_tangent_vertical_component = float(abs(aligned_pad_rotation_w[2, 1]))
    pad_normal_to_horizontal_error_deg = math.degrees(
        math.asin(min(1.0, pad_normal_vertical_component))
    )
    pad_tangent_to_cylinder_axis_error_deg = math.degrees(
        math.acos(min(1.0, pad_tangent_vertical_component))
    )
    print(
        "[V62_GRASP_ORIENTATION_ACTUAL] "
        f"pad_normal_to_horizontal_error_deg="
        f"{pad_normal_to_horizontal_error_deg:.6f} "
        f"pad_tangent_to_cylinder_axis_error_deg="
        f"{pad_tangent_to_cylinder_axis_error_deg:.6f}",
        flush=True,
    )
    object_boundary_initial_position_w, object_boundary_initial_quat_w, _ = _object_state(
        cylinder
    )
    (
        object_boundary_initial_position_pad_l,
        object_boundary_initial_quat_pad_l,
    ) = _object_pose_in_pad_frame(
        object_boundary_initial_position_w,
        object_boundary_initial_quat_w,
        initial_pad_position_w,
        initial_pad_quat_w,
    )

    membrane_source_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(membrane_mesh_path))
    source_points_l = np.asarray(
        membrane_source_mesh.GetPointsAttr().Get(), dtype=np.float64
    ).reshape(-1, 3)
    structured_points_l, structured_tetrahedra, structured_surface_triangles = (
        mainline_v9._structured_box_tet_mesh_l(
            np.min(source_points_l, axis=0),
            np.max(source_points_l, axis=0),
            STRUCTURED_CELLS,
        )
    )
    # The mounted asset mesh is the one and only solver membrane.  UipcObject
    # initially observes the live USD hierarchy, then its not-yet-installed
    # UIPC mesh is rebased once into Pad-local rest coordinates below.
    mainline_v9._write_triangle_mesh(
        stage,
        membrane_mesh_path,
        structured_points_l,
        structured_surface_triangles,
        color=(0.65, 0.05, 0.05),
    )
    mainline_v9._write_precomputed_tet_data(
        stage,
        membrane_mesh_path,
        structured_points_l,
        structured_tetrahedra,
        structured_surface_triangles,
    )

    print("[V62_INIT] creating link8 membrane and external object boundary", flush=True)
    uipc_sim = mainline_v9.UipcSim(
        mainline_v9.UipcSimCfg(
            dt=coupling_substep_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=str(workspace_dir),
            sanity_check_enable=bool(args_cli.uipc_sanity_check),
            newton=mainline_v9.UipcSimCfg.Newton(
                max_iter=int(args_cli.uipc_newton_max_iter)
            ),
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
    _rebase_uipc_mesh_to_pad_local(membrane, structured_points_l)
    back_tet_indices, _, _ = mainline_v9._face_indices(structured_points_l)
    back_attachment = _PadLocalBackFaceAttachment(
        membrane,
        back_tet_indices,
        structured_points_l[back_tet_indices],
        strength_ratio=float(args_cli.attachment_strength_ratio),
    )

    boundary_points_l, boundary_triangles, boundary_tetrahedra = _z_cylinder_surface(
        object_radius_m, object_height_m
    )
    boundary_points_world_reference = mainline_v9._world_from_local(
        boundary_points_l,
        object_boundary_initial_position_w,
        tuple(object_boundary_initial_quat_w),
    )
    boundary_points_pad_l = mainline_v9._world_from_local(
        boundary_points_l,
        object_boundary_initial_position_pad_l,
        tuple(object_boundary_initial_quat_pad_l),
    )
    membrane_points_world_reference = mainline_v9._world_from_local(
        structured_points_l,
        initial_pad_position_w,
        tuple(initial_pad_quat_w),
    )
    membrane_reconstruction_error_m = float(
        np.max(
            np.linalg.norm(
                _points_from_pad_local(
                    structured_points_l,
                    initial_pad_position_w,
                    initial_pad_quat_w,
                )
                - membrane_points_world_reference,
                axis=1,
            ),
            initial=0.0,
        )
    )
    boundary_reconstruction_error_m = float(
        np.max(
            np.linalg.norm(
                _points_from_pad_local(
                    boundary_points_pad_l,
                    initial_pad_position_w,
                    initial_pad_quat_w,
                )
                - boundary_points_world_reference,
                axis=1,
            ),
            initial=0.0,
        )
    )
    pad_local_reconstruction_error_m = max(
        membrane_reconstruction_error_m, boundary_reconstruction_error_m
    )
    if pad_local_reconstruction_error_m > 1.0e-8:
        raise RuntimeError(
            "Pad-local rebase changed the initial membrane/object world placement: "
            f"maximum_error={pad_local_reconstruction_error_m:.9e}m"
        )
    print(
        "[V62_PAD_LOCAL_INVARIANT] initial relative placement preserved; "
        f"maximum_world_reconstruction_error={pad_local_reconstruction_error_m:.9e}m",
        flush=True,
    )
    mainline_v9._write_triangle_mesh(
        stage,
        UIPC_BOUNDARY_MESH_PATH,
        boundary_points_pad_l,
        boundary_triangles,
        color=(0.24, 0.24, 0.24),
    )
    mainline_v9._write_precomputed_tet_data(
        stage,
        UIPC_BOUNDARY_MESH_PATH,
        boundary_points_pad_l,
        boundary_tetrahedra,
        boundary_triangles,
    )
    boundary_density = float(_v62_args.object_mass_kg) / max(
        math.pi * object_radius_m * object_radius_m * object_height_m, EPS
    )
    object_boundary = mainline_v9.UipcObject(
        mainline_v9.UipcObjectCfg(
            prim_path=UIPC_BOUNDARY_ROOT,
            mesh_cfg=None,
            mass_density=boundary_density,
            constitution_cfg=mainline_v9.UipcObjectCfg.AffineBodyConstitutionCfg(
                m_kappa=float(args_cli.tool_m_kappa_mpa), kinematic=True
            ),
        ),
        uipc_sim,
    )
    boundary_driver = _ExternalRigidCollisionBoundary(
        object_boundary,
        object_boundary_initial_position_pad_l,
        object_boundary_initial_quat_pad_l,
    )
    mainline_v9._ensure_asset_initialized(membrane)
    back_attachment.install()
    mainline_v9._ensure_asset_initialized(object_boundary)

    manual_uipc_step = uipc_sim.step
    uipc_sim.step = lambda dt=0.0: None
    print("[V62_INIT] UIPC setup begin", flush=True)
    uipc_sim.setup_sim()
    print("[V62_INIT] UIPC setup complete", flush=True)
    boundary_driver.synchronize(
        object_boundary_initial_position_pad_l,
        object_boundary_initial_quat_pad_l,
        coupling_substep_dt,
    )
    contact_feature = _require_uipc_contact_feature(uipc_sim)
    boundary_global_vertex_offset = _uipc_global_vertex_offset(object_boundary)
    aligned_surface_pad_l = _uipc_object_surface(membrane).astype(np.float32)
    runtime_surface_triangles = mainline_v9._uipc_surface_triangles(membrane)
    if frozen_contract_vertex_area.shape != (aligned_surface_pad_l.shape[0],):
        raise RuntimeError("7f vertex-area shape does not match the runtime membrane")
    runtime_to_contract, contract_mapping_error_m = _map_contract_vertices_to_runtime(
        frozen_contract_rest_pad_l,
        aligned_surface_pad_l,
    )
    contract_vertex_area = frozen_contract_vertex_area[runtime_to_contract]
    contract_front_mask = frozen_contract_front_mask[runtime_to_contract]
    _, aligned_front_indices, _ = mainline_v9._face_indices(aligned_surface_pad_l)
    aligned_front_mask = np.zeros(aligned_surface_pad_l.shape[0], dtype=bool)
    aligned_front_mask[aligned_front_indices] = True
    if not np.array_equal(contract_front_mask, aligned_front_mask):
        mismatch = int(np.count_nonzero(contract_front_mask != aligned_front_mask))
        raise RuntimeError(
            "Mapped 7f front-surface mask disagrees with aligned runtime geometry: "
            f"mismatch_count={mismatch}"
        )
    print(
        f"[V62_CONTRACT] mapped_vertices={runtime_to_contract.size} "
        f"permuted={int(np.count_nonzero(runtime_to_contract != np.arange(runtime_to_contract.size)))} "
        f"max_rest_error={contract_mapping_error_m * 1000.0:.6f}mm",
        flush=True,
    )
    print("[V62_INIT] first UIPC solve begin", flush=True)
    initial_uipc_elapsed = _run_native_uipc_step(
        manual_uipc_step,
        output_dir,
        stage="initial_alignment",
        frame=-1,
        substep=0,
    )
    (
        initial_uipc_force_pad_l,
        initial_uipc_torque_pad_l,
        _,
    ) = _read_uipc_boundary_reaction(
        contact_feature,
        boundary_global_vertex_offset=boundary_global_vertex_offset,
        boundary_vertex_positions_pad_l=boundary_points_pad_l,
        object_center_pad_l=object_boundary_initial_position_pad_l,
        dt=coupling_substep_dt,
    )
    previous_uipc_force_w = _vectors_from_pad_local(
        initial_uipc_force_pad_l, initial_pad_quat_w
    )[0]
    previous_uipc_torque_w = _vectors_from_pad_local(
        initial_uipc_torque_pad_l, initial_pad_quat_w
    )[0]
    previous_uipc_force_w, initial_contact_cone_scale = (
        _project_force_to_contact_cone(
            previous_uipc_force_w,
            initial_pad_quat_w,
            float(args_cli.uipc_friction_mu),
        )
    )
    previous_uipc_torque_w *= initial_contact_cone_scale
    previous_uipc_force_w, _ = _limit_vector_norm(
        previous_uipc_force_w, feedback_force_limit_n
    )
    previous_uipc_torque_w, _ = _limit_vector_norm(
        previous_uipc_torque_w, feedback_torque_limit_nm
    )
    print("[V62_INIT] first UIPC solve complete", flush=True)
    surface_pad_l = _uipc_object_surface(membrane).astype(np.float32)
    _sync_pad_local_membrane_render(
        membrane,
        surface_pad_l,
        initial_pad_position_w,
        initial_pad_quat_w,
    )
    UsdGeom.Imageable(stage.GetPrimAtPath(membrane_mesh_path)).MakeVisible()
    UsdGeom.Imageable(stage.GetPrimAtPath(UIPC_BOUNDARY_ROOT)).MakeInvisible()

    def advance_uipc(
        pad_position_w: np.ndarray,
        pad_quat_w: np.ndarray,
        object_position_w: np.ndarray,
        object_quat_w: np.ndarray,
        *,
        solve_stage: str,
        solve_frame: int,
        solve_substep: int,
    ) -> tuple[float, np.ndarray, np.ndarray, int]:
        object_position_pad_l, object_quat_pad_l = _object_pose_in_pad_frame(
            object_position_w,
            object_quat_w,
            pad_position_w,
            pad_quat_w,
        )
        boundary_driver.synchronize(
            object_position_pad_l,
            object_quat_pad_l,
            coupling_substep_dt,
        )
        elapsed = _run_native_uipc_step(
            manual_uipc_step,
            output_dir,
            stage=solve_stage,
            frame=solve_frame,
            substep=solve_substep,
        )
        (
            reaction_force_pad_l,
            reaction_torque_pad_l,
            reaction_vertex_count,
        ) = _read_uipc_boundary_reaction(
            contact_feature,
            boundary_global_vertex_offset=boundary_global_vertex_offset,
            boundary_vertex_positions_pad_l=mainline_v9._world_from_local(
                boundary_points_l,
                object_position_pad_l,
                tuple(object_quat_pad_l),
            ),
            object_center_pad_l=object_position_pad_l,
            dt=coupling_substep_dt,
        )
        reaction_force_w = _vectors_from_pad_local(
            reaction_force_pad_l, pad_quat_w
        )[0]
        reaction_torque_w = _vectors_from_pad_local(
            reaction_torque_pad_l, pad_quat_w
        )[0]
        return (
            elapsed,
            reaction_force_w,
            reaction_torque_w,
            reaction_vertex_count,
        )

    for warmup_frame in range(int(_v62_args.uipc_warmup_frames)):
        frame_wall_start = time.perf_counter()
        if warmup_frame % max(1, int(args_cli.log_every)) == 0:
            print(
                f"[V62_UIPC_WARMUP_BEGIN] frame={warmup_frame + 1}/"
                f"{int(_v62_args.uipc_warmup_frames)}",
                flush=True,
            )
        warmup_uipc_elapsed = 0.0
        warmup_substep_times: list[float] = []
        for coupling_substep in range(UIPC_SUBSTEPS):
            robot.set_joint_position_target(warmup_joint_target)
            if hasattr(robot, "write_data_to_sim"):
                robot.write_data_to_sim()
            _apply_physx_coupling_wrenches(
                cylinder,
                robot,
                mount_body_idx,
                previous_uipc_force_w,
                previous_uipc_torque_w,
            )
            sim.step(render=False)
            robot.update(coupling_substep_dt)
            cylinder.update(coupling_substep_dt)
            object_position_w, object_quat_w, _ = _object_state(cylinder)
            link_position_w, link_quat_w = mainline_v9._body_pose(robot, mount_body_idx)
            pad_position_w, pad_quat_w = mainline_v9._compose_child_pose(
                link_position_w, link_quat_w, pad_position_l, pad_quat_l
            )
            (
                elapsed,
                raw_reaction_force_w,
                raw_reaction_torque_w,
                _,
            ) = advance_uipc(
                pad_position_w,
                pad_quat_w,
                object_position_w,
                object_quat_w,
                solve_stage="warmup",
                solve_frame=warmup_frame,
                solve_substep=coupling_substep + 1,
            )
            admissible_reaction_force_w, contact_cone_scale = (
                _project_force_to_contact_cone(
                    raw_reaction_force_w,
                    pad_quat_w,
                    float(args_cli.uipc_friction_mu),
                )
            )
            admissible_reaction_torque_w = (
                raw_reaction_torque_w * contact_cone_scale
            )
            relaxation = float(_v62_args.uipc_feedback_relaxation)
            previous_uipc_force_w += relaxation * (
                admissible_reaction_force_w - previous_uipc_force_w
            )
            previous_uipc_torque_w += relaxation * (
                admissible_reaction_torque_w - previous_uipc_torque_w
            )
            previous_uipc_force_w, _ = _limit_vector_norm(
                previous_uipc_force_w, feedback_force_limit_n
            )
            previous_uipc_torque_w, _ = _limit_vector_norm(
                previous_uipc_torque_w, feedback_torque_limit_nm
            )
            warmup_uipc_elapsed += elapsed
            warmup_substep_times.append(elapsed)
        surface_pad_l = _uipc_object_surface(membrane).astype(np.float32)
        if bool(args_cli.render_viewport) and (
            warmup_frame % max(1, int(args_cli.render_every)) == 0
        ):
            _sync_pad_local_membrane_render(
                membrane, surface_pad_l, pad_position_w, pad_quat_w
            )
            # The former UipcSim.update_render_meshes() call rendered the whole
            # viewport as a side effect.  Pad-local membrane synchronization is
            # intentionally narrower, so advance the viewport explicitly to
            # show the live articulation and PhysX object as well.
            sim.render()
        frame_wall_elapsed = time.perf_counter() - frame_wall_start
        if frame_wall_elapsed > float(_v62_args.slow_frame_threshold_sec):
            slowest_substep = int(np.argmax(warmup_substep_times)) + 1
            print(
                f"[V62_SLOW_FRAME] phase=uipc_warmup frame={warmup_frame + 1} "
                f"elapsed={frame_wall_elapsed:.6f}s "
                f"threshold={float(_v62_args.slow_frame_threshold_sec):.6f}s "
                f"uipc_total={warmup_uipc_elapsed:.6f}s "
                f"slowest_uipc_substep={slowest_substep} "
                f"slowest_uipc_time={max(warmup_substep_times):.6f}s "
                f"non_uipc_overhead={max(frame_wall_elapsed - warmup_uipc_elapsed, 0.0):.6f}s "
                "action=continue_and_diagnose",
                flush=True,
            )
        if warmup_frame % max(1, int(args_cli.log_every)) == 0:
            print(
                f"[V62_UIPC_WARMUP_END] frame={warmup_frame + 1} "
                f"uipc_elapsed={warmup_uipc_elapsed:.6f}s "
                f"frame_wall={frame_wall_elapsed:.6f}s",
                flush=True,
            )

    rest_surface_pad_l = surface_pad_l.copy()
    front_surface_indices = np.flatnonzero(contract_front_mask)

    stable_object_position, _, _ = _object_state(cylinder)
    ee_position_w, _ = _ee_pose_w(robot, ee_body_idx, offset_position)
    membrane_front_center_pad_l = np.mean(
        rest_surface_pad_l[front_surface_indices], axis=0, dtype=np.float64
    )
    membrane_center_w = np.asarray(pad_position_w, dtype=np.float64) + _quat_matrix(
        pad_quat_w
    ) @ membrane_front_center_pad_l
    opposing_contact_sensor.update(0.0, force_recompute=True)
    opposing_body_position_w = (
        opposing_contact_sensor.data.pos_w[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=True)
    )
    opposing_body_quat_w = (
        opposing_contact_sensor.data.quat_w[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=True)
    )
    opposing_pad_position_w, opposing_pad_quat_w = mainline_v9._compose_child_pose(
        opposing_body_position_w,
        opposing_body_quat_w,
        opposing_pad_position_body_l,
        opposing_pad_quat_body_l,
    )
    opposing_pad_x_axis_w = _quat_matrix(opposing_pad_quat_w)[:, 0]
    opposing_face_sign = float(
        np.sign(
            np.dot(
                membrane_center_w - opposing_pad_position_w,
                opposing_pad_x_axis_w,
            )
        )
    )
    if opposing_face_sign == 0.0:
        raise RuntimeError("Could not select the opposing rigid Pad contact face")
    opposing_face_center_w = opposing_pad_position_w + (
        opposing_face_sign
        * 0.5
        * float(opposing_pad_scale_m[0])
        * opposing_pad_x_axis_w
    )
    open_contact_midpoint_w = 0.5 * (
        membrane_center_w + opposing_face_center_w
    )
    open_contact_separation_mm = float(
        np.linalg.norm(opposing_face_center_w - membrane_center_w) * 1000.0
    )
    predicted_closed_separation_mm = open_contact_separation_mm - 2.0 * (
        float(args_cli.gripper_opening_mm)
        - float(_v62_args.gripper_closed_mm)
    )
    predicted_normal_compression_mm = (
        2.0 * object_radius_m * 1000.0 - predicted_closed_separation_mm
    )
    print(
        "[V62_GRASP_CENTERING] "
        f"open_face_separation={open_contact_separation_mm:.6f}mm "
        f"closed_command={float(_v62_args.gripper_closed_mm):.6f}mm "
        f"predicted_closed_separation={predicted_closed_separation_mm:.6f}mm "
        f"object_diameter={2.0 * object_radius_m * 1000.0:.6f}mm "
        f"predicted_normal_compression={predicted_normal_compression_mm:+.6f}mm",
        flush=True,
    )
    # Center the cylinder between the two physical contact faces.  The former
    # target put the already-open link8 membrane directly on the cylinder,
    # leaving the link7 Pad tens of millimetres away and guaranteeing a
    # one-sided impact as soon as closing began.
    grasp_position = ee_position_w + (
        stable_object_position - open_contact_midpoint_w
    )
    above_position = grasp_position.copy()
    above_position[2] = float(_v62_args.approach_z)
    lift_position = grasp_position.copy()
    lift_position[2] += float(_v62_args.lift_distance_mm) * 1.0e-3
    motion_program = _build_motion_program(
        home_position,
        above_position,
        grasp_position,
        lift_position,
        opened_mm=float(args_cli.gripper_opening_mm),
        closed_mm=float(_v62_args.gripper_closed_mm),
        close_frames=int(_v62_args.close_frames),
        lift_frames=int(_v62_args.lift_frames),
    )
    planned_motion_frame_count = len(motion_program)
    if 0 < int(_v62_args.max_formal_frames) < planned_motion_frame_count:
        motion_program = motion_program[: int(_v62_args.max_formal_frames)]
        print(
            f"[V62_DIAGNOSTIC_CAP] running {len(motion_program)}/"
            f"{planned_motion_frame_count} formal frames",
            flush=True,
        )

    records = _new_records()
    estimator_config = frozen_7g.EstimatorConfig()
    termination_reason = (
        "looping until interrupted"
        if bool(_v62_args.loop_forever)
        else "diagnostic frame cap reached"
        if len(motion_program) < planned_motion_frame_count
        else "completed"
    )
    pending_error: BaseException | None = None
    completed_cycle_count = 0
    motion_iterator = (
        itertools.cycle(motion_program)
        if bool(_v62_args.loop_forever)
        else iter(motion_program)
    )
    try:
        for record_index, (motion_stage, target_position, target_opening_mm) in enumerate(
            motion_iterator
        ):
            frame_id = record_index % len(motion_program)
            cycle_index = record_index // len(motion_program)
            if not simulation_app.is_running():
                termination_reason = "simulation application stopped"
                break
            if bool(_v62_args.loop_forever) and record_index > 0 and frame_id == 0:
                completed_cycle_count = cycle_index
                records = _new_records()
                print(
                    f"[V62_LOOP] completed_cycle={completed_cycle_count}; restarting motion program",
                    flush=True,
                )
            frame_wall_start = time.perf_counter()
            applied_opening_mm = float(target_opening_mm)
            applied_force_substeps: list[np.ndarray] = []
            applied_torque_substeps: list[np.ndarray] = []
            reaction_force_substeps: list[np.ndarray] = []
            reaction_torque_substeps: list[np.ndarray] = []
            admissible_force_substeps: list[np.ndarray] = []
            admissible_torque_substeps: list[np.ndarray] = []
            contact_cone_scales: list[float] = []
            opposing_contact_force_substeps: list[np.ndarray] = []
            backing_contact_force_substeps: list[np.ndarray] = []
            feedback_force_scales: list[float] = []
            feedback_torque_scales: list[float] = []
            uipc_substep_times: list[float] = []
            uipc_frame_elapsed = 0.0
            uipc_reaction_vertex_count = 0
            for coupling_substep in range(UIPC_SUBSTEPS):
                applied_force_substeps.append(previous_uipc_force_w.copy())
                applied_torque_substeps.append(previous_uipc_torque_w.copy())
                _apply_ik_action(
                    robot,
                    ik_controller,
                    target_position,
                    desired_ee_quat_w,
                    applied_opening_mm,
                    ee_body_idx,
                    jacobi_body_idx,
                    finger_joint_ids,
                    finger_joint_signs,
                    offset_position,
                    offset_rotation,
                )
                _apply_physx_coupling_wrenches(
                    cylinder,
                    robot,
                    mount_body_idx,
                    previous_uipc_force_w,
                    previous_uipc_torque_w,
                )
                sim.step(render=False)
                robot.update(coupling_substep_dt)
                cylinder.update(coupling_substep_dt)
                opposing_contact_sensor.update(
                    coupling_substep_dt, force_recompute=True
                )
                opposing_contact_force_substeps.append(
                    opposing_contact_sensor.data.force_matrix_w[0, 0, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=True)
                )
                backing_contact_sensor.update(
                    coupling_substep_dt, force_recompute=True
                )
                backing_contact_force_substeps.append(
                    backing_contact_sensor.data.force_matrix_w[0, 0, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=True)
                )

                object_position_w, object_quat_w, object_velocity_w = _object_state(
                    cylinder
                )
                link_position_w, link_quat_w = mainline_v9._body_pose(
                    robot, mount_body_idx
                )
                pad_position_w, pad_quat_w = mainline_v9._compose_child_pose(
                    link_position_w, link_quat_w, pad_position_l, pad_quat_l
                )
                (
                    elapsed,
                    substep_reaction_force_w,
                    substep_reaction_torque_w,
                    substep_reaction_vertex_count,
                ) = advance_uipc(
                    pad_position_w,
                    pad_quat_w,
                    object_position_w,
                    object_quat_w,
                    solve_stage=motion_stage,
                    solve_frame=frame_id,
                    solve_substep=coupling_substep + 1,
                )
                reaction_force_substeps.append(substep_reaction_force_w)
                reaction_torque_substeps.append(substep_reaction_torque_w)
                admissible_reaction_force_w, contact_cone_scale = (
                    _project_force_to_contact_cone(
                        substep_reaction_force_w,
                        pad_quat_w,
                        float(args_cli.uipc_friction_mu),
                    )
                )
                admissible_reaction_torque_w = (
                    substep_reaction_torque_w * contact_cone_scale
                )
                admissible_force_substeps.append(admissible_reaction_force_w)
                admissible_torque_substeps.append(admissible_reaction_torque_w)
                contact_cone_scales.append(contact_cone_scale)
                relaxation = float(_v62_args.uipc_feedback_relaxation)
                previous_uipc_force_w += relaxation * (
                    admissible_reaction_force_w - previous_uipc_force_w
                )
                previous_uipc_torque_w += relaxation * (
                    admissible_reaction_torque_w - previous_uipc_torque_w
                )
                previous_uipc_force_w, force_scale = _limit_vector_norm(
                    previous_uipc_force_w, feedback_force_limit_n
                )
                previous_uipc_torque_w, torque_scale = _limit_vector_norm(
                    previous_uipc_torque_w, feedback_torque_limit_nm
                )
                feedback_force_scales.append(force_scale)
                feedback_torque_scales.append(torque_scale)
                uipc_frame_elapsed += elapsed
                uipc_substep_times.append(elapsed)
                uipc_reaction_vertex_count = max(
                    uipc_reaction_vertex_count,
                    int(substep_reaction_vertex_count),
                )

            applied_force_substeps_array = np.asarray(
                applied_force_substeps, dtype=np.float64
            )
            applied_torque_substeps_array = np.asarray(
                applied_torque_substeps, dtype=np.float64
            )
            reaction_force_substeps_array = np.asarray(
                reaction_force_substeps, dtype=np.float64
            )
            reaction_torque_substeps_array = np.asarray(
                reaction_torque_substeps, dtype=np.float64
            )
            admissible_force_substeps_array = np.asarray(
                admissible_force_substeps, dtype=np.float64
            )
            admissible_torque_substeps_array = np.asarray(
                admissible_torque_substeps, dtype=np.float64
            )
            contact_cone_scales_array = np.asarray(
                contact_cone_scales, dtype=np.float64
            )
            opposing_contact_force_substeps_array = np.asarray(
                opposing_contact_force_substeps, dtype=np.float64
            )
            backing_contact_force_substeps_array = np.asarray(
                backing_contact_force_substeps, dtype=np.float64
            )
            feedback_force_scales_array = np.asarray(
                feedback_force_scales, dtype=np.float64
            )
            feedback_torque_scales_array = np.asarray(
                feedback_torque_scales, dtype=np.float64
            )
            applied_uipc_force_w = np.mean(applied_force_substeps_array, axis=0)
            applied_uipc_torque_w = np.mean(applied_torque_substeps_array, axis=0)
            uipc_reaction_force_w = np.mean(reaction_force_substeps_array, axis=0)
            uipc_reaction_torque_w = np.mean(reaction_torque_substeps_array, axis=0)
            opposing_contact_force_w = np.mean(
                opposing_contact_force_substeps_array, axis=0
            )
            backing_contact_force_w = np.mean(
                backing_contact_force_substeps_array, axis=0
            )

            opposing_body_position_w = (
                opposing_contact_sensor.data.pos_w[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=True)
            )
            opposing_body_quat_w = (
                opposing_contact_sensor.data.quat_w[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=True)
            )
            opposing_pad_position_w, opposing_pad_quat_w = (
                mainline_v9._compose_child_pose(
                    opposing_body_position_w,
                    opposing_body_quat_w,
                    opposing_pad_position_body_l,
                    opposing_pad_quat_body_l,
                )
            )

            ee_position_w, ee_quat_w = _ee_pose_w(
                robot, ee_body_idx, offset_position
            )
            object_position_pad_l, object_quat_pad_l = _object_pose_in_pad_frame(
                object_position_w,
                object_quat_w,
                pad_position_w,
                pad_quat_w,
            )
            (
                object_position_opposing_pad_l,
                object_quat_opposing_pad_l,
            ) = _object_pose_in_pad_frame(
                object_position_w,
                object_quat_w,
                opposing_pad_position_w,
                opposing_pad_quat_w,
            )
            rigid_front_object_l = (
                rest_surface_pad_l[front_surface_indices]
                - object_position_pad_l.reshape(1, 3)
            ) @ _quat_matrix(object_quat_pad_l)
            rigid_gap_m = float(
                np.min(
                    _capped_z_cylinder_signed_distance(
                        rigid_front_object_l, object_radius_m, object_height_m
                    ),
                    initial=math.inf,
                )
            )
            if frame_id % max(1, int(args_cli.log_every)) == 0:
                print(
                    f"[V62_FRAME_BEGIN] cycle={cycle_index + 1} "
                    f"frame={frame_id + 1:04d}/{len(motion_program)} "
                    f"motion={motion_stage} rigid_gap={rigid_gap_m * 1000.0:+.4f}mm "
                    f"opening_command={applied_opening_mm:.3f}mm",
                    flush=True,
                )

            # This is a solver-state readback after world.advance()/retrieve(),
            # not the PhysX-derived target passed to synchronize().  Comparing
            # both positions makes a broken or misframed UIPC boundary visible
            # even when the geometric PhysX gap is already negative.
            uipc_object_position_pad_l = boundary_driver.actual_position_pad_l()
            uipc_object_position_w = _points_from_pad_local(
                uipc_object_position_pad_l.reshape(1, 3),
                pad_position_w,
                pad_quat_w,
            )[0]
            uipc_position_delta_mm = (
                uipc_object_position_w - object_position_w
            ) * 1000.0
            uipc_position_error_mm = float(np.linalg.norm(uipc_position_delta_mm))

            surface_pad_l = _uipc_object_surface(membrane).astype(np.float32)
            uipc_boundary_surface_pad_l = _uipc_object_surface(
                object_boundary
            ).astype(np.float64)
            expected_boundary_surface_pad_l = mainline_v9._world_from_local(
                boundary_points_l,
                object_position_pad_l,
                tuple(object_quat_pad_l),
            )
            boundary_surface_sync_error_m = _symmetric_nearest_distance(
                uipc_boundary_surface_pad_l, expected_boundary_surface_pad_l
            )
            deformation = (surface_pad_l - rest_surface_pad_l).astype(np.float32)
            normal_deformation_m = np.clip(
                -deformation[front_surface_indices, 0].astype(np.float64), 0.0, None
            )
            max_normal_deformation_m = float(
                np.max(normal_deformation_m, initial=0.0)
            )
            front_object_l = (
                surface_pad_l[front_surface_indices]
                - object_position_pad_l.reshape(1, 3)
            ) @ _quat_matrix(object_quat_pad_l)
            signed_gap_m = _capped_z_cylinder_signed_distance(
                front_object_l, object_radius_m, object_height_m
            )
            min_signed_gap_m = float(np.min(signed_gap_m, initial=math.inf))
            reaction_force_norm_n = float(
                np.max(np.linalg.norm(reaction_force_substeps_array, axis=1))
            )
            contact_active = bool(
                reaction_force_norm_n > float(_v62_args.contact_force_epsilon_n)
            )
            if contact_active:
                force_result = frozen_7g.estimate_deformation_force(
                    deformation,
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
            else:
                force_pad_local = np.zeros(3, dtype=np.float64)
                tactile_force_channels = np.zeros(3, dtype=np.float64)

            actual_opening_mm = float(mainline_v9._read_gripper_opening_mm(robot))

            records["frame_id"].append(frame_id)
            records["motion_stage"].append(motion_stage)
            records["surface_displacement_pad_local"].append(deformation)
            records["force_pad_local"].append(force_pad_local)
            records["tactile_force_channels"].append(tactile_force_channels)
            records["contact_active"].append(contact_active)
            records["minimum_signed_gap_mm"].append(min_signed_gap_m * 1000.0)
            records["maximum_normal_deformation_mm"].append(
                max_normal_deformation_m * 1000.0
            )
            records["object_pose_w"].append(
                np.asarray([*object_position_w, *object_quat_w], dtype=np.float32)
            )
            records["object_pose_pad_local"].append(
                np.asarray(
                    [*object_position_pad_l, *object_quat_pad_l], dtype=np.float32
                )
            )
            records["object_pose_opposing_pad_local"].append(
                np.asarray(
                    [
                        *object_position_opposing_pad_l,
                        *object_quat_opposing_pad_l,
                    ],
                    dtype=np.float32,
                )
            )
            records["object_velocity_w"].append(object_velocity_w.astype(np.float32))
            records["pad_pose_w"].append(
                np.asarray([*pad_position_w, *pad_quat_w], dtype=np.float32)
            )
            records["opposing_pad_pose_w"].append(
                np.asarray(
                    [*opposing_pad_position_w, *opposing_pad_quat_w],
                    dtype=np.float32,
                )
            )
            records["end_effector_pose_w"].append(
                np.asarray([*ee_position_w, *ee_quat_w], dtype=np.float32)
            )
            records["gripper_opening_mm"].append(
                actual_opening_mm
            )
            records["commanded_gripper_opening_mm"].append(applied_opening_mm)
            records["uipc_reaction_force_w"].append(uipc_reaction_force_w)
            records["uipc_reaction_torque_w"].append(uipc_reaction_torque_w)
            records["applied_uipc_force_w"].append(applied_uipc_force_w)
            records["applied_uipc_torque_w"].append(applied_uipc_torque_w)
            records["uipc_reaction_force_substeps_w"].append(
                reaction_force_substeps_array
            )
            records["uipc_reaction_torque_substeps_w"].append(
                reaction_torque_substeps_array
            )
            records["uipc_admissible_force_substeps_w"].append(
                admissible_force_substeps_array
            )
            records["uipc_admissible_torque_substeps_w"].append(
                admissible_torque_substeps_array
            )
            records["applied_uipc_force_substeps_w"].append(
                applied_force_substeps_array
            )
            records["applied_uipc_torque_substeps_w"].append(
                applied_torque_substeps_array
            )
            records["opposing_contact_force_w"].append(opposing_contact_force_w)
            records["opposing_contact_force_substeps_w"].append(
                opposing_contact_force_substeps_array
            )
            records["backing_contact_force_w"].append(backing_contact_force_w)
            records["backing_contact_force_substeps_w"].append(
                backing_contact_force_substeps_array
            )
            records["uipc_feedback_force_scale_substeps"].append(
                feedback_force_scales_array
            )
            records["uipc_feedback_torque_scale_substeps"].append(
                feedback_torque_scales_array
            )
            records["uipc_contact_cone_scale_substeps"].append(
                contact_cone_scales_array
            )
            records["uipc_boundary_surface_sync_error_mm"].append(
                boundary_surface_sync_error_m * 1000.0
            )
            records["uipc_reaction_vertex_count"].append(uipc_reaction_vertex_count)
            records["uipc_step_time_sec"].append(uipc_frame_elapsed)
            records["uipc_substep_time_sec"].append(
                np.asarray(uipc_substep_times, dtype=np.float64)
            )

            if bool(args_cli.render_viewport) and (
                frame_id % max(1, int(args_cli.render_every)) == 0
            ):
                _sync_pad_local_membrane_render(
                    membrane, surface_pad_l, pad_position_w, pad_quat_w
                )
                sim.render()
            frame_wall_elapsed = time.perf_counter() - frame_wall_start
            records["frame_wall_time_sec"].append(frame_wall_elapsed)
            if frame_wall_elapsed > float(_v62_args.slow_frame_threshold_sec):
                slowest_substep = int(np.argmax(uipc_substep_times)) + 1
                print(
                    f"[V62_SLOW_FRAME] phase=formal cycle={cycle_index + 1} "
                    f"frame={frame_id + 1:04d}/{len(motion_program)} "
                    f"motion={motion_stage} elapsed={frame_wall_elapsed:.6f}s "
                    f"threshold={float(_v62_args.slow_frame_threshold_sec):.6f}s "
                    f"uipc_total={uipc_frame_elapsed:.6f}s "
                    f"slowest_uipc_substep={slowest_substep} "
                    f"slowest_uipc_time={max(uipc_substep_times):.6f}s "
                    f"non_uipc_overhead={max(frame_wall_elapsed - uipc_frame_elapsed, 0.0):.6f}s "
                    f"contact={int(contact_active)} gap_mm={min_signed_gap_m * 1000.0:+.6f} "
                    f"opposing_contact_peak_n="
                    f"{np.max(np.linalg.norm(opposing_contact_force_substeps_array, axis=1)):.6f} "
                    f"backing_contact_peak_n="
                    f"{np.max(np.linalg.norm(backing_contact_force_substeps_array, axis=1)):.6f} "
                    f"raw_uipc_peak_n="
                    f"{np.max(np.linalg.norm(reaction_force_substeps_array, axis=1)):.6f} "
                    f"admissible_uipc_peak_n="
                    f"{np.max(np.linalg.norm(admissible_force_substeps_array, axis=1)):.6f} "
                    f"applied_uipc_peak_n="
                    f"{np.max(np.linalg.norm(applied_force_substeps_array, axis=1)):.6f} "
                    f"minimum_contact_cone_scale="
                    f"{np.min(contact_cone_scales_array):.6f} "
                    f"minimum_feedback_force_scale="
                    f"{np.min(feedback_force_scales_array):.6f} "
                    f"boundary_surface_sync_error_mm="
                    f"{boundary_surface_sync_error_m * 1000.0:.6f} "
                    "action=continue_and_diagnose",
                    flush=True,
                )
            if frame_id % max(1, int(args_cli.log_every)) == 0:
                print(
                    "[V62_CYLINDER_SYNC] "
                    f"physx_position_w_m={object_position_w.tolist()} "
                    f"uipc_actual_position_w_m={uipc_object_position_w.tolist()} "
                    f"delta_xyz_mm={uipc_position_delta_mm.tolist()} "
                    f"error_mm={uipc_position_error_mm:.6f}",
                    flush=True,
                )
                print(
                    f"[V62] cycle={cycle_index + 1} "
                    f"frame={frame_id + 1:04d}/{len(motion_program)} "
                    f"motion={motion_stage} contact={int(contact_active)} "
                    f"gap={min_signed_gap_m * 1000.0:+.4f}mm "
                    f"deformation={max_normal_deformation_m * 1000.0:.4f}mm "
                    f"uipc={uipc_frame_elapsed:.6f}s "
                    f"frame_wall={frame_wall_elapsed:.6f}s "
                    f"opening={actual_opening_mm:.3f}mm "
                    f"reaction={uipc_reaction_force_w.tolist()}N "
                    f"opposing_contact={opposing_contact_force_w.tolist()}N "
                    f"backing_contact={backing_contact_force_w.tolist()}N "
                    f"feedback_force_scale_min="
                    f"{np.min(feedback_force_scales_array):.6f} "
                    f"contact_cone_scale_min="
                    f"{np.min(contact_cone_scales_array):.6f} "
                    f"boundary_surface_sync_error_mm="
                    f"{boundary_surface_sync_error_m * 1000.0:.6f} "
                    f"object_opposing_pad_l_mm="
                    f"{(object_position_opposing_pad_l * 1000.0).tolist()} "
                    f"Fxyz={tactile_force_channels.tolist()}",
                    flush=True,
                )
            if frame_id + 1 == len(motion_program):
                completed_cycle_count = cycle_index + 1
                if bool(_v62_args.loop_forever):
                    print(
                        f"[V62_LOOP] completed_cycle={completed_cycle_count}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        termination_reason = "KeyboardInterrupt requested by user"
    except BaseException as exc:
        termination_reason = f"{type(exc).__name__}: {exc}"
        pending_error = exc
    finally:
        metadata = {
            "version": "v6.2_simple_grasp_tactile",
            "completed_frame_count": len(records["frame_id"]),
            "completed_cycle_count": completed_cycle_count,
            "planned_frame_count": planned_motion_frame_count,
            "executed_frame_limit": len(motion_program),
            "termination_reason": termination_reason,
            "force_source": "frozen_7g_membrane_deformation_only",
            "force_contract": {
                "surface_coordinate_frame": "pad_local",
                "surface_deformation_definition": "current_minus_rest",
                "force_pad_local_channel_order": ["X", "Y", "Z"],
                "tactile_force_channel_order": ["Fx", "Fy", "Fz"],
                "tactile_from_pad_local": ["pad_Y", "-pad_Z", "-pad_X"],
                "unit": "TU",
            },
            "contact_gate": {
                "role": "native_uipc_reaction_output_validity_only",
                "reaction_force_threshold_n": float(
                    _v62_args.contact_force_epsilon_n
                ),
                "inactive_output": [0.0, 0.0, 0.0],
            },
            "time_step": {
                "record_dt_sec": record_dt,
                "physx_step_dt_sec": coupling_substep_dt,
                "uipc_step_dt_sec": coupling_substep_dt,
                "coupling_substeps_per_record": UIPC_SUBSTEPS,
                "uipc_substeps_per_physx_step": 1,
                "uipc_boundary_motion": "authoritative_pose_after_each_physx_substep",
                "reaction_substep_reduction": "time_average",
                "reaction_feedback_relaxation": float(
                    _v62_args.uipc_feedback_relaxation
                ),
                "reaction_feedback_force_limit_n": feedback_force_limit_n,
                "reaction_feedback_torque_limit_nm": feedback_torque_limit_nm,
                "reaction_contact_cone_friction_mu": float(
                    args_cli.uipc_friction_mu
                ),
                "reaction_feedback_force_scale_array": (
                    "uipc_feedback_force_scale_substeps.npy"
                ),
                "reaction_feedback_torque_scale_array": (
                    "uipc_feedback_torque_scale_substeps.npy"
                ),
                "reaction_contact_cone_force_array": (
                    "uipc_admissible_force_substeps_w.npy"
                ),
                "reaction_contact_cone_torque_array": (
                    "uipc_admissible_torque_substeps_w.npy"
                ),
                "reaction_contact_cone_scale_array": (
                    "uipc_contact_cone_scale_substeps.npy"
                ),
                "slow_frame_threshold_sec": float(
                    _v62_args.slow_frame_threshold_sec
                ),
                "slow_frame_is_failure": False,
                "slow_frame_action": "continue_and_record_substep_diagnostics",
            },
            "uipc_coupling": {
                "coordinate_frame": "link8_pad_local",
                "solver_membrane_path": membrane_mesh_path,
                "solver_membrane_count": 1,
                "membrane_back_face": "fixed_pad_local_rest_targets",
                "membrane_front_face": "uipc_solved_in_pad_local_frame",
                "external_boundary_source": "inverse(T_world_pad) * T_world_object",
                "external_boundary_driver": "kinematic_collision_boundary_with_pose_history_for_ccd",
                "external_boundary_has_independent_dynamics": False,
                "continuous_collision_detection": True,
                "reaction_source": (
                    "negative_UIPC_incremental_contact_potential_gradient_"
                    "on_boundary_vertices_divided_by_uipc_dt_squared"
                ),
                "reaction_frame_conversion": "pad_local_vector_rotated_to_world",
                "reaction_force_unit": "N",
                "reaction_torque_unit": "N*m",
                "reaction_feedback_timing": (
                    "UIPC coupling substep n reaction applied to PhysX coupling "
                    "substep n+1 after unilateral Coulomb-cone projection, "
                    "explicit under-relaxation, and norm limiting"
                ),
                "reaction_feedback_regularization": (
                    "positive-Pad-X unilateral normal and tangential norm <= "
                    "uipc_friction_mu*normal; force_norm_limited_to_bound "
                    "one-substep_ballistic_motion; "
                    "torque_norm_limited_by_force_limit_times_object_radius"
                ),
                "physx_reaction_recipients": (
                    "object_wrench_and_equal_opposite_link8_wrench"
                ),
                "link8_reaction_moment_transfer": (
                    "-object_torque-(object_com-link8_com)x_object_force"
                ),
                "tactile_tu_is_applied_to_physx": False,
                "link7_uipc_representation": False,
                "initial_relative_placement_preserved": True,
                "initial_world_reconstruction_error_m": pad_local_reconstruction_error_m,
            },
            "opposing_contact": {
                "path": opposing_pad_path,
                "mounted_body": "link7",
                "contact_report_body": OPPOSING_CONTACT_BODY_PATH,
                "fixed_joint_parent": "link7",
                "representation": "PhysX_rigid_cube_collider",
                "collision_enabled": True,
                "filtered_contact_force": "GraspCylinder_only",
                "force_substep_array": "opposing_contact_force_substeps_w.npy",
                "pad_pose_array": "opposing_pad_pose_w.npy",
                "object_relative_pose_array": (
                    "object_pose_opposing_pad_local.npy"
                ),
                "purpose": "symmetric_opposition_to_link8_uipc_membrane",
            },
            "membrane_rigid_backing": {
                "path": backing_path,
                "mounted_body": "link8",
                "contact_report_body": mount_link_path,
                "representation": "authored_box_mesh_convex_hull_collider",
                "collision_enabled": True,
                "membrane_front_to_backing_mm": 0.5,
                "contact_offset_mm": 0.25,
                "rest_offset_mm": 0.0,
                "filtered_contact_force": "GraspCylinder_only",
                "force_array": "backing_contact_force_w.npy",
                "force_substep_array": "backing_contact_force_substeps_w.npy",
                "purpose": (
                    "physical_sensor_backing_prevents_penetration_after_full_"
                    "membrane_thickness_compression"
                ),
            },
            "grasp_centering": {
                "target_reference": (
                    "midpoint_of_open_link8_membrane_front_and_link7_rigid_pad_face"
                ),
                "former_one_sided_target_removed": True,
                "open_contact_face_separation_mm": open_contact_separation_mm,
                "predicted_closed_contact_face_separation_mm": (
                    predicted_closed_separation_mm
                ),
                "object_diameter_mm": 2.0 * object_radius_m * 1000.0,
                "predicted_normal_compression_mm": (
                    predicted_normal_compression_mm
                ),
            },
            "grasp_orientation": {
                "controller_command_type": "pose",
                "cylinder_axis_world": [0.0, 0.0, 1.0],
                "aligned_pad_tangent_axis": "pad_local_y",
                "pad_normal_to_horizontal_error_deg": (
                    pad_normal_to_horizontal_error_deg
                ),
                "pad_tangent_to_cylinder_axis_error_deg": (
                    pad_tangent_to_cylinder_axis_error_deg
                ),
                "desired_ee_quat_wxyz": desired_ee_quat_w.tolist(),
                "desired_pad_quat_wxyz": desired_pad_quat_w.tolist(),
            },
            "motion_stage_specs": [
                [
                    stage,
                    int(_v62_args.close_frames)
                    if stage == "close"
                    else int(_v62_args.lift_frames)
                    if stage == "lift"
                    else frame_count,
                ]
                for stage, frame_count in MOTION_STAGE_SPECS
            ],
            "gripper_drive": {
                "control": "scheduled_position_target_with_finite_implicit_drive",
                "gap_or_tactile_feedback": False,
                "closed_target_mm": float(_v62_args.gripper_closed_mm),
                "close_frames": int(_v62_args.close_frames),
                "stiffness": float(_v62_args.gripper_drive_stiffness),
                "damping": float(_v62_args.gripper_drive_damping),
                "effort_limit_n": float(_v62_args.gripper_effort_limit_n),
                "velocity_limit_m_s": float(
                    _v62_args.gripper_closing_velocity_m_s
                ),
            },
            "lift_motion": {
                "distance_mm": float(_v62_args.lift_distance_mm),
                "frames": int(_v62_args.lift_frames),
                "record_rate_hz": float(args_cli.sim_hz),
                "coupling_substeps_per_record": UIPC_SUBSTEPS,
                "interpolation": "smoothstep",
                "nominal_peak_command_increment_per_coupling_substep_mm": (
                    1.5
                    * float(_v62_args.lift_distance_mm)
                    / float(_v62_args.lift_frames)
                    / float(UIPC_SUBSTEPS)
                ),
            },
            "physx_object_authority": {
                "authoritative_state": "PhysX GraspCylinder",
                "formal_motion_pose_write_count": 0,
                "initialization_pose_source": "RigidObjectCfg.InitialStateCfg",
                "uipc_is_free_rigid_body": False,
            },
            "contract_vertex_mapping": {
                "method": "one_to_one_nearest_rest_position",
                "permuted_vertex_count": int(
                    np.count_nonzero(
                        runtime_to_contract != np.arange(runtime_to_contract.size)
                    )
                ),
                "maximum_rest_position_error_mm": contract_mapping_error_m * 1000.0,
            },
            "object_radius_mm": float(_v62_args.object_radius_mm),
            "object_height_mm": float(_v62_args.object_height_mm),
            "estimator": {
                "normal_gain_tu_per_m3": estimator_config.normal_gain_tu_per_m3,
                "tangent_y_gain_tu_per_m3": estimator_config.tangent_y_gain_tu_per_m3,
                "tangent_z_gain_tu_per_m3": estimator_config.tangent_z_gain_tu_per_m3,
                "activation_start_m": estimator_config.activation_start_m,
                "activation_full_m": estimator_config.activation_full_m,
            },
            "surface_topology": {
                "surface_triangles_saved": True,
                "front_surface_triangles_saved": True,
                "triangle_indexing": "runtime_compact_surface_vertex_index",
            },
            "online_tactile_field": False,
            "camera": False,
            "viewport": bool(args_cli.render_viewport),
            "loop_forever": bool(_v62_args.loop_forever),
            "record_scope": (
                "current_motion_cycle" if bool(_v62_args.loop_forever) else "single_motion_cycle"
            ),
            "contract_dir": str(contract_dir),
            "workspace_dir": str(workspace_dir),
        }
        _save_dataset(
            output_dir,
            records,
            rest_surface_pad_l=rest_surface_pad_l,
            contract_vertex_area=contract_vertex_area,
            contract_front_mask=contract_front_mask,
            surface_triangles=runtime_surface_triangles,
            metadata=metadata,
        )
        if pending_error is not None:
            _json_write(
                output_dir / "error.json",
                {"error": termination_reason, "completed_frame_count": len(records["frame_id"])},
            )
        simulation_app.close()

    if pending_error is not None:
        raise pending_error
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "completed_frame_count": len(records["frame_id"]),
                "termination_reason": termination_reason,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        fatal_reason = f"{type(exc).__name__}: {exc}"
        print(f"[V62_FATAL] {fatal_reason}", flush=True)
        traceback.print_exc()
        try:
            fatal_output_dir = Path(args_cli.output_dir).expanduser().resolve()
            fatal_output_dir.mkdir(parents=True, exist_ok=True)
            _json_write(
                fatal_output_dir / "error.json",
                {"error": fatal_reason, "traceback": traceback.format_exc()},
            )
        except BaseException as save_exc:
            print(f"[V62_FATAL_SAVE_FAILED] {save_exc}", flush=True)
        try:
            if simulation_app.is_running():
                simulation_app.close()
        except BaseException:
            pass
        raise
