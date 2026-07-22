from __future__ import annotations

"""V6.0 free-object grasp/lift integration on the frozen 7f -> 7g -> v9 path.

PhysX is the sole authority for the robot and free cylinder.  The cylinder pose is
mirrored, one way, to an affine UIPC contact body before every UIPC solve.  The
only tactile source is the rest-relative deformation of the link8-mounted
``simulation/membrane_sim_mesh``; contact geometry is diagnostic only.

This file intentionally imports the passed v9 executable as a read-only runtime
library.  Its AppLauncher owns the Isaac application, while this module supplies
only the new IK, free-rigid-body, phase-state, synchronization, and acceptance
logic.  New v6-only command-line options are removed before v9 parses the shared
IsaacLab and frozen-mainline options.
"""

import argparse
import hashlib
import json
import math
import os
import sys
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
_v6_parser.add_argument("--home_frames", type=int, default=30)
_v6_parser.add_argument("--approach_frames", type=int, default=45)
_v6_parser.add_argument("--lower_frames", type=int, default=70)
_v6_parser.add_argument("--close_frames_v6", type=int, default=35)
_v6_parser.add_argument("--confirm_grasp_frames", type=int, default=15)
_v6_parser.add_argument("--lift_frames", type=int, default=65)
_v6_parser.add_argument("--check_grasp_frames", type=int, default=10)
_v6_parser.add_argument("--hold_lifted_frames", type=int, default=30)
_v6_parser.add_argument("--lower_to_place_frames", type=int, default=65)
_v6_parser.add_argument("--confirm_support_frames", type=int, default=20)
_v6_parser.add_argument("--release_frames_v6", type=int, default=35)
_v6_parser.add_argument("--retreat_frames", type=int, default=45)
_v6_parser.add_argument("--final_recovery_frames", type=int, default=30)
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
_v6_args, _v9_argv = _v6_parser.parse_known_args()

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0], *_v9_argv]
import OpenWorldTactile_v5_new_9_tu_tactile_field_rendering as mainline_v9
sys.argv = _original_argv

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_frozen_hashes() -> dict[str, str]:
    paths = list(tactile_field_v9.frozen_source_paths())
    paths.append(Path(tactile_field_v9.__file__).resolve())
    return {path.name: _sha256(path) for path in paths}


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
    ):
        if float(getattr(_v6_args, name)) <= 0.0:
            raise ValueError(f"--{name} must be > 0")
    if not 0.0 <= float(_v6_args.gripper_closed_mm) < float(args_cli.gripper_opening_mm):
        raise ValueError("Require 0 <= --gripper_closed_mm < --gripper_opening_mm")
    if len(_v6_args.piper_tip_offset) != 3:
        raise ValueError("--piper_tip_offset requires three values")
    for name in (
        "initial_settle_frames",
        "home_frames",
        "approach_frames",
        "lower_frames",
        "close_frames_v6",
        "confirm_grasp_frames",
        "lift_frames",
        "check_grasp_frames",
        "hold_lifted_frames",
        "lower_to_place_frames",
        "confirm_support_frames",
        "release_frames_v6",
        "retreat_frames",
        "final_recovery_frames",
    ):
        if int(getattr(_v6_args, name)) <= 0:
            raise ValueError(f"--{name} must be > 0")
    if int(_v6_args.hold_lifted_frames) < 30:
        raise ValueError("--hold_lifted_frames must be at least 30")
    if int(args_cli.warmup_max_stability_steps) < (
        int(args_cli.uipc_warmup_steps) + int(args_cli.warmup_stability_frames)
    ):
        raise ValueError(
            "--warmup_max_stability_steps must cover the unrecorded UIPC prewarm "
            "and the consecutive stability window"
        )
    if int(args_cli.membrane_cells_x) != 1 or int(args_cli.membrane_cells_y) != 22 or int(args_cli.membrane_cells_z) != 26:
        raise ValueError("v6.0 freezes the structured membrane at 1x22x26 cells")
    if str(args_cli.membrane_mesh_mode) != "structured":
        raise ValueError("v6.0 requires the frozen structured membrane")
    if not math.isclose(float(args_cli.youngs_modulus_mpa), 0.05) or not math.isclose(float(args_cli.poisson_rate), 0.49) or not math.isclose(float(args_cli.mass_density), 1050.0):
        raise ValueError("v6.0 freezes membrane material at E=0.05 MPa, nu=0.49, density=1050")
    if int(args_cli.field_height) != 81 or int(args_cli.field_width) != 65 or not math.isclose(float(args_cli.gaussian_sigma_cells), 1.25):
        raise ValueError("v6.0 freezes the v9 field at 81x65 and sigma=1.25 cells")
    if int(_v6_args.episode_index) < 0:
        raise ValueError("--episode_index must be >= 0")
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


def _build_waypoints(object_position: np.ndarray) -> dict[str, np.ndarray]:
    cx, cy = float(object_position[0]), float(object_position[1])
    norm = max(math.hypot(cx, cy), EPS)
    grasp = np.asarray(
        (
            cx + cx / norm * float(_v6_args.grasp_forward_offset),
            cy + cy / norm * float(_v6_args.grasp_forward_offset),
            float(object_position[2]) + float(_v6_args.grasp_z_offset),
        ),
        dtype=np.float64,
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


def _phase_plan(waypoints: dict[str, np.ndarray]) -> list[dict[str, object]]:
    opened = float(args_cli.gripper_opening_mm)
    closed = float(_v6_args.gripper_closed_mm)
    return [
        {"name": "HOME", "target": waypoints["home"], "opening": opened, "frames": int(_v6_args.home_frames)},
        {"name": "APPROACH_PICK", "target": waypoints["above_pick"], "opening": opened, "frames": int(_v6_args.approach_frames)},
        {"name": "LOWER_TO_GRASP", "target": waypoints["grasp"], "opening": opened, "frames": int(_v6_args.lower_frames)},
        {"name": "CLOSE_GRIPPER", "target": waypoints["grasp"], "opening": closed, "frames": int(_v6_args.close_frames_v6)},
        {"name": "CONFIRM_GRASP", "target": waypoints["grasp"], "opening": closed, "frames": int(_v6_args.confirm_grasp_frames)},
        {"name": "LIFT_OBJECT", "target": waypoints["lift"], "opening": closed, "frames": int(_v6_args.lift_frames)},
        {"name": "CHECK_GRASP", "target": waypoints["lift"], "opening": closed, "frames": int(_v6_args.check_grasp_frames)},
        {"name": "HOLD_LIFTED", "target": waypoints["lift"], "opening": closed, "frames": int(_v6_args.hold_lifted_frames)},
        {"name": "LOWER_TO_PLACE", "target": waypoints["place"], "opening": closed, "frames": int(_v6_args.lower_to_place_frames)},
        {"name": "CONFIRM_SUPPORT", "target": waypoints["place"], "opening": closed, "frames": int(_v6_args.confirm_support_frames)},
        {"name": "RELEASE_GRIPPER", "target": waypoints["place"], "opening": opened, "frames": int(_v6_args.release_frames_v6)},
        {"name": "RETREAT", "target": waypoints["above_pick"], "opening": opened, "frames": int(_v6_args.retreat_frames)},
        {"name": "FINAL_RECOVERY", "target": waypoints["home"], "opening": opened, "frames": int(_v6_args.final_recovery_frames)},
    ]


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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _save_record_arrays(output_dir: Path, records: dict[str, list[object]]) -> None:
    array_names = {
        "robot_joint_position": "robot_joint_position.npy",
        "robot_joint_velocity": "robot_joint_velocity.npy",
        "gripper_opening_mm": "gripper_opening_mm.npy",
        "end_effector_pose_w": "end_effector_pose_w.npy",
        "pad_pose_w": "pad_pose_w.npy",
        "object_pose_w": "object_pose_w.npy",
        "object_velocity_w": "object_velocity_w.npy",
        "object_angular_velocity_w": "object_angular_velocity_w.npy",
        "object_lift_height_mm": "object_lift_height_mm.npy",
        "object_gripper_distance_mm": "object_gripper_distance_mm.npy",
        "object_relative_pose_to_gripper": "object_relative_pose_to_gripper.npy",
        "physx_uipc_object_position_error_mm": "physx_uipc_object_position_error_mm.npy",
        "physx_uipc_object_orientation_error_deg": "physx_uipc_object_orientation_error_deg.npy",
        "uipc_surface_w": "uipc_surface_w.npy",
        "surface_deformation": "surface_deformation.npy",
        "contact_vertex_mask": "contact_vertex_mask.npy",
        "contact_vertex_count": "contact_vertex_count.npy",
        "max_penetration_mm": "max_penetration_mm.npy",
        "max_front_deformation_mm": "max_front_deformation_mm.npy",
        "max_normal_compression_mm": "max_normal_compression_mm.npy",
        "back_target_error_mm": "back_target_error_mm.npy",
        "support_clearance_mm": "support_clearance_mm.npy",
        "commanded_opening_mm": "commanded_opening_mm.npy",
        "cycle_index": "cycle_index.npy",
    }
    for key, filename in array_names.items():
        dtype = bool if key == "contact_vertex_mask" else None
        np.save(output_dir / filename, np.asarray(records[key], dtype=dtype))
    # Compatibility input for the unchanged v9 rendering functions.
    np.save(
        output_dir / "measured_opening_mm.npy",
        np.asarray(records["gripper_opening_mm"], dtype=np.float32),
    )
    _json_write(output_dir / "phase_history.json", records["phase"])


def main() -> None:
    _validate_v6_args()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "error.json").unlink(missing_ok=True)
    contract_dir = Path(args_cli.contract_dir).expanduser().resolve()
    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)

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

    robot = mainline_v9._make_native_piper_articulation()
    object_height_m = float(_v6_args.object_height_mm) * 1.0e-3
    object_radius_m = float(_v6_args.object_radius_mm) * 1.0e-3
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

    scene_camera = None
    if bool(args_cli.save_camera_rgb):
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
        eye = torch.tensor([[0.58, -0.48, 0.42]], device=scene_camera.device)
        target = torch.tensor([[0.28, -0.01, 0.13]], device=scene_camera.device)
        scene_camera.set_world_poses_from_view(eye, target)

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
            dt=sim_dt,
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
    initial_mirror_vertices_w = mainline_v9._uipc_vertices(mirror).copy()
    initial_mirror_surface_w = mainline_v9._uipc_surface(mirror).copy()

    def write_mirror_pose(target_matrix: np.ndarray, frame_index: int) -> None:
        target_vertices_w = mainline_v9._world_from_affine_matrix(
            initial_mirror_vertices_w, target_matrix
        )
        mirror.write_vertex_positions_to_sim(
            torch.as_tensor(
                target_vertices_w, device=sim.device, dtype=torch.float64
            ).contiguous()
        )
        mirror_state["target_matrix"] = np.asarray(target_matrix, dtype=np.float64)
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
    for stability_step in range(int(args_cli.warmup_max_stability_steps)):
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
        for substep_index in range(substep_count):
            substep_alpha = float(substep_index + 1) / float(substep_count)
            interpolated_position = previous_mirror_position_w + substep_alpha * (
                object_position_w - previous_mirror_position_w
            )
            interpolated_quat = _quat_slerp_wxyz(
                previous_mirror_quat_w, object_quat_w, substep_alpha
            )
            target_matrix = _pose_delta_matrix(
                interpolated_position,
                interpolated_quat,
                mirror_initial_position,
                mirror_initial_quat,
            )
            write_mirror_pose(target_matrix, -(stability_step + 1))
            attachment._compute_aim_positions()
            manual_uipc_step()
        previous_mirror_position_w = np.asarray(object_position_w, dtype=np.float64).copy()
        previous_mirror_quat_w = np.asarray(object_quat_w, dtype=np.float64).copy()
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
        mirror_transform = np.asarray(
            mirror.geo_slot_list[0].geometry().transforms().view()[0], dtype=np.float64
        )
        mirror_surface_w = mainline_v9._world_from_affine_matrix(
            initial_mirror_surface_w, mirror_transform
        )
        mirror_surface_pad_l = mainline_v9._local_from_world(
            mirror_surface_w, pad_position_w, pad_quat_w
        )
        _, contact_diagnostics = mainline_v9._contact_geometry_diagnostics(
            surface_pad_l,
            provisional_rest,
            provisional_front_indices,
            tool_min_x_l=float(np.min(mirror_surface_pad_l[:, 0])),
            tool_center_yz_l=np.mean(mirror_surface_pad_l[:, 1:3], axis=0),
            tool_footprint_half_extent_m=object_radius_m,
            tool_shape="cylinder",
        )
        contact_count = int(contact_diagnostics["contact_vertex_count"])
        stable = bool(
            contact_count == 0
            and float(np.linalg.norm(force_vector)) < float(args_cli.accept_max_warmup_force_tu)
        )
        in_unrecorded_prewarm = stability_step < int(args_cli.uipc_warmup_steps)
        if in_unrecorded_prewarm:
            # Mandatory no-record prewarm precedes the consecutive 30-frame stability gate.
            provisional_rest = surface_pad_l.copy()
            consecutive_stable = 0
            warmup_force.clear()
            warmup_contact.clear()
            warmup_surface.clear()
        elif stable:
            consecutive_stable += 1
            warmup_force.append(force_vector.copy())
            warmup_contact.append(contact_count)
            warmup_surface.append(surface_pad_l.copy())
        else:
            provisional_rest = surface_pad_l.copy()
            consecutive_stable = 0
            warmup_force.clear()
            warmup_contact.clear()
            warmup_surface.clear()
        if stability_step % max(1, int(args_cli.log_every)) == 0:
            print(
                f"[V6_INIT_UIPC] step={stability_step + 1:04d} "
                f"prewarm={in_unrecorded_prewarm} "
                f"stable={consecutive_stable:02d}/{int(args_cli.warmup_stability_frames)} "
                f"force={np.linalg.norm(force_vector):.6g}TU contact={contact_count}",
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
    waypoints = _build_waypoints(stable_object_position)
    phases = _phase_plan(waypoints)
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
        )
    }
    total_formal_frames = sum(int(phase["frames"]) for phase in phases)
    scene_video_path = output_dir / "free_object_grasp_scene.mp4"
    scene_writer = None
    if scene_camera is not None:
        scene_writer = cv2.VideoWriter(
            str(scene_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args_cli.video_fps),
            (int(args_cli.camera_width), int(args_cli.camera_height)),
        )
        if not scene_writer.isOpened():
            raise RuntimeError(f"Could not open scene video: {scene_video_path}")

    formal_frame = 0
    previous_target = waypoints["home"].copy()
    previous_opening = float(args_cli.gripper_opening_mm)
    try:
        for phase in phases:
            phase_name = str(phase["name"])
            target = np.asarray(phase["target"], dtype=np.float64)
            target_opening = float(phase["opening"])
            frame_count = int(phase["frames"])
            print(f"[V6_PHASE] {phase_name}", flush=True)
            for phase_frame in range(frame_count):
                if not simulation_app.is_running():
                    raise RuntimeError("Simulation application stopped during the formal trajectory")
                alpha = _smoothstep01(float(phase_frame + 1) / float(frame_count))
                target_position = previous_target + (target - previous_target) * alpha
                commanded_opening = previous_opening + (target_opening - previous_opening) * alpha

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
                sim.step(render=False)
                robot.update(sim_dt)
                cylinder.update(sim_dt)
                object_position_w, object_quat_w, object_velocity_w, object_angular_velocity_w = _object_state(cylinder)
                ee_position_w, ee_quat_w = _ee_pose_w(robot, ee_body_idx, offset_position)
                link_position_w, link_quat_w = mainline_v9._body_pose(robot, mount_body_idx)
                pad_position_w, pad_quat_w = mainline_v9._compose_child_pose(
                    link_position_w, link_quat_w, pad_position_l, pad_quat_l
                )

                # 4) same-frame PhysX pose -> UIPC mirror; 5) attachment; 6) UIPC substeps.
                substep_count = int(args_cli.uipc_substeps_per_record)
                for substep_index in range(substep_count):
                    substep_alpha = float(substep_index + 1) / float(substep_count)
                    interpolated_position = previous_mirror_position_w + substep_alpha * (
                        object_position_w - previous_mirror_position_w
                    )
                    interpolated_quat = _quat_slerp_wxyz(
                        previous_mirror_quat_w, object_quat_w, substep_alpha
                    )
                    target_matrix = _pose_delta_matrix(
                        interpolated_position,
                        interpolated_quat,
                        mirror_initial_position,
                        mirror_initial_quat,
                    )
                    write_mirror_pose(target_matrix, formal_frame)
                    attachment._compute_aim_positions()
                    manual_uipc_step()
                previous_mirror_position_w = np.asarray(
                    object_position_w, dtype=np.float64
                ).copy()
                previous_mirror_quat_w = np.asarray(object_quat_w, dtype=np.float64).copy()

                # 7-12) unique membrane -> Pad local -> frozen deformation/force/field inputs -> record.
                vertices_w = mainline_v9._uipc_vertices(membrane)
                surface_w = mainline_v9._uipc_surface(membrane)
                surface_pad_l = mainline_v9._local_from_world(
                    surface_w, pad_position_w, pad_quat_w
                )
                surface_deformation = (surface_pad_l - rest_surface_pad_l).astype(np.float32)
                actual_mirror_transform = np.asarray(
                    mirror.geo_slot_list[0].geometry().transforms().view()[0], dtype=np.float64
                )
                actual_mirror_surface_w = mainline_v9._world_from_affine_matrix(
                    initial_mirror_surface_w, actual_mirror_transform
                )
                target_mirror_surface_w = mainline_v9._world_from_affine_matrix(
                    initial_mirror_surface_w, target_matrix
                )
                mirror_surface_pad_l = mainline_v9._local_from_world(
                    actual_mirror_surface_w, pad_position_w, pad_quat_w
                )
                contact_mask, contact_diagnostics = mainline_v9._contact_geometry_diagnostics(
                    surface_pad_l,
                    rest_surface_pad_l,
                    front_surface_indices,
                    tool_min_x_l=float(np.min(mirror_surface_pad_l[:, 0])),
                    tool_center_yz_l=np.mean(mirror_surface_pad_l[:, 1:3], axis=0),
                    tool_footprint_half_extent_m=object_radius_m,
                    tool_shape="cylinder",
                )
                position_sync_error_mm = float(
                    np.linalg.norm(
                        np.mean(actual_mirror_surface_w, axis=0)
                        - np.mean(target_mirror_surface_w, axis=0)
                    )
                    * 1000.0
                )
                actual_rotation_raw = actual_mirror_transform[:3, :3]
                u_value, _, vt_value = np.linalg.svd(actual_rotation_raw)
                actual_rotation = u_value @ vt_value
                orientation_sync_error_deg = _rotation_error_deg(
                    actual_rotation, target_matrix[:3, :3]
                )
                if position_sync_error_mm > 0.05 or orientation_sync_error_deg > 0.05:
                    print(
                        f"[V6_SYNC_FAIL] frame={formal_frame + 1:04d} "
                        f"position={position_sync_error_mm:.6f}mm "
                        f"orientation={orientation_sync_error_deg:.6f}deg",
                        flush=True,
                    )
                    raise RuntimeError(
                        "Same-frame PhysX-to-UIPC mirror synchronization exceeded "
                        "0.05 mm or 0.05 deg"
                    )
                back_target_error_mm = np.linalg.norm(
                    vertices_w[back_tet_indices] - attachment.aim_positions, axis=1
                ) * 1000.0
                mainline_v9._sync_render_surface(membrane, surface_w)
                mainline_v9._sync_render_surface(mirror, actual_mirror_surface_w)

                if scene_camera is not None and scene_writer is not None:
                    sim.render()
                    scene_camera.update(sim_dt)
                    rgb = mainline_v9._to_uint8_rgb(scene_camera.data.output["rgb"][0])
                    scene_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

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
                records["object_relative_pose_to_gripper"].append(
                    _relative_pose(object_position_w, object_quat_w, ee_position_w, ee_quat_w)
                )
                records["physx_uipc_object_position_error_mm"].append(position_sync_error_mm)
                records["physx_uipc_object_orientation_error_deg"].append(
                    orientation_sync_error_deg
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

                if formal_frame % max(1, int(args_cli.log_every)) == 0:
                    print(
                        f"[V6_RECORD] frame={formal_frame + 1:04d}/{total_formal_frames} "
                        f"phase={phase_name} lift={object_lift_height_mm:+.3f}mm "
                        f"distance={object_gripper_distance_mm:.3f}mm "
                        f"contact={int(contact_diagnostics['contact_vertex_count'])} "
                        f"sync={position_sync_error_mm:.6f}mm/{orientation_sync_error_deg:.6f}deg",
                        flush=True,
                    )
                formal_frame += 1
            previous_target = target.copy()
            previous_opening = target_opening
    finally:
        if scene_writer is not None:
            scene_writer.release()

    if formal_frame != total_formal_frames:
        raise RuntimeError(
            f"Formal trajectory is incomplete: {formal_frame}/{total_formal_frames} frames"
        )
    if runtime_object_pose_write_count != 0:
        raise RuntimeError("A forbidden runtime PhysX object pose write was recorded")
    _save_record_arrays(output_dir, records)

    _finish_outputs(
        output_dir=output_dir,
        contract_dir=contract_dir,
        records=records,
        contract_verdict=contract_verdict,
        warmup_force=np.asarray(warmup_force, dtype=np.float64),
        warmup_contact=np.asarray(warmup_contact, dtype=np.int64),
        stable_object_position=stable_object_position,
        stable_object_velocity=stable_object_velocity,
        waypoints=waypoints,
        pad_root=pad_root,
        membrane_mesh_path=membrane_mesh_path,
        mount_body_name=mount_body_name,
        ee_body_name=ee_body_name,
        hidden_visual_paths=hidden_visual_paths,
        back_tet_indices=back_tet_indices,
        front_tet_indices=front_tet_indices,
        tet_thickness_m=tet_thickness_m,
        surface_thickness_m=surface_thickness_m,
        back_constrained_coverage=back_constrained_coverage,
        front_constrained_fraction=front_constrained_fraction,
        attachment=attachment,
        mirror_state=mirror_state,
        initialization_object_pose_write_count=initialization_object_pose_write_count,
        runtime_object_pose_write_count=runtime_object_pose_write_count,
        stability_steps_used=stability_steps_used,
        scene_video_path=scene_video_path,
    )


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

    phases = np.asarray(records["phase"], dtype=str)
    cycles = np.asarray(records["cycle_index"], dtype=np.int64)
    video_metrics = tactile_field_v9.render_tactile_videos(
        output_dir=output_dir,
        tactile_field=tactile_field,
        shear_magnitude=shear_magnitude,
        phases=phases,
        cycles=cycles,
        fps=float(args_cli.video_fps),
    )
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
    zero_tolerance = float(args_cli.accept_max_warmup_force_tu)

    approach_indices = np.flatnonzero(
        np.isin(phases, ("HOME", "APPROACH_PICK", "LOWER_TO_GRASP"))
    )
    close_indices = np.flatnonzero(np.isin(phases, ("CLOSE_GRIPPER", "CONFIRM_GRASP")))
    closing_motion_indices = np.flatnonzero(phases == "CLOSE_GRIPPER")
    check_indices = np.flatnonzero(phases == "CHECK_GRASP")
    hold_indices = np.flatnonzero(phases == "HOLD_LIFTED")
    support_indices = np.flatnonzero(phases == "CONFIRM_SUPPORT")
    recovery_indices = np.flatnonzero(phases == "FINAL_RECOVERY")
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

    gate_checks = {
        "gate_1_frozen_mainline_integrity": bool(
            frozen_hashes_unchanged
            and contract_verdict.get("deformation_contract_passed", False)
            and estimator_verdict.get("deformation_based_force_estimator_passed", False)
        ),
        "gate_2_architecture_and_attachment": bool(
            pad_root == "/World/envs/env_0/Robot/link8/UIPC_Pad"
            and membrane_mesh_path.endswith("UIPC_Pad/simulation/membrane_sim_mesh")
            and back_constrained_coverage == 1.0
            and front_constrained_fraction == 0.0
            and max_back_target_error <= float(args_cli.accept_max_back_target_error_mm)
        ),
        "gate_3_free_object_authenticity": bool(
            initialization_object_pose_write_count >= 1
            and runtime_object_pose_write_count == 0
        ),
        "gate_4_same_frame_physx_to_uipc_sync": bool(
            max_sync_position <= float(_v6_args.accept_max_sync_position_error_mm)
            and max_sync_orientation
            <= float(_v6_args.accept_max_sync_orientation_error_deg)
        ),
        "gate_5_formal_warmup": bool(
            warmup_force.shape[0] >= 30
            and np.max(warmup_contact, initial=0) == 0
            and np.max(warmup_force_magnitude, initial=0.0) < zero_tolerance
        ),
        "gate_6_no_early_contact": bool(
            np.all(contact_count[approach_indices] == 0)
            and np.all(active_cells[approach_indices] == 0)
            and approach_nonzero_ratio < float(args_cli.accept_max_no_contact_nonzero_ratio)
            and np.max(force_magnitude[approach_indices], initial=0.0) < zero_tolerance
        ),
        "gate_7_close_contact": bool(
            np.max(contact_count[close_indices], initial=0)
            >= int(args_cli.accept_min_contact_vertices)
            and np.max(max_front_deformation[close_indices], initial=0.0)
            >= float(args_cli.accept_min_peak_deformation_mm)
            and float(np.mean(tactile_total[close_indices, 2])) > 0.0
            and np.max(active_cells[close_indices], initial=0) > 0
            and close_fz_spearman > float(args_cli.accept_min_close_spearman)
            and np.max(max_penetration, initial=0.0)
            <= float(args_cli.accept_max_penetration_mm)
        ),
        "gate_8_real_grasp_lift": bool(
            minimum_object_lift > float(_v6_args.accept_lift_height_mm)
            and maximum_check_distance < float(_v6_args.accept_gripper_distance_mm)
            and minimum_check_vertical_velocity
            > float(_v6_args.accept_min_vertical_velocity_m_s)
            and minimum_check_support_clearance
            > float(_v6_args.support_contact_tolerance_mm)
            and runtime_object_pose_write_count == 0
        ),
        "gate_9_lifted_hold_stability": bool(
            hold_indices.size >= 30
            and hold_position_drift_mm
            <= float(_v6_args.accept_max_hold_position_drift_mm)
            and hold_orientation_drift_deg
            <= float(_v6_args.accept_max_hold_orientation_drift_deg)
            and hold_fz_cv < float(args_cli.accept_max_hold_cv)
            and hold_centroid_drift
            < float(_v6_args.accept_max_hold_field_centroid_drift_cells)
            and hold_field_correlation
            > float(_v6_args.accept_min_hold_field_correlation)
            and tangent_gate_passed
        ),
        "gate_10_supported_before_release": support_confirmed,
        "gate_11_release_and_recovery": bool(
            contact_count[-1] == 0
            and active_cells[-1] == 0
            and np.all(final_axis_ratio < float(args_cli.accept_max_release_peak_ratio))
            and final_deformation_mm <= float(args_cli.accept_max_recovery_deformation_mm)
            and abs(float(support_clearance[-1]))
            <= float(_v6_args.support_contact_tolerance_mm)
            and abs(float(object_velocity[-1, 2])) < 0.02
        ),
        "gate_12_tactile_field_conservation": bool(
            float(field_conservation["maximum_nonzero_relative_error"]) < 0.01
            and float(field_conservation["maximum_near_zero_absolute_error_tu"]) < 1.0e-6
            and float(vertex_conservation["maximum_nonzero_relative_error"]) < 1.0e-6
            and float(vertex_conservation["maximum_absolute_error_tu"]) < 1.0e-8
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
        "script_version": "v6_0_free_object_grasp_lift_tactile",
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
        "uipc_force_feedback_to_physx": False,
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
            "driver": "kinematic_uipc_affine_pose_write",
            "precomputed_mesh_vertices": 66,
            "precomputed_mesh_tetrahedra": 96,
            "runtime_full_mirror_vertex_write": True,
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
        "phase_order": [
            "HOME",
            "APPROACH_PICK",
            "LOWER_TO_GRASP",
            "CLOSE_GRIPPER",
            "CONFIRM_GRASP",
            "LIFT_OBJECT",
            "CHECK_GRASP",
            "HOLD_LIFTED",
            "LOWER_TO_PLACE",
            "CONFIRM_SUPPORT",
            "RELEASE_GRIPPER",
            "RETREAT",
            "FINAL_RECOVERY",
        ],
        "episode_index": int(_v6_args.episode_index),
        "run_uuid": RUN_UUID,
        "process_id": int(os.getpid()),
        "formal_frame_count": int(phases.size),
        "warmup_stability_steps_used": int(stability_steps_used),
        "uipc_unrecorded_prewarm_steps": int(args_cli.uipc_warmup_steps),
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
        "outputs": {
            "scene_video": str(scene_video_path),
            "tactile_video": str(output_dir / "tactile_fxyz_composite_sequence.mp4"),
        },
    }
    verdict = {
        "v6_0_single_episode_passed": single_episode_passed,
        "v6_0_full_5_episode_acceptance_passed": False,
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
    print(json.dumps(verdict, indent=2), flush=True)
    if bool(args_cli.fail_on_verdict_fail) and not single_episode_passed:
        raise RuntimeError(f"v6.0 single-episode acceptance failed: {gate_checks}")
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            error_output_dir = Path(args_cli.output_dir).expanduser().resolve()
            error_output_dir.mkdir(parents=True, exist_ok=True)
            _json_write(
                error_output_dir / "error.json",
                {
                    "script_version": "v6_0_free_object_grasp_lift_tactile",
                    "error": traceback.format_exc(),
                },
            )
        except Exception:
            pass
        simulation_app.close()
        raise
