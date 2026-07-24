# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run Franka IK-Abs cube lift and preview friction-driven COP OpenWorldTactile RGB.

This script keeps the stock Franka lift state machine and adds true PhysX
ContactSensors on both Panda fingers. It reconstructs a dense virtual tactile
force map from sparse contact points and contact forces:

* total map force equals the measured finger contact force;
* the force-map center of pressure matches the measured contact center;
* the map is non-negative and smooth over the whole virtual pad;
* the map is rendered to textured OpenWorldTactile-style RGB: normal pressure drives hue,
  and PhysX friction force drives texture displacement for SDK fx/fy.

It does not use raycast or Gaussian pressure patches.

.. code-block:: bash

    ./isaaclab.sh -p experiments/franka/current/cop_rgb_sdk_friction.py

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Pick/lift DexCube with Franka and preview friction-driven COP textured OpenWorldTactile RGB maps."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=800,
    help="Maximum environment steps before exiting; 0 or negative runs until the app is closed.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs/franka_lift_object_contact_cop_rgb_friction",
    help="Directory for raw force maps and RGB images when saving is enabled.",
)
parser.add_argument("--save_every", type=int, default=5, help="Save every N steps while grasping/lifting.")
parser.add_argument(
    "--max_saved_frames",
    type=int,
    default=0,
    help="Stop saving after this many frame pairs; 0 disables saving by default; negative saves without a frame limit.",
)
parser.add_argument(
    "--save_buffer_size",
    type=int,
    default=0,
    help="If positive, save outputs in a circular buffer with this many frame slots.",
)
parser.add_argument("--record_env_id", type=int, default=0, help="Environment index to save.")
parser.add_argument(
    "--virtual_pad_size",
    type=float,
    nargs=2,
    default=(0.038, 0.048),
    metavar=("SIZE_X", "SIZE_Z"),
    help="Virtual tactile pad grid size in meters, matching the Franka finger tactile plane.",
)
parser.add_argument(
    "--virtual_pad_resolution",
    type=float,
    default=0.0005,
    help="Virtual tactile pad grid resolution in meters.",
)
parser.add_argument("--virtual_pad_z_offset", type=float, default=0.046, help="Virtual pad Z offset in each finger frame.")
parser.add_argument(
    "--contact_force_threshold",
    type=float,
    default=0.05,
    help="Minimum total true finger-object contact force in newtons required to assign per-point force.",
)
parser.add_argument(
    "--point_force_threshold",
    type=float,
    default=0.0,
    help="Per-point force values below this threshold in newtons are set to zero after distribution.",
)
parser.add_argument(
    "--rgb_force_full_scale",
    type=float,
    default=0.0,
    help="Full-scale per-point force in newtons for OpenWorldTactile pressure; 0 uses each frame's maximum force.",
)
parser.add_argument(
    "--wavelength_high_nm",
    type=float,
    default=645.0,
    help="Low/no-force wavelength in nanometers for RGB mapping; 645 nm is pure red in this mapping.",
)
parser.add_argument(
    "--wavelength_low_nm",
    type=float,
    default=400.0,
    help="High-force wavelength in nanometers for RGB mapping.",
)
parser.add_argument("--openworldtactile_baseline_frames", type=int, default=5, help="Number of zero-force RGB frames for SDK baseline.")
parser.add_argument("--openworldtactile_fx_p1", type=float, default=1.0, help="OpenWorldTactile SDK fx scale parameter.")
parser.add_argument("--openworldtactile_fy_p1", type=float, default=1.0, help="OpenWorldTactile SDK fy scale parameter.")
parser.add_argument("--openworldtactile_fz_p1", type=float, default=1.0, help="OpenWorldTactile SDK fz scale parameter.")
parser.add_argument(
    "--openworldtactile_shear_px_per_n",
    type=float,
    default=0.45,
    help="Pixels of OpenWorldTactile texture displacement per local PhysX friction newton.",
)
parser.add_argument(
    "--openworldtactile_shear_clip_px",
    type=float,
    default=18.0,
    help="Maximum absolute OpenWorldTactile texture displacement in pixels.",
)
parser.add_argument("--openworldtactile_texture_strength", type=int, default=36, help="OpenWorldTactile value-channel texture strength.")
parser.add_argument("--openworldtactile_texture_blur", type=int, default=3, help="OpenWorldTactile value-channel texture blur kernel.")
parser.add_argument("--openworldtactile_pressure_blur", type=int, default=5, help="OpenWorldTactile pressure blur kernel.")
parser.add_argument(
    "--show_rgb_maps",
    action="store_true",
    default=True,
    help="Show a live Isaac UI preview of the left/right textured OpenWorldTactile RGB and SDK fxyz maps.",
)
parser.add_argument(
    "--hide_rgb_maps",
    action="store_false",
    dest="show_rgb_maps",
    help="Disable the live Isaac UI four-panel RGB/fxyz preview.",
)
parser.add_argument(
    "--show_force_maps",
    action="store_true",
    dest="show_rgb_maps",
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--force_preview_every",
    type=int,
    default=1,
    help="Refresh the live RGB preview every N simulation steps.",
)
parser.add_argument(
    "--force_preview_scale",
    type=int,
    default=3,
    help="Nearest-neighbor scale factor for the live RGB preview window.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import gymnasium as gym
import numpy as np
import torch
import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.math import quat_apply_inverse
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


def _find_sdk_root() -> Path:
    """Find the repository-local OpenWorldTactile SDK root."""
    for parent in Path(__file__).resolve().parents:
        sdk_root = parent / "hardware-sdk/openworldtactile"
        if (sdk_root / "api" / "isaaclab_openworldtactile_bridge.py").exists():
            return sdk_root
    raise RuntimeError("Could not find hardware-sdk/openworldtactile from this script path.")


SDK_ROOT = _find_sdk_root()
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from api import IsaacLabOpenWorldTactileBridge, OpenWorldTactileRGBRenderer  # noqa: E402

_FORCE_PREVIEW_TITLE = "Virtual Force RGB Maps"
_force_preview_available = True
_force_preview_announced = False
_force_preview_window = None


@dataclass
class ContactPadMaps:
    """Dense virtual pad fields reconstructed from one finger contact sensor."""

    force_map: np.ndarray
    displacement_x: np.ndarray
    displacement_y: np.ndarray
    total_force: float
    contact_count: int
    warning: str | None


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times in seconds before switching states."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]

    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """Simple task-space state machine to pick and lift an object."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def pad_image_shape() -> tuple[int, int]:
    """Return force-map rows and columns for the configured grid pattern."""
    size_x, size_z = args_cli.virtual_pad_size
    resolution = args_cli.virtual_pad_resolution
    cols = int(round(size_x / resolution)) + 1
    rows = int(round(size_z / resolution)) + 1
    return rows, cols


def pad_grid_coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Return local pad x/z coordinate grids in meters."""
    rows, cols = pad_image_shape()
    size_x, size_z = args_cli.virtual_pad_size
    x = np.linspace(-size_x / 2.0, size_x / 2.0, cols, dtype=np.float32)
    z = np.linspace(-size_z / 2.0, size_z / 2.0, rows, dtype=np.float32)
    return x, z


def make_finger_contact_cfg(finger_link: str) -> ContactSensorCfg:
    """Create a filtered finger-object contact sensor for one Panda finger."""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{finger_link}",
        update_period=0.0,
        history_length=5,
        debug_vis=False,
        track_contact_points=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=8,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )


def add_finger_contact_sensors(env_cfg: LiftEnvCfg):
    """Add true PhysX contact sensors for left and right Panda fingers."""
    env_cfg.scene.left_finger_contact = make_finger_contact_cfg("panda_leftfinger")
    env_cfg.scene.right_finger_contact = make_finger_contact_cfg("panda_rightfinger")


def contact_points_and_forces(contact_sensor, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite contact positions and force vectors from one finger ContactSensor."""
    contact_data = contact_sensor.data
    if contact_data.contact_pos_w is None:
        device = contact_data.net_forces_w.device
        return torch.empty((0, 3), device=device), torch.empty((0, 3), device=device)
    contact_pos_w = contact_data.contact_pos_w[env_id].reshape(-1, 3)
    if contact_data.force_matrix_w is not None:
        contact_force_w = contact_data.force_matrix_w[env_id].reshape(-1, 3)
        if contact_force_w.shape[0] != contact_pos_w.shape[0]:
            count = min(contact_force_w.shape[0], contact_pos_w.shape[0])
            contact_pos_w = contact_pos_w[:count]
            contact_force_w = contact_force_w[:count]
    else:
        finite_pos = torch.isfinite(contact_pos_w).all(dim=-1)
        valid_pos_count = int(finite_pos.sum().item())
        if valid_pos_count == 0:
            return contact_pos_w[:0], contact_pos_w[:0]
        net_force_w = fallback_net_contact_force_w(contact_sensor, env_id)
        contact_pos_w = contact_pos_w[finite_pos]
        contact_force_w = net_force_w.unsqueeze(0).repeat(valid_pos_count, 1) / float(valid_pos_count)

    finite_contact = torch.isfinite(contact_pos_w).all(dim=-1) & torch.isfinite(contact_force_w).all(dim=-1)
    force_norm = torch.linalg.norm(torch.nan_to_num(contact_force_w), dim=-1)
    valid_contact = finite_contact & (force_norm > 1.0e-9)
    return contact_pos_w[valid_contact], contact_force_w[valid_contact]


def fallback_total_contact_force(contact_sensor, env_id: int) -> float:
    """Return a contact-force magnitude even when contact positions are unavailable."""
    net_force_w = fallback_net_contact_force_w(contact_sensor, env_id)
    return float(torch.linalg.norm(net_force_w).item())


def fallback_net_contact_force_w(contact_sensor, env_id: int) -> torch.Tensor:
    """Return the finite net contact-force vector for one finger."""
    net_forces_w = torch.nan_to_num(contact_sensor.data.net_forces_w[env_id].reshape(-1, 3))
    finite_force = torch.isfinite(net_forces_w).all(dim=-1)
    if not bool(finite_force.any()):
        return torch.zeros(3, device=net_forces_w.device, dtype=net_forces_w.dtype)
    return torch.sum(net_forces_w[finite_force], dim=0)


def total_friction_force_w(contact_sensor, env_id: int) -> torch.Tensor:
    """Return the summed finite PhysX friction force for one finger in world frame."""
    friction_forces_w = contact_sensor.data.friction_forces_w
    if friction_forces_w is None:
        return torch.zeros_like(fallback_net_contact_force_w(contact_sensor, env_id))

    friction_forces_w = friction_forces_w[env_id].reshape(-1, 3)
    finite_force = torch.isfinite(friction_forces_w).all(dim=-1)
    if not bool(finite_force.any()):
        return torch.zeros(3, device=friction_forces_w.device, dtype=friction_forces_w.dtype)
    return torch.sum(torch.nan_to_num(friction_forces_w[finite_force]), dim=0)


def local_friction_force_b(robot, finger_body_id: int, contact_sensor, env_id: int) -> torch.Tensor:
    """Return the summed PhysX friction force in the finger body frame."""
    friction_force_w = total_friction_force_w(contact_sensor, env_id).unsqueeze(0)
    body_quat_w = robot.data.body_quat_w[env_id, finger_body_id].unsqueeze(0)
    return quat_apply_inverse(body_quat_w, friction_force_w).squeeze(0)


def friction_displacement_maps(
    contact_weight: np.ndarray, friction_force_b: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local pad-plane friction force into OpenWorldTactile texture displacement maps."""
    shear_x_n = float(friction_force_b[0].item())
    shear_z_n = float(friction_force_b[2].item())
    shear_clip = float(args_cli.openworldtactile_shear_clip_px)
    displacement_x_px = float(np.clip(shear_x_n * args_cli.openworldtactile_shear_px_per_n, -shear_clip, shear_clip))
    # Image y grows downward.  A positive local z friction force is rendered as upward texture motion.
    displacement_y_px = float(np.clip(-shear_z_n * args_cli.openworldtactile_shear_px_per_n, -shear_clip, shear_clip))
    displacement_x = (contact_weight * displacement_x_px).astype(np.float32)
    displacement_y = (contact_weight * displacement_y_px).astype(np.float32)
    return displacement_x, displacement_y


def axis_max_entropy_distribution(axis: np.ndarray, target: float) -> np.ndarray:
    """Return the smoothest non-negative axis distribution whose mean matches target."""
    axis = axis.astype(np.float64)
    if axis.size == 1 or float(axis[-1] - axis[0]) <= 1.0e-12:
        return np.ones_like(axis, dtype=np.float32) / float(axis.size)

    normalized_axis = axis / max(abs(float(axis[0])), abs(float(axis[-1])), 1.0e-12)
    normalized_target = float(
        np.clip(target / max(abs(float(axis[0])), abs(float(axis[-1])), 1.0e-12), normalized_axis[0], normalized_axis[-1])
    )

    if abs(normalized_target - float(normalized_axis.mean())) <= 1.0e-6:
        return np.ones(axis.shape, dtype=np.float32) / float(axis.size)

    low, high = -80.0, 80.0
    for _ in range(60):
        beta = 0.5 * (low + high)
        logits = beta * normalized_axis
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        mean = float(np.sum(probs * normalized_axis))
        if mean < normalized_target:
            low = beta
        else:
            high = beta

    beta = 0.5 * (low + high)
    logits = beta * normalized_axis
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return probs.astype(np.float32)


def contact_cop_force_map(
    robot,
    finger_body_id: int,
    contact_sensor,
    env_id: int,
) -> ContactPadMaps:
    """Reconstruct dense normal force and friction-driven texture-displacement maps."""
    rows, cols = pad_image_shape()
    zero_force_map = np.zeros((rows, cols), dtype=np.float32)
    zero_displacement = np.zeros((rows, cols), dtype=np.float32)

    contact_pos_w, contact_force_w = contact_points_and_forces(contact_sensor, env_id)
    contact_count = int(contact_pos_w.shape[0])
    if contact_count == 0:
        total_force = fallback_total_contact_force(contact_sensor, env_id)
        if total_force < args_cli.contact_force_threshold:
            return ContactPadMaps(zero_force_map, zero_displacement, zero_displacement, total_force, 0, None)
        force_map = np.full((rows, cols), total_force / float(rows * cols), dtype=np.float32)
        contact_weight = np.ones((rows, cols), dtype=np.float32)
        friction_force_b = local_friction_force_b(robot, finger_body_id, contact_sensor, env_id)
        displacement_x, displacement_y = friction_displacement_maps(contact_weight, friction_force_b)
        return ContactPadMaps(
            force_map,
            displacement_x,
            displacement_y,
            total_force,
            0,
            "true contact force exists but contact positions are unavailable; using centered uniform reconstruction",
        )

    force_norm = torch.linalg.norm(contact_force_w, dim=-1)
    total_force = float(force_norm.sum().item())
    if total_force < args_cli.contact_force_threshold:
        return ContactPadMaps(zero_force_map, zero_displacement, zero_displacement, total_force, contact_count, None)

    body_pos_w = robot.data.body_pos_w[env_id, finger_body_id].unsqueeze(0)
    body_quat_w = robot.data.body_quat_w[env_id, finger_body_id].unsqueeze(0).repeat(contact_count, 1)
    contact_pos_b = quat_apply_inverse(body_quat_w, contact_pos_w - body_pos_w)
    contact_x = contact_pos_b[:, 0]
    contact_z = contact_pos_b[:, 2] - float(args_cli.virtual_pad_z_offset)
    force_weights = force_norm / torch.clamp(force_norm.sum(), min=1.0e-9)
    cop_x = float(torch.sum(contact_x * force_weights).item())
    cop_z = float(torch.sum(contact_z * force_weights).item())

    size_x, size_z = args_cli.virtual_pad_size
    clipped_cop_x = float(np.clip(cop_x, -size_x / 2.0, size_x / 2.0))
    clipped_cop_z = float(np.clip(cop_z, -size_z / 2.0, size_z / 2.0))
    warning = None
    if clipped_cop_x != cop_x or clipped_cop_z != cop_z:
        warning = (
            "contact center lies outside virtual pad; clipped to pad bounds "
            f"cop=({cop_x:.5f}, {cop_z:.5f}) clipped=({clipped_cop_x:.5f}, {clipped_cop_z:.5f})"
        )

    x_axis, z_axis = pad_grid_coordinates()
    x_probs = axis_max_entropy_distribution(x_axis, clipped_cop_x)
    z_probs = axis_max_entropy_distribution(z_axis, clipped_cop_z)
    distribution = np.outer(z_probs, x_probs).astype(np.float32)
    distribution_sum = float(np.sum(distribution))
    if distribution_sum <= 0.0:
        return ContactPadMaps(
            zero_force_map,
            zero_displacement,
            zero_displacement,
            total_force,
            contact_count,
            "COP reconstruction distribution is empty",
        )

    force_map = (total_force * distribution / distribution_sum).astype(np.float32)

    if args_cli.point_force_threshold > 0.0:
        force_map = np.where(force_map >= args_cli.point_force_threshold, force_map, 0.0).astype(np.float32)

    contact_weight = distribution / max(float(np.max(distribution)), 1.0e-9)
    friction_force_b = local_friction_force_b(robot, finger_body_id, contact_sensor, env_id)
    displacement_x, displacement_y = friction_displacement_maps(contact_weight, friction_force_b)

    return ContactPadMaps(force_map, displacement_x, displacement_y, total_force, contact_count, warning)


def save_force_map(path: Path, force_map: np.ndarray):
    """Save one raw per-point force map in newtons."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, force_map.astype(np.float32))


def wavelength_to_rgb(wavelength_nm: np.ndarray) -> np.ndarray:
    """Convert visible-light wavelength values in nanometers to RGB."""
    wl = np.asarray(wavelength_nm, dtype=np.float32)
    rgb = np.zeros((*wl.shape, 3), dtype=np.float32)

    mask = (wl >= 380.0) & (wl < 440.0)
    rgb[..., 0] = np.where(mask, -(wl - 440.0) / 60.0, rgb[..., 0])
    rgb[..., 2] = np.where(mask, 1.0, rgb[..., 2])

    mask = (wl >= 440.0) & (wl < 490.0)
    rgb[..., 1] = np.where(mask, (wl - 440.0) / 50.0, rgb[..., 1])
    rgb[..., 2] = np.where(mask, 1.0, rgb[..., 2])

    mask = (wl >= 490.0) & (wl < 510.0)
    rgb[..., 1] = np.where(mask, 1.0, rgb[..., 1])
    rgb[..., 2] = np.where(mask, -(wl - 510.0) / 20.0, rgb[..., 2])

    mask = (wl >= 510.0) & (wl < 580.0)
    rgb[..., 0] = np.where(mask, (wl - 510.0) / 70.0, rgb[..., 0])
    rgb[..., 1] = np.where(mask, 1.0, rgb[..., 1])

    mask = (wl >= 580.0) & (wl < 645.0)
    rgb[..., 0] = np.where(mask, 1.0, rgb[..., 0])
    rgb[..., 1] = np.where(mask, -(wl - 645.0) / 65.0, rgb[..., 1])

    mask = (wl >= 645.0) & (wl <= 780.0)
    rgb[..., 0] = np.where(mask, 1.0, rgb[..., 0])

    factor = np.ones_like(wl, dtype=np.float32)
    factor = np.where((wl >= 380.0) & (wl < 420.0), 0.3 + 0.7 * (wl - 380.0) / 40.0, factor)
    factor = np.where((wl > 700.0) & (wl <= 780.0), 0.3 + 0.7 * (780.0 - wl) / 80.0, factor)
    rgb *= np.clip(factor, 0.0, 1.0)[..., None]

    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def force_map_rgb_scale(force_map: np.ndarray) -> float:
    """Return the per-point force scale used by OpenWorldTactile pressure rendering."""
    if args_cli.rgb_force_full_scale > 0.0:
        return float(args_cli.rgb_force_full_scale)
    finite_force = np.nan_to_num(force_map, nan=0.0, posinf=0.0, neginf=0.0)
    max_force = float(np.max(np.clip(finite_force, 0.0, None))) if finite_force.size else 0.0
    return max(max_force, 1.0e-9)


def force_map_to_wavelength_rgb(force_map: np.ndarray) -> np.ndarray:
    """Map one force map to wavelength RGB from low-force red to high-force blue/purple."""
    scale = force_map_rgb_scale(force_map)
    force = np.nan_to_num(force_map, nan=0.0, posinf=scale, neginf=0.0)
    pressure = np.clip(np.clip(force, 0.0, None) / scale, 0.0, 1.0)
    wavelength = args_cli.wavelength_high_nm - pressure * (args_cli.wavelength_high_nm - args_cli.wavelength_low_nm)
    return wavelength_to_rgb(wavelength)


def force_map_to_gray_image(force_map: np.ndarray) -> np.ndarray:
    """Map one force map to an 8-bit grayscale image using the same scale as the RGB renderer."""
    scale = force_map_rgb_scale(force_map)
    force = np.nan_to_num(force_map, nan=0.0, posinf=scale, neginf=0.0)
    return (np.clip(np.clip(force, 0.0, None) / scale, 0.0, 1.0) * 255.0).astype(np.uint8)


def force_map_to_openworldtactile_pressure(force_map: np.ndarray) -> np.ndarray:
    """Normalize one dense force map into the pressure field consumed by OpenWorldTactileRGBRenderer."""
    scale = force_map_rgb_scale(force_map)
    force = np.nan_to_num(force_map, nan=0.0, posinf=scale, neginf=0.0)
    return np.clip(np.clip(force, 0.0, None) / scale, 0.0, 1.0).astype(np.float32)


def make_openworldtactile_renderer() -> OpenWorldTactileRGBRenderer:
    """Create one textured OpenWorldTactile renderer for the virtual pad grid."""
    rows, cols = pad_image_shape()
    return OpenWorldTactileRGBRenderer(
        height=rows,
        width=cols,
        max_pressure=1.0,
        base_hue=0.0,
        max_hue=120.0,
        saturation=255,
        base_value=220,
        texture_strength=args_cli.openworldtactile_texture_strength,
        texture_blur=args_cli.openworldtactile_texture_blur,
        pressure_blur=args_cli.openworldtactile_pressure_blur,
        displacement_scale=1.0,
    )


def render_openworldtactile_rgb(pad_maps: ContactPadMaps, renderer: OpenWorldTactileRGBRenderer) -> np.ndarray:
    """Render one virtual pad with OpenWorldTactile pressure color and friction-driven texture displacement."""
    return renderer.render(
        force_map_to_openworldtactile_pressure(pad_maps.force_map),
        displacement_x=pad_maps.displacement_x,
        displacement_y=pad_maps.displacement_y,
    )


def make_openworldtactile_bridge(renderer: OpenWorldTactileRGBRenderer | None = None) -> IsaacLabOpenWorldTactileBridge:
    """Create one SDK bridge for one virtual finger preview stream."""
    return IsaacLabOpenWorldTactileBridge(
        fx_p1=args_cli.openworldtactile_fx_p1,
        fy_p1=args_cli.openworldtactile_fy_p1,
        fz_p1=args_cli.openworldtactile_fz_p1,
        baseline_frames=args_cli.openworldtactile_baseline_frames,
        env_id=0,
        renderer=renderer,
        save_input_rgb=False,
        record_buffer_size=args_cli.save_buffer_size if args_cli.save_buffer_size > 0 else None,
    )


def calibrate_openworldtactile_bridge_baseline(bridge: IsaacLabOpenWorldTactileBridge, renderer: OpenWorldTactileRGBRenderer):
    """Calibrate the SDK bridge with the textured no-contact frame from the same renderer."""
    zero_rgb = renderer.render_baseline()
    for _ in range(args_cli.openworldtactile_baseline_frames):
        bridge.update(zero_rgb)


def openworldtactile_fxyz_visual_image(
    bridge: IsaacLabOpenWorldTactileBridge | None,
    _force_xyz: tuple[float, float, float] | None,
) -> np.ndarray:
    """Return one SDK-style fxyz visualization image for a live preview panel."""
    rows, cols = pad_image_shape()
    image_bgr = np.zeros((rows, cols, 3), dtype=np.uint8)
    if bridge is None:
        return image_bgr[..., ::-1]

    if bridge.calibrated:
        fx_matrix = fy_matrix = fz_matrix = None
        try:
            fx_matrix, fy_matrix, fz_matrix = bridge.sensor.get_calib_matrix()
        except Exception:
            fx_matrix = fy_matrix = fz_matrix = None

        if fz_matrix is None:
            fz_matrix = bridge.sensor.get_hue_matrix()
        if fz_matrix is not None:
            fz = np.nan_to_num(np.asarray(fz_matrix, dtype=np.float32), nan=0.0, posinf=255.0, neginf=0.0)
            if fz.shape != (rows, cols):
                fz = cv2.resize(fz, (cols, rows), interpolation=cv2.INTER_LINEAR)
            gray = np.uint8(np.clip(fz, 0.0, 255.0))
            image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if fx_matrix is not None and fy_matrix is not None:
            fx = np.nan_to_num(np.asarray(fx_matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            fy = np.nan_to_num(np.asarray(fy_matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if fx.shape != (rows, cols):
                fx = cv2.resize(fx, (cols, rows), interpolation=cv2.INTER_LINEAR)
            if fy.shape != (rows, cols):
                fy = cv2.resize(fy, (cols, rows), interpolation=cv2.INTER_LINEAR)
            arrow_step = 12
            for y in range(0, rows, arrow_step):
                for x in range(0, cols, arrow_step):
                    end_x = int(np.clip(round(x + float(fx[y, x]) * 8.0), 0, cols - 1))
                    end_y = int(np.clip(round(y + float(fy[y, x]) * 8.0), 0, rows - 1))
                    cv2.arrowedLine(image_bgr, (x, y), (end_x, end_y), (255, 255, 255), 1, tipLength=0.35)

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, rgb: np.ndarray):
    """Save one RGB image through OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_gray(path: Path, gray: np.ndarray):
    """Save one grayscale image through OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), gray)


def compute_force_map_pair(env, left_body_id: int, right_body_id: int):
    """Compute left and right COP-reconstructed virtual pad force maps for the selected environment."""
    scene = env.unwrapped.scene
    robot = scene["robot"]
    left_result = contact_cop_force_map(
        robot, left_body_id, scene["left_finger_contact"], args_cli.record_env_id
    )
    right_result = contact_cop_force_map(
        robot, right_body_id, scene["right_finger_contact"], args_cli.record_env_id
    )
    return left_result, right_result


class ForceRgbPreviewWindow:
    """In-app Isaac UI window for live RGB and SDK fxyz previews."""

    def __init__(
        self,
        rows: int,
        cols: int,
        scale: int,
        left_zero_rgb: np.ndarray | None = None,
        right_zero_rgb: np.ndarray | None = None,
    ):
        import omni.ui as ui

        self._ui = ui
        self._rows = rows
        self._cols = cols
        self._scale = max(1, int(scale))
        self._image_width = self._cols * self._scale
        self._image_height = self._rows * self._scale
        window_width = self._image_width * 2 + 72
        window_height = self._image_height * 2 + 144

        self._left_rgb_provider = ui.ByteImageProvider()
        self._right_rgb_provider = ui.ByteImageProvider()
        self._left_fxyz_provider = ui.ByteImageProvider()
        self._right_fxyz_provider = ui.ByteImageProvider()
        zero_force_map = np.zeros((rows, cols), dtype=np.float32)
        zero_rgb = force_map_to_wavelength_rgb(zero_force_map)
        left_zero_rgb = zero_rgb if left_zero_rgb is None else left_zero_rgb
        right_zero_rgb = zero_rgb if right_zero_rgb is None else right_zero_rgb
        zero_fxyz = np.zeros((rows, cols, 3), dtype=np.uint8)
        self._update_provider(self._left_rgb_provider, left_zero_rgb)
        self._update_provider(self._right_rgb_provider, right_zero_rgb)
        self._update_provider(self._left_fxyz_provider, zero_fxyz)
        self._update_provider(self._right_fxyz_provider, zero_fxyz)

        self._window = ui.Window(
            _FORCE_PREVIEW_TITLE,
            width=window_width,
            height=window_height,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                with ui.HStack(height=22, spacing=8):
                    ui.Label("LEFT RGB", width=self._image_width, alignment=ui.Alignment.CENTER)
                    ui.Spacer(width=8)
                    ui.Label("RIGHT RGB", width=self._image_width, alignment=ui.Alignment.CENTER)
                with ui.HStack(height=self._image_height, spacing=8):
                    with ui.Frame(width=self._image_width, height=self._image_height):
                        ui.ImageWithProvider(self._left_rgb_provider)
                    ui.Spacer(width=8)
                    with ui.Frame(width=self._image_width, height=self._image_height):
                        ui.ImageWithProvider(self._right_rgb_provider)
                with ui.HStack(height=22, spacing=8):
                    ui.Label("LEFT FXYZ", width=self._image_width, alignment=ui.Alignment.CENTER)
                    ui.Spacer(width=8)
                    ui.Label("RIGHT FXYZ", width=self._image_width, alignment=ui.Alignment.CENTER)
                with ui.HStack(height=self._image_height, spacing=8):
                    with ui.Frame(width=self._image_width, height=self._image_height):
                        ui.ImageWithProvider(self._left_fxyz_provider)
                    ui.Spacer(width=8)
                    with ui.Frame(width=self._image_width, height=self._image_height):
                        ui.ImageWithProvider(self._right_fxyz_provider)

        workspace_window = ui.Workspace.get_window(_FORCE_PREVIEW_TITLE)
        if workspace_window is not None:
            workspace_window.focus()

    @staticmethod
    def _to_rgba(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        if image.ndim == 2:
            rgba[..., 0] = image
            rgba[..., 1] = image
            rgba[..., 2] = image
        else:
            rgba[..., :3] = image[..., :3]
        rgba[..., 3] = 255
        return rgba

    def _update_provider(self, provider, image: np.ndarray):
        rgba = np.ascontiguousarray(self._to_rgba(image))
        provider.set_bytes_data(rgba.flatten().data, [rgba.shape[1], rgba.shape[0]])

    def update(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        left_fxyz: np.ndarray,
        right_fxyz: np.ndarray,
    ):
        self._update_provider(self._left_rgb_provider, left_rgb)
        self._update_provider(self._right_rgb_provider, right_rgb)
        self._update_provider(self._left_fxyz_provider, left_fxyz)
        self._update_provider(self._right_fxyz_provider, right_fxyz)

    def destroy(self):
        if self._window is not None:
            self._window.visible = False
            self._window.destroy()
            self._window = None


def show_rgb_pair(
    left_pad_maps: ContactPadMaps,
    right_pad_maps: ContactPadMaps,
    left_openworldtactile_renderer: OpenWorldTactileRGBRenderer,
    right_openworldtactile_renderer: OpenWorldTactileRGBRenderer,
    left_openworldtactile_bridge: IsaacLabOpenWorldTactileBridge | None,
    right_openworldtactile_bridge: IsaacLabOpenWorldTactileBridge | None,
):
    """Show a live preview of RGB and SDK fxyz maps for both virtual pads."""
    global _force_preview_announced, _force_preview_available, _force_preview_window

    if not args_cli.show_rgb_maps or not _force_preview_available:
        return

    try:
        left_rgb = render_openworldtactile_rgb(left_pad_maps, left_openworldtactile_renderer)
        right_rgb = render_openworldtactile_rgb(right_pad_maps, right_openworldtactile_renderer)
        left_force_xyz = left_openworldtactile_bridge.update(left_rgb) if left_openworldtactile_bridge is not None else None
        right_force_xyz = right_openworldtactile_bridge.update(right_rgb) if right_openworldtactile_bridge is not None else None

        if _force_preview_window is None:
            rows, cols = pad_image_shape()
            _force_preview_window = ForceRgbPreviewWindow(
                rows,
                cols,
                args_cli.force_preview_scale,
                left_zero_rgb=left_openworldtactile_renderer.render_baseline(),
                right_zero_rgb=right_openworldtactile_renderer.render_baseline(),
            )
        if not _force_preview_announced:
            print(f"[INFO]: Force RGB preview panel opened in Isaac UI: {_FORCE_PREVIEW_TITLE}.")
            _force_preview_announced = True
        _force_preview_window.update(
            left_rgb,
            right_rgb,
            openworldtactile_fxyz_visual_image(left_openworldtactile_bridge, left_force_xyz),
            openworldtactile_fxyz_visual_image(right_openworldtactile_bridge, right_force_xyz),
        )
    except Exception as exc:
        print(f"[WARN]: Force RGB preview disabled: {exc}")
        _force_preview_available = False


def destroy_force_preview_window():
    """Destroy the in-app RGB preview window if it was created."""
    global _force_preview_window

    if _force_preview_window is not None:
        _force_preview_window.destroy()
        _force_preview_window = None


def maybe_save_rgb_pair(
    env,
    step_count: int,
    saved_count: int,
    pick_sm: PickAndLiftSm,
    output_dir: Path,
    left_body_id: int,
    right_body_id: int,
    left_openworldtactile_renderer: OpenWorldTactileRGBRenderer,
    right_openworldtactile_renderer: OpenWorldTactileRGBRenderer,
) -> int:
    """Save COP force maps and textured OpenWorldTactile RGB images during grasp and lift states."""
    if args_cli.max_saved_frames == 0:
        return saved_count
    if args_cli.save_buffer_size <= 0 and args_cli.max_saved_frames > 0 and saved_count >= args_cli.max_saved_frames:
        return saved_count
    if step_count % args_cli.save_every != 0:
        return saved_count
    if pick_sm.sm_state[args_cli.record_env_id].item() < 3:
        return saved_count

    left_result, right_result = compute_force_map_pair(env, left_body_id, right_body_id)

    frame_slot = saved_count % args_cli.save_buffer_size if args_cli.save_buffer_size > 0 else saved_count
    frame_id = f"{frame_slot:05d}"
    save_force_map(output_dir / "left_force_map" / f"{frame_id}.npy", left_result.force_map)
    save_force_map(output_dir / "right_force_map" / f"{frame_id}.npy", right_result.force_map)
    left_rgb = render_openworldtactile_rgb(left_result, left_openworldtactile_renderer)
    right_rgb = render_openworldtactile_rgb(right_result, right_openworldtactile_renderer)
    left_gray = force_map_to_gray_image(left_result.force_map)
    right_gray = force_map_to_gray_image(right_result.force_map)
    save_rgb(output_dir / "left_rgb" / f"{frame_id}.png", left_rgb)
    save_rgb(output_dir / "right_rgb" / f"{frame_id}.png", right_rgb)
    save_gray(output_dir / "left_gray" / f"{frame_id}.png", left_gray)
    save_gray(output_dir / "right_gray" / f"{frame_id}.png", right_gray)

    left_sum = float(np.sum(left_result.force_map))
    right_sum = float(np.sum(right_result.force_map))
    print(
        f"[COP_RGB] frame={frame_id} "
        f"left_total={left_result.total_force:.6f}N left_sum={left_sum:.6f}N "
        f"left_rgb_scale={force_map_rgb_scale(left_result.force_map):.6f}N "
        f"left_points={left_result.contact_count} left_nonzero={int(np.count_nonzero(left_result.force_map))} "
        f"right_total={right_result.total_force:.6f}N right_sum={right_sum:.6f}N "
        f"right_rgb_scale={force_map_rgb_scale(right_result.force_map):.6f}N "
        f"right_points={right_result.contact_count} right_nonzero={int(np.count_nonzero(right_result.force_map))}"
    )
    if left_result.warning is not None:
        print(f"[WARN]: left frame={frame_id}: {left_result.warning}.")
    if right_result.warning is not None:
        print(f"[WARN]: right frame={frame_id}: {right_result.warning}.")

    return saved_count + 1


def main():
    if args_cli.save_every <= 0:
        raise ValueError("--save_every must be positive.")
    if args_cli.record_env_id < 0 or args_cli.record_env_id >= args_cli.num_envs:
        raise ValueError("--record_env_id must be in [0, num_envs).")
    if args_cli.virtual_pad_resolution <= 0.0:
        raise ValueError("--virtual_pad_resolution must be positive.")
    if args_cli.save_buffer_size < 0:
        raise ValueError("--save_buffer_size must be non-negative.")
    if args_cli.contact_force_threshold < 0.0:
        raise ValueError("--contact_force_threshold must be non-negative.")
    if args_cli.point_force_threshold < 0.0:
        raise ValueError("--point_force_threshold must be non-negative.")
    if args_cli.rgb_force_full_scale < 0.0:
        raise ValueError("--rgb_force_full_scale must be non-negative.")
    if not 380.0 <= args_cli.wavelength_low_nm <= 780.0:
        raise ValueError("--wavelength_low_nm must be in the visible spectrum range [380, 780].")
    if not 380.0 <= args_cli.wavelength_high_nm <= 780.0:
        raise ValueError("--wavelength_high_nm must be in the visible spectrum range [380, 780].")
    if args_cli.openworldtactile_baseline_frames <= 0:
        raise ValueError("--openworldtactile_baseline_frames must be positive.")
    if args_cli.openworldtactile_shear_px_per_n < 0.0:
        raise ValueError("--openworldtactile_shear_px_per_n must be non-negative.")
    if args_cli.openworldtactile_shear_clip_px <= 0.0:
        raise ValueError("--openworldtactile_shear_clip_px must be positive.")
    if args_cli.openworldtactile_texture_strength < 0:
        raise ValueError("--openworldtactile_texture_strength must be non-negative.")
    if args_cli.openworldtactile_texture_blur <= 0:
        raise ValueError("--openworldtactile_texture_blur must be positive.")
    if args_cli.openworldtactile_pressure_blur <= 0:
        raise ValueError("--openworldtactile_pressure_blur must be positive.")
    if args_cli.force_preview_every <= 0:
        raise ValueError("--force_preview_every must be positive.")
    if args_cli.force_preview_scale <= 0:
        raise ValueError("--force_preview_scale must be positive.")

    rows, cols = pad_image_shape()
    if rows <= 0 or cols <= 0:
        raise ValueError("--virtual_pad_size values must be positive.")

    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    add_finger_contact_sensors(env_cfg)

    env_step_dt = env_cfg.sim.dt * env_cfg.decimation
    if args_cli.max_steps > 0:
        # Keep the scripted run from timing out before the requested capture window.
        env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.max_steps * env_step_dt + 1.0)

    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    left_body_id = robot.find_bodies("panda_leftfinger")[0][0]
    right_body_id = robot.find_bodies("panda_rightfinger")[0][0]

    saving_enabled = args_cli.max_saved_frames != 0
    output_dir = Path(args_cli.output_dir)
    if saving_enabled:
        output_dir.mkdir(parents=True, exist_ok=True)

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0

    pick_sm = PickAndLiftSm(env_step_dt, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.01)
    saved_count = 0
    left_openworldtactile_renderer = make_openworldtactile_renderer()
    right_openworldtactile_renderer = make_openworldtactile_renderer()
    left_openworldtactile_bridge = None
    right_openworldtactile_bridge = None
    if args_cli.show_rgb_maps:
        left_openworldtactile_bridge = make_openworldtactile_bridge(left_openworldtactile_renderer)
        right_openworldtactile_bridge = make_openworldtactile_bridge(right_openworldtactile_renderer)
        calibrate_openworldtactile_bridge_baseline(left_openworldtactile_bridge, left_openworldtactile_renderer)
        calibrate_openworldtactile_bridge_baseline(right_openworldtactile_bridge, right_openworldtactile_renderer)

    if saving_enabled:
        print(f"[INFO]: Saving COP force maps, textured OpenWorldTactile RGB images, and grayscale images under: {output_dir.resolve()}")
    else:
        print("[INFO]: Local COP RGB saving disabled by default. Use --max_saved_frames N to save frames.")
    print(
        "[INFO]: Virtual pad grid "
        f"rows={rows} cols={cols} size={tuple(args_cli.virtual_pad_size)} "
        f"resolution={args_cli.virtual_pad_resolution} z_offset={args_cli.virtual_pad_z_offset}."
    )
    print(
        "[INFO]: Force map model: ContactSensor COP maximum-entropy reconstruction "
        "with PhysX friction-driven texture displacement. "
        f"contact_force_threshold={args_cli.contact_force_threshold} "
        f"point_force_threshold={args_cli.point_force_threshold} "
        f"rgb_force_full_scale={args_cli.rgb_force_full_scale}."
    )
    print(
        "[INFO]: OpenWorldTactile textured RGB renderer "
        f"texture_strength={args_cli.openworldtactile_texture_strength} "
        f"texture_blur={args_cli.openworldtactile_texture_blur} "
        f"pressure_blur={args_cli.openworldtactile_pressure_blur} "
        f"friction_px_per_n={args_cli.openworldtactile_shear_px_per_n} "
        f"shear_clip_px={args_cli.openworldtactile_shear_clip_px}."
    )
    if args_cli.show_rgb_maps:
        print(
            "[INFO]: Live COP OpenWorldTactile textured RGB + OpenWorldTactile SDK fxyz preview enabled in Isaac UI "
            f"(every {args_cli.force_preview_every} step(s), scale={args_cli.force_preview_scale})."
        )
        print(
            "[INFO]: OpenWorldTactile SDK baseline calibrated from zero-force RGB frames "
            f"(frames={args_cli.openworldtactile_baseline_frames}, "
            f"fx_p1={args_cli.openworldtactile_fx_p1}, fy_p1={args_cli.openworldtactile_fy_p1}, fz_p1={args_cli.openworldtactile_fz_p1})."
        )

    try:
        step_count = 0
        while simulation_app.is_running() and (args_cli.max_steps <= 0 or step_count < args_cli.max_steps):
            with torch.inference_mode():
                dones = env.step(actions)[-2]

                ee_frame_sensor = env.unwrapped.scene["ee_frame"]
                tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
                tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

                object_data: RigidObjectData = env.unwrapped.scene["object"].data
                object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins

                desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

                actions = pick_sm.compute(
                    torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                    torch.cat([object_position, desired_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),
                )

                if args_cli.show_rgb_maps and step_count % args_cli.force_preview_every == 0:
                    left_result, right_result = compute_force_map_pair(env, left_body_id, right_body_id)
                    show_rgb_pair(
                        left_result,
                        right_result,
                        left_openworldtactile_renderer,
                        right_openworldtactile_renderer,
                        left_openworldtactile_bridge,
                        right_openworldtactile_bridge,
                    )

                saved_count = maybe_save_rgb_pair(
                    env,
                    step_count,
                    saved_count,
                    pick_sm,
                    output_dir,
                    left_body_id,
                    right_body_id,
                    left_openworldtactile_renderer,
                    right_openworldtactile_renderer,
                )
                if args_cli.save_buffer_size <= 0 and args_cli.max_saved_frames > 0 and saved_count >= args_cli.max_saved_frames:
                    break

                if dones.any():
                    reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    pick_sm.reset_idx(reset_ids)

            step_count += 1

        if saving_enabled:
            print(f"[INFO]: Saved {saved_count} COP textured OpenWorldTactile RGB + grayscale frame pairs.")
        else:
            print("[INFO]: Finished without saving local COP RGB files.")
    finally:
        env.close()
        if left_openworldtactile_bridge is not None:
            left_openworldtactile_bridge.close()
        if right_openworldtactile_bridge is not None:
            right_openworldtactile_bridge.close()
        if args_cli.show_rgb_maps:
            destroy_force_preview_window()


if __name__ == "__main__":
    main()
    simulation_app.close()
