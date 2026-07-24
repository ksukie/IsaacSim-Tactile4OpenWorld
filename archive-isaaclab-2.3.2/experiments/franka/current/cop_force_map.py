# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run Franka IK-Abs cube lift and preview ContactSensor-only COP force maps.

This script keeps the stock Franka lift state machine and adds true PhysX
ContactSensors on both Panda fingers. It reconstructs a dense virtual tactile
force map from sparse contact points and contact forces:

* total map force equals the measured finger contact force;
* the force-map center of pressure matches the measured contact center;
* the map is non-negative and smooth over the whole virtual pad.

It does not use raycast, Gaussian pressure patches, OpenWorldTactile RGB, or the OpenWorldTactile SDK.

.. code-block:: bash

    ./isaaclab.sh -p experiments/franka/current/cop_force_map.py

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick/lift DexCube with Franka and preview COP-reconstructed force maps.")
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
    default="outputs/franka_lift_object_contact_cop_force_map",
    help="Directory for raw force maps and debug images when saving is enabled.",
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
    "--force_debug_full_scale",
    type=float,
    default=0.0,
    help="Full-scale force in newtons for debug PNGs; 0 uses each frame's maximum force.",
)
parser.add_argument(
    "--show_force_maps",
    action="store_true",
    default=False,
    help="Show a live Isaac UI preview of the left and right virtual force maps while the simulation runs.",
)
parser.add_argument(
    "--force_preview_every",
    type=int,
    default=1,
    help="Refresh the live force-map preview every N simulation steps.",
)
parser.add_argument(
    "--force_preview_scale",
    type=int,
    default=4,
    help="Nearest-neighbor scale factor for the live force-map preview window.",
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
from pathlib import Path

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

_FORCE_PREVIEW_TITLE = "Virtual Force Maps"
_force_preview_available = True
_force_preview_announced = False
_force_preview_window = None


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
        total_force = fallback_total_contact_force(contact_sensor, env_id)
        contact_pos_w = contact_pos_w[finite_pos]
        contact_force_w = torch.zeros_like(contact_pos_w)
        contact_force_w[:, 0] = float(total_force) / float(valid_pos_count)

    finite_contact = torch.isfinite(contact_pos_w).all(dim=-1) & torch.isfinite(contact_force_w).all(dim=-1)
    force_norm = torch.linalg.norm(torch.nan_to_num(contact_force_w), dim=-1)
    valid_contact = finite_contact & (force_norm > 1.0e-9)
    return contact_pos_w[valid_contact], contact_force_w[valid_contact]


def fallback_total_contact_force(contact_sensor, env_id: int) -> float:
    """Return a contact-force magnitude even when contact positions are unavailable."""
    net_forces_w = torch.nan_to_num(contact_sensor.data.net_forces_w[env_id].reshape(-1, 3))
    finite_force = torch.isfinite(net_forces_w).all(dim=-1)
    if not bool(finite_force.any()):
        return 0.0
    return float(torch.linalg.norm(net_forces_w[finite_force], dim=-1).sum().item())


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
) -> tuple[np.ndarray, float, int, str | None]:
    """Reconstruct a dense force map from true contact force and center of pressure."""
    rows, cols = pad_image_shape()
    zero_force_map = np.zeros((rows, cols), dtype=np.float32)

    contact_pos_w, contact_force_w = contact_points_and_forces(contact_sensor, env_id)
    contact_count = int(contact_pos_w.shape[0])
    if contact_count == 0:
        total_force = fallback_total_contact_force(contact_sensor, env_id)
        if total_force < args_cli.contact_force_threshold:
            return zero_force_map, total_force, 0, None
        force_map = np.full((rows, cols), total_force / float(rows * cols), dtype=np.float32)
        return force_map, total_force, 0, "true contact force exists but contact positions are unavailable; using centered uniform reconstruction"

    force_norm = torch.linalg.norm(contact_force_w, dim=-1)
    total_force = float(force_norm.sum().item())
    if total_force < args_cli.contact_force_threshold:
        return zero_force_map, total_force, contact_count, None

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
        return zero_force_map, total_force, contact_count, "COP reconstruction distribution is empty"

    force_map = (total_force * distribution / distribution_sum).astype(np.float32)

    if args_cli.point_force_threshold > 0.0:
        force_map = np.where(force_map >= args_cli.point_force_threshold, force_map, 0.0).astype(np.float32)

    return force_map, total_force, contact_count, warning


def save_force_map(path: Path, force_map: np.ndarray):
    """Save one raw per-point force map in newtons."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, force_map.astype(np.float32))


def force_map_to_debug_image(force_map: np.ndarray) -> np.ndarray:
    """Convert one raw force map to the same 8-bit grayscale image used for debug PNGs."""
    if args_cli.force_debug_full_scale > 0.0:
        scale = float(args_cli.force_debug_full_scale)
    else:
        scale = float(np.max(force_map)) if force_map.size else 0.0

    if scale <= 0.0:
        return np.zeros(force_map.shape, dtype=np.uint8)

    return (np.clip(force_map / scale, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_force_debug(path: Path, force_map: np.ndarray):
    """Save one force map as an 8-bit grayscale debug image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), force_map_to_debug_image(force_map))


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


class ForceMapPreviewWindow:
    """In-app Isaac UI window for live left/right virtual force-map previews."""

    def __init__(self, rows: int, cols: int, scale: int):
        import omni.ui as ui

        self._ui = ui
        self._rows = rows
        self._cols = cols
        self._scale = max(1, int(scale))
        self._image_width = self._cols * self._scale
        self._image_height = self._rows * self._scale
        window_width = self._image_width * 2 + 72
        window_height = self._image_height + 96

        self._left_provider = ui.ByteImageProvider()
        self._right_provider = ui.ByteImageProvider()
        self._update_provider(self._left_provider, np.zeros((rows, cols), dtype=np.uint8))
        self._update_provider(self._right_provider, np.zeros((rows, cols), dtype=np.uint8))

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
                    ui.Label("LEFT", width=self._image_width, alignment=ui.Alignment.CENTER)
                    ui.Spacer(width=8)
                    ui.Label("RIGHT", width=self._image_width, alignment=ui.Alignment.CENTER)
                with ui.HStack(height=self._image_height, spacing=8):
                    with ui.Frame(width=self._image_width, height=self._image_height):
                        ui.ImageWithProvider(self._left_provider)
                    ui.Spacer(width=8)
                    with ui.Frame(width=self._image_width, height=self._image_height):
                        ui.ImageWithProvider(self._right_provider)

        workspace_window = ui.Workspace.get_window(_FORCE_PREVIEW_TITLE)
        if workspace_window is not None:
            workspace_window.focus()

    @staticmethod
    def _to_rgba(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = image
        rgba[..., 1] = image
        rgba[..., 2] = image
        rgba[..., 3] = 255
        return rgba

    def _update_provider(self, provider, image: np.ndarray):
        rgba = np.ascontiguousarray(self._to_rgba(image))
        provider.set_bytes_data(rgba.flatten().data, [rgba.shape[1], rgba.shape[0]])

    def update(self, left_force_map: np.ndarray, right_force_map: np.ndarray):
        self._update_provider(self._left_provider, force_map_to_debug_image(left_force_map))
        self._update_provider(self._right_provider, force_map_to_debug_image(right_force_map))

    def destroy(self):
        if self._window is not None:
            self._window.visible = False
            self._window.destroy()
            self._window = None


def show_force_map_pair(left_force_map: np.ndarray, right_force_map: np.ndarray):
    """Show a side-by-side live preview of the two virtual pad force maps."""
    global _force_preview_announced, _force_preview_available, _force_preview_window

    if not args_cli.show_force_maps or not _force_preview_available:
        return

    try:
        if _force_preview_window is None:
            rows, cols = pad_image_shape()
            _force_preview_window = ForceMapPreviewWindow(rows, cols, args_cli.force_preview_scale)
        if not _force_preview_announced:
            print(f"[INFO]: Force-map preview panel opened in Isaac UI: {_FORCE_PREVIEW_TITLE}.")
            _force_preview_announced = True
        _force_preview_window.update(left_force_map, right_force_map)
    except Exception as exc:
        print(f"[WARN]: Force-map preview disabled: {exc}")
        _force_preview_available = False


def destroy_force_preview_window():
    """Destroy the in-app force-map preview window if it was created."""
    global _force_preview_window

    if _force_preview_window is not None:
        _force_preview_window.destroy()
        _force_preview_window = None


def maybe_save_force_map_pair(
    env,
    step_count: int,
    saved_count: int,
    pick_sm: PickAndLiftSm,
    output_dir: Path,
    left_body_id: int,
    right_body_id: int,
) -> int:
    """Save contact-gated force maps during grasp and lift states."""
    if args_cli.max_saved_frames == 0:
        return saved_count
    if args_cli.save_buffer_size <= 0 and args_cli.max_saved_frames > 0 and saved_count >= args_cli.max_saved_frames:
        return saved_count
    if step_count % args_cli.save_every != 0:
        return saved_count
    if pick_sm.sm_state[args_cli.record_env_id].item() < 3:
        return saved_count

    left_result, right_result = compute_force_map_pair(env, left_body_id, right_body_id)
    left_force_map, left_total_force, left_hit_count, left_warning = left_result
    right_force_map, right_total_force, right_hit_count, right_warning = right_result

    frame_slot = saved_count % args_cli.save_buffer_size if args_cli.save_buffer_size > 0 else saved_count
    frame_id = f"{frame_slot:05d}"
    save_force_map(output_dir / "left_force_map" / f"{frame_id}.npy", left_force_map)
    save_force_map(output_dir / "right_force_map" / f"{frame_id}.npy", right_force_map)
    save_force_debug(output_dir / "debug_force_left" / f"{frame_id}.png", left_force_map)
    save_force_debug(output_dir / "debug_force_right" / f"{frame_id}.png", right_force_map)

    left_sum = float(np.sum(left_force_map))
    right_sum = float(np.sum(right_force_map))
    print(
        f"[FORCE_MAP] frame={frame_id} "
        f"left_total={left_total_force:.6f}N left_sum={left_sum:.6f}N "
        f"left_points={left_hit_count} left_nonzero={int(np.count_nonzero(left_force_map))} "
        f"right_total={right_total_force:.6f}N right_sum={right_sum:.6f}N "
        f"right_points={right_hit_count} right_nonzero={int(np.count_nonzero(right_force_map))}"
    )
    if left_warning is not None:
        print(f"[WARN]: left frame={frame_id}: {left_warning}.")
    if right_warning is not None:
        print(f"[WARN]: right frame={frame_id}: {right_warning}.")

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
    if args_cli.force_debug_full_scale < 0.0:
        raise ValueError("--force_debug_full_scale must be non-negative.")
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

    if saving_enabled:
        print(f"[INFO]: Saving contact-gated force maps under: {output_dir.resolve()}")
    else:
        print("[INFO]: Local force-map saving disabled by default. Use --max_saved_frames N to save frames.")
    print(
        "[INFO]: Virtual pad grid "
        f"rows={rows} cols={cols} size={tuple(args_cli.virtual_pad_size)} "
        f"resolution={args_cli.virtual_pad_resolution} z_offset={args_cli.virtual_pad_z_offset}."
    )
    print(
        "[INFO]: Force map model: ContactSensor-only COP maximum-entropy reconstruction. "
        f"contact_force_threshold={args_cli.contact_force_threshold} "
        f"point_force_threshold={args_cli.point_force_threshold} "
        f"force_debug_full_scale={args_cli.force_debug_full_scale}."
    )
    if args_cli.show_force_maps:
        print(
            "[INFO]: Live force-map preview enabled in Isaac UI "
            f"(every {args_cli.force_preview_every} step(s), scale={args_cli.force_preview_scale})."
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

                if args_cli.show_force_maps and step_count % args_cli.force_preview_every == 0:
                    left_result, right_result = compute_force_map_pair(env, left_body_id, right_body_id)
                    show_force_map_pair(left_result[0], right_result[0])

                saved_count = maybe_save_force_map_pair(
                    env, step_count, saved_count, pick_sm, output_dir, left_body_id, right_body_id
                )
                if args_cli.save_buffer_size <= 0 and args_cli.max_saved_frames > 0 and saved_count >= args_cli.max_saved_frames:
                    break

                if dones.any():
                    reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    pick_sm.reset_idx(reset_ids)

            step_count += 1

        if saving_enabled:
            print(f"[INFO]: Saved {saved_count} contact-gated force map frame pairs.")
        else:
            print("[INFO]: Finished without saving local force-map files.")
    finally:
        env.close()
        if args_cli.show_force_maps:
            destroy_force_preview_window()


if __name__ == "__main__":
    main()
    simulation_app.close()
