# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run Franka IK-Abs cube lift and save contact-gated per-point virtual pad force maps.

This script keeps the stock Franka lift state machine and adds two read-only
virtual tactile pads on the Panda fingers. Each pad samples an X/Z grid over the
gripper-side plane. True PhysX finger-object contact gates whether the pad can
receive force; when contact exists, the total finger contact force is distributed
across the whole virtual plane with raycast-distance weighting.

The script stops at force-map generation. It does not render OpenWorldTactile RGB, does not
create Gaussian pressure patches, and does not call the OpenWorldTactile SDK.

.. code-block:: bash

    ./isaaclab.sh -p experiments/franka/current/raycast_force_map.py --headless

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick/lift DexCube with Franka and preview contact-gated force maps.")
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
    default="outputs/franka_lift_object_contact_force_map",
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
parser.add_argument(
    "--virtual_pad_max_distance",
    type=float,
    default=0.020,
    help="Maximum raycast distance in meters used for proximity weighting.",
)
parser.add_argument(
    "--virtual_pad_side_offset",
    type=float,
    default=0.015,
    help="Side offset from each Panda finger link center to the gripper-side virtual plane.",
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
    "--force_weight_floor_ratio",
    type=float,
    default=0.05,
    help=(
        "Minimum per-point weighting floor relative to full proximity weight; keeps every point non-zero only when "
        "raycast has at least one valid proximity hit."
    ),
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
from typing import Callable

import cv2
import gymnasium as gym
import numpy as np
import torch
import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()

_FORCE_PREVIEW_TITLE = "Virtual Force Maps"
_force_preview_available = True
_force_preview_announced = False
_force_preview_window = None
_last_valid_weight_maps: dict[tuple[int, int], np.ndarray] = {}


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


def virtual_pad_xz_pattern(cfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a tactile-pad grid in local x/z with rays along local y."""
    size_x, size_z = cfg.size
    x = torch.arange(start=-size_x / 2, end=size_x / 2 + 1.0e-9, step=cfg.resolution, device=device)
    z = torch.arange(start=-size_z / 2, end=size_z / 2 + 1.0e-9, step=cfg.resolution, device=device)
    grid_x, grid_z = torch.meshgrid(x, z, indexing="xy")

    ray_starts = torch.zeros(grid_x.numel(), 3, device=device)
    ray_starts[:, 0] = grid_x.flatten()
    ray_starts[:, 2] = grid_z.flatten()

    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., :] = torch.tensor(list(cfg.direction), device=device)
    return ray_starts, ray_directions


@configclass
class VirtualPadXZPatternCfg:
    """Grid pattern for a finger-mounted virtual tactile pad."""

    func: Callable = virtual_pad_xz_pattern
    size: tuple[float, float] = (0.038, 0.048)
    resolution: float = 0.0005
    direction: tuple[float, float, float] = (0.0, -1.0, 0.0)


def pad_image_shape() -> tuple[int, int]:
    """Return force-map rows and columns for the configured grid pattern."""
    size_x, size_z = args_cli.virtual_pad_size
    resolution = args_cli.virtual_pad_resolution
    cols = int(round(size_x / resolution)) + 1
    rows = int(round(size_z / resolution)) + 1
    return rows, cols


def make_virtual_pad_cfg(finger_link: str, side_sign: float):
    """Create a finger-mounted virtual tactile pad sampler."""
    return MultiMeshRayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{finger_link}",
        update_period=0.0,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Object",
                track_mesh_transforms=True,
                merge_prim_meshes=True,
            )
        ],
        ray_alignment="base",
        max_distance=args_cli.virtual_pad_max_distance,
        pattern_cfg=VirtualPadXZPatternCfg(
            size=tuple(args_cli.virtual_pad_size),
            resolution=args_cli.virtual_pad_resolution,
            direction=(0.0, side_sign, 0.0),
        ),
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=(0.0, side_sign * args_cli.virtual_pad_side_offset, args_cli.virtual_pad_z_offset),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        debug_vis=False,
    )


def add_virtual_pad_sensors(env_cfg: LiftEnvCfg):
    """Add left and right virtual tactile pad samplers."""
    env_cfg.scene.left_virtual_pad = make_virtual_pad_cfg("panda_leftfinger", -1.0)
    env_cfg.scene.right_virtual_pad = make_virtual_pad_cfg("panda_rightfinger", 1.0)


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


def total_contact_force(contact_sensor, env_id: int) -> float:
    """Return the total finite finger-object contact force magnitude in newtons."""
    contact_data = contact_sensor.data
    contact_pos_w = contact_data.contact_pos_w[env_id].reshape(-1, 3)
    if contact_data.force_matrix_w is not None:
        contact_force_w = contact_data.force_matrix_w[env_id].reshape(-1, 3)
    else:
        contact_force_w = contact_data.net_forces_w[env_id].reshape(-1, 3)

    finite_contact = torch.isfinite(contact_pos_w).all(dim=-1) & torch.isfinite(contact_force_w).all(dim=-1)
    if not bool(finite_contact.any()):
        return 0.0

    force_norm = torch.linalg.norm(contact_force_w[finite_contact], dim=-1)
    return float(force_norm.sum().item())


def contact_gated_force_map(raycaster, contact_sensor, env_id: int) -> tuple[np.ndarray, float, int, str | None]:
    """Distribute true total contact force over all points with raycast-distance weighting."""
    rows, cols = pad_image_shape()
    zero_force_map = np.zeros((rows, cols), dtype=np.float32)

    total_force = total_contact_force(contact_sensor, env_id)
    if total_force < args_cli.contact_force_threshold:
        return zero_force_map, total_force, 0, None

    # Update finger-attached rays and object hits for distance-based weighting.
    ray_hits_w = raycaster.data.ray_hits_w[env_id]
    ray_starts_w = raycaster._ray_starts_w[env_id]
    if ray_starts_w.shape[0] != rows * cols:
        raise RuntimeError(f"Virtual pad ray count mismatch: got {ray_starts_w.shape[0]}, expected {rows * cols}.")

    hit_distance = torch.linalg.norm(ray_hits_w - ray_starts_w, dim=-1)
    finite_hits = torch.isfinite(ray_hits_w).all(dim=-1)
    proximity = torch.clamp(args_cli.virtual_pad_max_distance - hit_distance, min=0.0)
    proximity = torch.where(finite_hits, proximity, torch.zeros_like(proximity))
    active_hits = proximity > 0.0
    active_hit_count = int(active_hits.sum().item())

    floor_weight = args_cli.force_weight_floor_ratio * args_cli.virtual_pad_max_distance
    cache_key = (id(raycaster), env_id)
    warning = None
    if active_hit_count > 0:
        weights = proximity + floor_weight
        _last_valid_weight_maps[cache_key] = weights.detach().cpu().numpy().astype(np.float32)
    else:
        cached_weights = _last_valid_weight_maps.get(cache_key)
        if cached_weights is not None:
            weights = torch.as_tensor(cached_weights, device=proximity.device, dtype=proximity.dtype)
        else:
            weights = torch.ones_like(proximity)
            warning = "true contact exists but raycast has no hits and no cache exists; using uniform fallback"

    weight_sum = torch.sum(weights)
    if float(weight_sum.item()) <= 0.0:
        return zero_force_map, total_force, 0, "all force-map weights are zero"

    force_flat = (float(total_force) * weights / weight_sum).to(dtype=torch.float32)
    force_map = force_flat.reshape(rows, cols).detach().cpu().numpy().astype(np.float32)

    if args_cli.point_force_threshold > 0.0:
        force_map = np.where(force_map >= args_cli.point_force_threshold, force_map, 0.0).astype(np.float32)

    return force_map, total_force, active_hit_count, warning


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


def compute_force_map_pair(env):
    """Compute left and right contact-gated virtual pad force maps for the selected environment."""
    scene = env.unwrapped.scene
    left_result = contact_gated_force_map(
        scene["left_virtual_pad"], scene["left_finger_contact"], args_cli.record_env_id
    )
    right_result = contact_gated_force_map(
        scene["right_virtual_pad"], scene["right_finger_contact"], args_cli.record_env_id
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

    left_result, right_result = compute_force_map_pair(env)
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
    if args_cli.virtual_pad_max_distance <= 0.0:
        raise ValueError("--virtual_pad_max_distance must be positive.")
    if args_cli.save_buffer_size < 0:
        raise ValueError("--save_buffer_size must be non-negative.")
    if args_cli.contact_force_threshold < 0.0:
        raise ValueError("--contact_force_threshold must be non-negative.")
    if args_cli.point_force_threshold < 0.0:
        raise ValueError("--point_force_threshold must be non-negative.")
    if args_cli.force_weight_floor_ratio < 0.0:
        raise ValueError("--force_weight_floor_ratio must be non-negative.")
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
    add_virtual_pad_sensors(env_cfg)
    add_finger_contact_sensors(env_cfg)

    env_step_dt = env_cfg.sim.dt * env_cfg.decimation
    if args_cli.max_steps > 0:
        # Keep the scripted run from timing out before the requested capture window.
        env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.max_steps * env_step_dt + 1.0)

    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

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
        f"resolution={args_cli.virtual_pad_resolution} side_offset={args_cli.virtual_pad_side_offset} "
        f"z_offset={args_cli.virtual_pad_z_offset} max_distance={args_cli.virtual_pad_max_distance}."
    )
    print(
        "[INFO]: Force map model: true contact gates whole-plane raycast-distance weighted assignment. "
        f"contact_force_threshold={args_cli.contact_force_threshold} "
        f"point_force_threshold={args_cli.point_force_threshold} "
        f"force_weight_floor_ratio={args_cli.force_weight_floor_ratio} "
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
                    left_result, right_result = compute_force_map_pair(env)
                    show_force_map_pair(left_result[0], right_result[0])

                saved_count = maybe_save_force_map_pair(env, step_count, saved_count, pick_sm, output_dir)
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
