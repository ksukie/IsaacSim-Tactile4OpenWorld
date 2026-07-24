# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run Franka IK-Abs cube lift with GelSight-like virtual contact pads and OpenWorldTactile RGB.

This script keeps the stock Franka lift state machine and adds two read-only
virtual tactile pads on the Panda fingers. Each pad samples a virtual elastomer
surface. True PhysX finger-object contacts are projected onto that surface to
form local deformation patches, rendered through the OpenWorldTactile RGB renderer, and
fed to the OpenWorldTactile SDK bridge for virtual force output.

The virtual pads do not add geometry, collisions, cameras, or extra articulations
to the Franka grasp, so they should not disturb the stable lift physics.

.. code-block:: bash

    ./isaaclab.sh -p experiments/franka/current/contact_points_openworldtactile.py --headless

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick/lift DexCube with Franka and save contact-based OpenWorldTactile tactile RGB.")
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
    default="outputs/franka_lift_object_contact_openworldtactile",
    help="Directory for RGB, debug pressure, and OpenWorldTactile force outputs.",
)
parser.add_argument("--save_every", type=int, default=5, help="Save every N steps while grasping/lifting.")
parser.add_argument(
    "--max_saved_frames",
    type=int,
    default=80,
    help="Stop saving after this many frame pairs; 0 disables saving; negative saves without a frame limit.",
)
parser.add_argument(
    "--save_buffer_size",
    type=int,
    default=0,
    help="If positive, save PNGs in a circular buffer with this many frame slots.",
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
    help="Maximum helper raycast distance in meters; pressure does not use ray hit regions.",
)
parser.add_argument(
    "--virtual_pad_side_offset",
    type=float,
    default=0.012,
    help="Side offset from each Panda finger link center toward the gripper gap.",
)
parser.add_argument("--virtual_pad_z_offset", type=float, default=0.074, help="Virtual pad Z offset in each finger frame.")
parser.add_argument("--save_debug_pressure", action="store_true", help="Save normalized pressure-map debug images.")
parser.add_argument("--openworldtactile_baseline_frames", type=int, default=5, help="Number of zero-pressure RGB frames for SDK baseline.")
parser.add_argument(
    "--openworldtactile_image_size",
    type=int,
    nargs=2,
    default=(320, 240),
    metavar=("HEIGHT", "WIDTH"),
    help="Output OpenWorldTactile RGB/debug pressure image size.",
)
parser.add_argument(
    "--openworldtactile_max_pressure",
    type=float,
    default=1.0,
    help="Maximum pressure value used by OpenWorldTactileRGBRenderer; pressure maps are normalized to [0, 1].",
)
parser.add_argument(
    "--pressure_spread_sigma_px",
    type=float,
    default=7.0,
    help="Gaussian blur sigma in final OpenWorldTactile pixels for elastomer pressure spread.",
)
parser.add_argument(
    "--pressure_hardness",
    type=float,
    default=1.0,
    help="Power curve applied to normalized pressure; 1.0 keeps it linear.",
)
parser.add_argument(
    "--contact_force_threshold",
    type=float,
    default=0.05,
    help="Minimum filtered finger-object contact force in newtons required to render pressure.",
)
parser.add_argument(
    "--contact_force_full_scale",
    type=float,
    default=20.0,
    help="Filtered contact force in newtons that maps to pressure amplitude 1.",
)
parser.add_argument(
    "--contact_point_max_pad_distance",
    type=float,
    default=0.020,
    help="Maximum distance in meters from contact point to virtual pad before ignoring that contact.",
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
import sys
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
    """Return pressure-map rows and columns for the configured grid pattern."""
    size_x, size_z = args_cli.virtual_pad_size
    resolution = args_cli.virtual_pad_resolution
    cols = int(round(size_x / resolution)) + 1
    rows = int(round(size_z / resolution)) + 1
    return rows, cols


def openworldtactile_image_shape() -> tuple[int, int]:
    """Return final OpenWorldTactile image height and width."""
    height, width = args_cli.openworldtactile_image_size
    return int(height), int(width)


def make_virtual_pad_cfg(name: str, finger_link: str, side_sign: float):
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
    env_cfg.scene.left_virtual_pad = make_virtual_pad_cfg(
        name="left",
        finger_link="panda_leftfinger",
        side_sign=-1.0,
    )
    env_cfg.scene.right_virtual_pad = make_virtual_pad_cfg(
        name="right",
        finger_link="panda_rightfinger",
        side_sign=1.0,
    )


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


def contact_point_pressure_map(raycaster, contact_sensor, env_id: int) -> np.ndarray:
    """Render a local pressure patch from true finger-object PhysX contact."""
    rows, cols = pad_image_shape()
    image_height, image_width = openworldtactile_image_shape()
    zero_pressure = np.zeros((image_height, image_width), dtype=np.float32)

    # Update the finger-attached virtual pad positions.
    _ = raycaster.data.ray_hits_w
    starts = raycaster._ray_starts_w[env_id]
    if starts.shape[0] != rows * cols:
        raise RuntimeError(f"Virtual pad ray count mismatch: got {starts.shape[0]}, expected {rows * cols}.")

    contact_data = contact_sensor.data
    contact_pos_w = contact_data.contact_pos_w[env_id].reshape(-1, 3)
    if contact_data.force_matrix_w is not None:
        contact_force_w = contact_data.force_matrix_w[env_id].reshape(-1, 3)
    else:
        contact_force_w = contact_data.net_forces_w[env_id].reshape(-1, 3)

    finite_contact = torch.isfinite(contact_pos_w).all(dim=-1) & torch.isfinite(contact_force_w).all(dim=-1)
    if not bool(finite_contact.any()):
        return zero_pressure

    force_norm = torch.linalg.norm(contact_force_w, dim=-1)
    valid_contact = finite_contact & (force_norm >= args_cli.contact_force_threshold)
    if not bool(valid_contact.any()):
        return zero_pressure

    yy, xx = np.mgrid[0:image_height, 0:image_width].astype(np.float32)
    pressure = np.zeros((image_height, image_width), dtype=np.float32)
    sigma = float(args_cli.pressure_spread_sigma_px)

    for contact_point, force_value in zip(contact_pos_w[valid_contact], force_norm[valid_contact]):
        distances = torch.linalg.norm(starts - contact_point.unsqueeze(0), dim=-1)
        min_distance, nearest_idx = torch.min(distances, dim=0)
        if float(min_distance.item()) > args_cli.contact_point_max_pad_distance:
            continue

        nearest_idx_int = int(nearest_idx.item())
        row = nearest_idx_int // cols
        col = nearest_idx_int % cols
        center_x = 0.0 if cols <= 1 else col * (image_width - 1) / float(cols - 1)
        center_y = 0.0 if rows <= 1 else row * (image_height - 1) / float(rows - 1)
        amplitude = float(torch.clamp(force_value / args_cli.contact_force_full_scale, 0.0, 1.0).item())
        amplitude = amplitude ** args_cli.pressure_hardness

        if sigma <= 0.0:
            pressure[int(round(center_y)), int(round(center_x))] = max(
                pressure[int(round(center_y)), int(round(center_x))], amplitude
            )
        else:
            patch = amplitude * np.exp(-0.5 * (((xx - center_x) / sigma) ** 2 + ((yy - center_y) / sigma) ** 2))
            pressure = np.maximum(pressure, patch.astype(np.float32))

    return np.clip(pressure, 0.0, 1.0).astype(np.float32)


def make_openworldtactile_renderer() -> OpenWorldTactileRGBRenderer:
    """Create one OpenWorldTactile renderer for normalized virtual pressure maps."""
    height, width = openworldtactile_image_shape()
    return OpenWorldTactileRGBRenderer(
        height=height,
        width=width,
        max_pressure=args_cli.openworldtactile_max_pressure,
        base_value=220,
        pressure_blur=5,
        displacement_scale=12000.0,
    )


def make_openworldtactile_bridge(side: str, output_dir: Path) -> IsaacLabOpenWorldTactileBridge:
    """Create one SDK bridge and force CSV writer for one side."""
    return IsaacLabOpenWorldTactileBridge(
        fx_p1=1.0,
        fy_p1=1.0,
        fz_p1=1.0,
        baseline_frames=args_cli.openworldtactile_baseline_frames,
        env_id=0,
        csv_path=output_dir / f"{side}_openworldtactile_forces.csv",
        save_input_rgb=False,
        record_buffer_size=args_cli.save_buffer_size if args_cli.save_buffer_size > 0 else None,
    )


def calibrate_openworldtactile_baseline(renderer: OpenWorldTactileRGBRenderer, bridge: IsaacLabOpenWorldTactileBridge):
    """Calibrate the SDK bridge with zero-pressure OpenWorldTactile RGB frames."""
    height, width = openworldtactile_image_shape()
    zero_pressure = np.zeros((height, width), dtype=np.float32)
    for _ in range(args_cli.openworldtactile_baseline_frames):
        bridge.update(renderer.render(zero_pressure)[None, ...])


def save_rgb(path: Path, rgb: np.ndarray):
    """Save an RGB image through OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_pressure_debug(path: Path, pressure: np.ndarray):
    """Save one normalized pressure map as a grayscale debug image."""
    image = (np.clip(pressure, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def force_to_text(force: tuple[float, float, float] | None) -> str:
    """Return compact force text for logging."""
    if force is None:
        return "baseline"
    fx, fy, fz = force
    return f"fx={fx:.3f} fy={fy:.3f} fz={fz:.3f}"


def maybe_save_virtual_tactile_pair(
    env,
    step_count: int,
    saved_count: int,
    pick_sm: PickAndLiftSm,
    output_dir: Path,
    left_renderer: OpenWorldTactileRGBRenderer,
    right_renderer: OpenWorldTactileRGBRenderer,
    left_bridge: IsaacLabOpenWorldTactileBridge,
    right_bridge: IsaacLabOpenWorldTactileBridge,
) -> int:
    """Save contact-based OpenWorldTactile RGB frames during grasp and lift states."""
    if args_cli.max_saved_frames == 0:
        return saved_count
    if args_cli.save_buffer_size <= 0 and args_cli.max_saved_frames > 0 and saved_count >= args_cli.max_saved_frames:
        return saved_count
    if step_count % args_cli.save_every != 0:
        return saved_count
    if pick_sm.sm_state[args_cli.record_env_id].item() < 3:
        return saved_count

    scene = env.unwrapped.scene
    left_pressure = contact_point_pressure_map(
        scene["left_virtual_pad"], scene["left_finger_contact"], args_cli.record_env_id
    )
    right_pressure = contact_point_pressure_map(
        scene["right_virtual_pad"], scene["right_finger_contact"], args_cli.record_env_id
    )

    left_rgb = left_renderer.render(left_pressure)
    right_rgb = right_renderer.render(right_pressure)
    left_force = left_bridge.update(left_rgb[None, ...])
    right_force = right_bridge.update(right_rgb[None, ...])

    frame_slot = saved_count % args_cli.save_buffer_size if args_cli.save_buffer_size > 0 else saved_count
    frame_id = f"{frame_slot:05d}"
    save_rgb(output_dir / "left" / f"{frame_id}.png", left_rgb)
    save_rgb(output_dir / "right" / f"{frame_id}.png", right_rgb)
    if args_cli.save_debug_pressure:
        save_pressure_debug(output_dir / "debug_pressure_left" / f"{frame_id}.png", left_pressure)
        save_pressure_debug(output_dir / "debug_pressure_right" / f"{frame_id}.png", right_pressure)

    print(
        f"[VIRTUAL_TACTILE] frame={frame_id} "
        f"left_pressure_max={float(np.max(left_pressure)):.3f} "
        f"right_pressure_max={float(np.max(right_pressure)):.3f} "
        f"left_{force_to_text(left_force)} right_{force_to_text(right_force)}"
    )
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
    if args_cli.openworldtactile_baseline_frames <= 0:
        raise ValueError("--openworldtactile_baseline_frames must be positive.")
    if args_cli.openworldtactile_max_pressure <= 0.0:
        raise ValueError("--openworldtactile_max_pressure must be positive.")
    if args_cli.save_buffer_size < 0:
        raise ValueError("--save_buffer_size must be non-negative.")
    image_height, image_width = openworldtactile_image_shape()
    if image_height <= 0 or image_width <= 0:
        raise ValueError("--openworldtactile_image_size values must be positive.")
    if args_cli.pressure_spread_sigma_px < 0.0:
        raise ValueError("--pressure_spread_sigma_px must be non-negative.")
    if args_cli.pressure_hardness <= 0.0:
        raise ValueError("--pressure_hardness must be positive.")
    if args_cli.contact_force_threshold < 0.0:
        raise ValueError("--contact_force_threshold must be non-negative.")
    if args_cli.contact_force_full_scale <= 0.0:
        raise ValueError("--contact_force_full_scale must be positive.")
    if args_cli.contact_point_max_pad_distance <= 0.0:
        raise ValueError("--contact_point_max_pad_distance must be positive.")

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

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    left_renderer = make_openworldtactile_renderer()
    right_renderer = make_openworldtactile_renderer()
    left_bridge = make_openworldtactile_bridge("left", output_dir)
    right_bridge = make_openworldtactile_bridge("right", output_dir)
    calibrate_openworldtactile_baseline(left_renderer, left_bridge)
    calibrate_openworldtactile_baseline(right_renderer, right_bridge)

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0

    pick_sm = PickAndLiftSm(env_step_dt, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.01)
    saved_count = 0

    rows, cols = pad_image_shape()
    image_height, image_width = openworldtactile_image_shape()
    print(f"[INFO]: Saving contact-based OpenWorldTactile tactile outputs under: {output_dir.resolve()}")
    print(
        "[INFO]: Virtual pad grid "
        f"rows={rows} cols={cols} size={tuple(args_cli.virtual_pad_size)} "
        f"resolution={args_cli.virtual_pad_resolution} side_offset={args_cli.virtual_pad_side_offset} "
        f"z_offset={args_cli.virtual_pad_z_offset}."
    )
    print(
        "[INFO]: Contact model "
        f"force_threshold={args_cli.contact_force_threshold} "
        f"force_full_scale={args_cli.contact_force_full_scale} "
        f"max_pad_distance={args_cli.contact_point_max_pad_distance} "
        f"spread_sigma_px={args_cli.pressure_spread_sigma_px} hardness={args_cli.pressure_hardness}."
    )
    print(f"[INFO]: OpenWorldTactile output height={image_height} width={image_width}.")
    print("[INFO]: OpenWorldTactile SDK baseline calibrated from zero-pressure RGB frames.")

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

                saved_count = maybe_save_virtual_tactile_pair(
                    env,
                    step_count,
                    saved_count,
                    pick_sm,
                    output_dir,
                    left_renderer,
                    right_renderer,
                    left_bridge,
                    right_bridge,
                )
                if args_cli.save_buffer_size <= 0 and args_cli.max_saved_frames > 0 and saved_count >= args_cli.max_saved_frames:
                    break

                if dones.any():
                    reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    pick_sm.reset_idx(reset_ids)

            step_count += 1

        print(f"[INFO]: Saved {saved_count} contact-based OpenWorldTactile tactile RGB frame pairs.")
    finally:
        left_bridge.close()
        right_bridge.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
