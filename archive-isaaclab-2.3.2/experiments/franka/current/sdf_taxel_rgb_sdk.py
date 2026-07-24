# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run Franka IK-Abs nut lift and preview SDF-taxel OpenWorldTactile RGB.

This script keeps the stock Franka lift state machine and replaces the prior
COP-style reconstruction with a virtual taxel pad on each Panda finger. Each taxel
queries the object's SDF collision mesh to produce dense tactile fields:

* normal force comes from SDF penetration depth;
* shear force comes from relative tangential velocity with a Coulomb cap;
* normal pressure drives OpenWorldTactile RGB hue;
* local shear drives OpenWorldTactile texture displacement for SDK fx/fy.

It does not fall back to the older COP or friction scripts if an SDF collision mesh is missing.

.. code-block:: bash

    ./isaaclab.sh -p experiments/franka/current/sdf_taxel_rgb_sdk.py

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

DEFAULT_NUT_GRASP_Z_OFFSET = 0.0

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Pick/lift an SDF object with Franka and preview SDF-taxel textured OpenWorldTactile RGB maps."
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
    default="outputs/franka_lift_object_sdf_taxel_rgb",
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
    "--fxyz_source",
    type=str,
    choices=("direct_sdf", "sdk_flow"),
    default="direct_sdf",
    help="Source for FXYZ preview/calculation: direct SDF shear/normal maps or SDK optical flow from RGB.",
)
parser.add_argument(
    "--openworldtactile_shear_px_per_n",
    type=float,
    default=0.45,
    help="Pixels of OpenWorldTactile texture displacement per local SDF shear newton.",
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
    "--sdf_object_usd",
    type=str,
    default="Factory/factory_nut_m16.usd",
    help="SDF object USD path. Relative paths are resolved under ISAACLAB_NUCLEUS_DIR.",
)
parser.add_argument(
    "--sdf_object_scale",
    type=float,
    nargs=3,
    default=(1.0, 1.0, 1.0),
    metavar=("SX", "SY", "SZ"),
    help="Scale for the SDF object USD.",
)
parser.add_argument("--sdf_object_mass", type=float, default=0.03, help="Mass of the SDF object in kg.")
parser.add_argument("--sdf_object_z", type=float, default=0.055, help="Initial world/env-local object Z position.")
parser.add_argument("--sdf_object_static_friction", type=float, default=1.0, help="Static friction for the SDF object.")
parser.add_argument("--sdf_object_dynamic_friction", type=float, default=1.0, help="Dynamic friction for the SDF object.")
parser.add_argument(
    "--sdf_object_solver_position_iterations",
    type=int,
    default=64,
    help="PhysX solver position iterations for the SDF object.",
)
parser.add_argument(
    "--sdf_object_solver_velocity_iterations",
    type=int,
    default=4,
    help="PhysX solver velocity iterations for the SDF object.",
)
parser.add_argument(
    "--robot_solver_position_iterations",
    type=int,
    default=16,
    help="PhysX articulation solver position iterations for the Franka robot.",
)
parser.add_argument(
    "--robot_solver_velocity_iterations",
    type=int,
    default=1,
    help="PhysX articulation solver velocity iterations for the Franka robot.",
)
parser.add_argument(
    "--finger_surface_offset",
    type=float,
    default=0.012,
    help="Finger-local Y offset of the virtual taxel plane toward the gripper gap.",
)
parser.add_argument(
    "--grasp_position_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, DEFAULT_NUT_GRASP_Z_OFFSET),
    metavar=("DX", "DY", "DZ"),
    help="Env-local offset added to the object root position before grasping.",
)
parser.add_argument(
    "--lift_with_grasp_offset",
    action="store_true",
    default=True,
    help="Apply the grasp position offset to the lift target so the TCP keeps the same grasp point.",
)
parser.add_argument(
    "--no_lift_with_grasp_offset",
    action="store_false",
    dest="lift_with_grasp_offset",
    help="Lift the TCP to the commanded object target without the grasp position offset.",
)
parser.add_argument(
    "--lift_target_mode",
    type=str,
    choices=("straight_up", "command"),
    default="straight_up",
    help="Use a straight-up demo lift target or the environment command target.",
)
parser.add_argument("--lift_height", type=float, default=0.12, help="Straight-up lift height above the reset object pose.")
parser.add_argument("--approach_height", type=float, default=0.10, help="Height above the grasp pose for approach.")
parser.add_argument(
    "--position_threshold",
    type=float,
    default=0.02,
    help="TCP distance threshold in meters for state-machine transitions.",
)
parser.add_argument(
    "--approach_above_timeout",
    type=float,
    default=2.0,
    help="Fallback seconds before leaving the above-object approach state.",
)
parser.add_argument(
    "--approach_object_timeout",
    type=float,
    default=2.0,
    help="Fallback seconds before closing the gripper near the object.",
)
parser.add_argument(
    "--grasp_wait_time",
    type=float,
    default=1.5,
    help="Seconds to keep closing the gripper before lifting; small objects need more time.",
)
parser.add_argument("--normal_contact_stiffness", type=float, default=1.0, help="SDF taxel normal stiffness.")
parser.add_argument("--tangential_stiffness", type=float, default=0.1, help="SDF taxel tangential stiffness.")
parser.add_argument("--friction_coefficient", type=float, default=2.0, help="SDF taxel friction coefficient.")
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
parser.add_argument("--debug_grasp", action="store_true", help="Print periodic grasp/lift diagnostics.")
parser.add_argument("--debug_grasp_every", type=int, default=25, help="Print grasp diagnostics every N steps.")
parser.add_argument(
    "--grasp_success_height",
    type=float,
    default=0.02,
    help="Object height increase in meters used by --debug_grasp to report lifted=True.",
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

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaacsim.core.simulation_manager import SimulationManager
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from pxr import Usd, UsdPhysics

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

_FORCE_PREVIEW_TITLE = "SDF Taxel RGB + SDK FXYZ"
_force_preview_available = True
_force_preview_announced = False
_force_preview_window = None


@dataclass
class ContactPadMaps:
    """Dense virtual pad fields computed from one finger SDF taxel pad."""

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
    APPROACH_ABOVE_TIMEOUT = wp.constant(float(args_cli.approach_above_timeout))
    APPROACH_OBJECT_TIMEOUT = wp.constant(float(args_cli.approach_object_timeout))
    GRASP_OBJECT = wp.constant(float(args_cli.grasp_wait_time))
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
        reached = distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        )
        if (reached and sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT) or (
            sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_TIMEOUT
        ):
            sm_state[tid] = PickSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        reached = distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        )
        if (reached and sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT) or (
            sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT_TIMEOUT
        ):
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

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str = "cpu",
        position_threshold=0.01,
        approach_height: float = 0.1,
    ):
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
        self.offset[:, 2] = float(approach_height)
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


def resolve_sdf_object_usd() -> str:
    """Resolve the configured SDF object USD path."""
    object_usd = str(args_cli.sdf_object_usd)
    if "://" in object_usd or object_usd.startswith("/"):
        return object_usd
    return f"{ISAACLAB_NUCLEUS_DIR}/{object_usd.lstrip('/')}"


def configure_sdf_object(env_cfg: LiftEnvCfg, object_usd: str):
    """Replace the stock DexCube with a rigid USD object that must contain an SDF collision mesh."""
    env_cfg.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, args_cli.sdf_object_z], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=object_usd,
            scale=tuple(float(value) for value in args_cli.sdf_object_scale),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=args_cli.sdf_object_solver_position_iterations,
                solver_velocity_iteration_count=args_cli.sdf_object_solver_velocity_iterations,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.sdf_object_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
    )
    if env_cfg.events.reset_object_position is not None:
        env_cfg.events.reset_object_position.params["asset_cfg"] = SceneEntityCfg("object")
    env_cfg.events.object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (
                args_cli.sdf_object_static_friction,
                args_cli.sdf_object_static_friction,
            ),
            "dynamic_friction_range": (
                args_cli.sdf_object_dynamic_friction,
                args_cli.sdf_object_dynamic_friction,
            ),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    env_cfg.events.robot_finger_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_.*finger"),
            "static_friction_range": (
                args_cli.sdf_object_static_friction,
                args_cli.sdf_object_static_friction,
            ),
            "dynamic_friction_range": (
                args_cli.sdf_object_dynamic_friction,
                args_cli.sdf_object_dynamic_friction,
            ),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )


def configure_robot_contact_solver(env_cfg: LiftEnvCfg):
    """Increase Franka articulation solver iterations for small-object grasps."""
    articulation_props = env_cfg.scene.robot.spawn.articulation_props
    if articulation_props is None:
        return
    articulation_props.solver_position_iteration_count = args_cli.robot_solver_position_iterations
    articulation_props.solver_velocity_iteration_count = args_cli.robot_solver_velocity_iterations


def _is_sdf_mesh(prim: Usd.Prim) -> bool:
    """Return true when a USD mesh prim is configured as an SDF collision mesh."""
    return (
        prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "sdf"
    )


def _find_sdf_mesh_path(scene) -> str:
    """Find the first SDF collision mesh under env_0/Object and return an env_* path pattern."""
    object_path = f"{scene.env_ns}/env_0/Object"
    object_prim = sim_utils.find_first_matching_prim(object_path)
    if object_prim is None:
        raise RuntimeError(f"No Object prim found at expected path: {object_path}")

    sdf_mesh_prim = sim_utils.get_first_matching_child_prim(object_prim.GetPath(), predicate=_is_sdf_mesh)
    if sdf_mesh_prim is None:
        raise RuntimeError(
            "No SDF mesh found under contact object at path: "
            f"{object_prim.GetPath().pathString}. "
            "Use a USD object with MeshCollisionAPI approximation='sdf'."
        )
    return sdf_mesh_prim.GetPath().pathString.replace("env_0", "env_*")


def _make_taxel_points(side_sign: float) -> torch.Tensor:
    """Create finger-local taxel points on the virtual Panda finger pad."""
    x_axis, z_axis = pad_grid_coordinates()
    grid_z, grid_x = np.meshgrid(z_axis, x_axis, indexing="ij")
    y = np.full_like(grid_x, side_sign * float(args_cli.finger_surface_offset), dtype=np.float32)
    z = grid_z + float(args_cli.virtual_pad_z_offset)
    points = np.stack([grid_x, y, z], axis=-1).reshape(-1, 3)
    return torch.tensor(points, dtype=torch.float32)


class SdfTaxelPadSampler:
    """Compute dense virtual tactile fields from Panda finger taxels and an object SDF."""

    def __init__(self, env, left_body_id: int, right_body_id: int):
        self.scene = env.unwrapped.scene
        self.robot = self.scene["robot"]
        self.object = self.scene["object"]
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self.rows, self.cols = pad_image_shape()
        self.num_taxels = self.rows * self.cols
        self.left_body_id = left_body_id
        self.right_body_id = right_body_id

        self.left_points_b = _make_taxel_points(-1.0).to(self.device)
        self.right_points_b = _make_taxel_points(1.0).to(self.device)
        self.sdf_mesh_path = _find_sdf_mesh_path(self.scene)
        self.sdf_view = SimulationManager.get_physics_sim_view().create_sdf_shape_view(
            self.sdf_mesh_path, self.num_taxels
        )

    def compute_pair(self, env_id: int) -> tuple[ContactPadMaps, ContactPadMaps]:
        """Compute left and right SDF-taxel maps for one environment."""
        left_result = self.compute_one(self.left_body_id, self.left_points_b, env_id)
        right_result = self.compute_one(self.right_body_id, self.right_points_b, env_id)
        return left_result, right_result

    def compute_one(self, finger_body_id: int, points_b: torch.Tensor, env_id: int) -> ContactPadMaps:
        """Compute one finger's normal-force and shear-displacement maps."""
        body_pos_w = self.robot.data.body_pos_w[:, finger_body_id]
        body_quat_w = self.robot.data.body_quat_w[:, finger_body_id]
        body_quat_taxel = body_quat_w[:, None, :].expand(-1, self.num_taxels, -1)
        points_b_all = points_b.unsqueeze(0).expand(self.num_envs, -1, -1)
        points_w = quat_apply(body_quat_taxel, points_b_all) + body_pos_w[:, None, :]

        object_pos_w = self.object.data.root_pos_w
        object_quat_w = self.object.data.root_quat_w
        object_quat_taxel = object_quat_w[:, None, :].expand(-1, self.num_taxels, -1)
        points_object = quat_apply_inverse(object_quat_taxel, points_w - object_pos_w[:, None, :])

        sdf_values_and_gradients = self.sdf_view.get_sdf_and_gradients(points_object)
        sdf_values = torch.nan_to_num(sdf_values_and_gradients[..., -1], nan=0.0, posinf=0.0, neginf=0.0)
        sdf_gradients = torch.nan_to_num(sdf_values_and_gradients[..., :-1], nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.clamp(-sdf_values, min=0.0)
        normal_force = float(args_cli.normal_contact_stiffness) * depth

        normals_local = torch.nn.functional.normalize(sdf_gradients, dim=-1)
        normals_world = quat_apply(object_quat_taxel, normals_local)

        finger_lin_vel_w = self.robot.data.body_lin_vel_w[:, finger_body_id]
        finger_ang_vel_w = self.robot.data.body_ang_vel_w[:, finger_body_id]
        finger_r_w = points_w - body_pos_w[:, None, :]
        taxel_vel_w = finger_lin_vel_w[:, None, :] + torch.cross(
            finger_ang_vel_w[:, None, :].expand(-1, self.num_taxels, -1), finger_r_w, dim=-1
        )

        object_lin_vel_w = self.object.data.body_lin_vel_w[:, 0]
        object_ang_vel_w = self.object.data.body_ang_vel_w[:, 0]
        closest_points_object = points_object + depth.unsqueeze(-1) * normals_local
        closest_points_w = quat_apply(object_quat_taxel, closest_points_object) + object_pos_w[:, None, :]
        object_r_w = closest_points_w - object_pos_w[:, None, :]
        object_point_vel_w = object_lin_vel_w[:, None, :] + torch.cross(
            object_ang_vel_w[:, None, :].expand(-1, self.num_taxels, -1), object_r_w, dim=-1
        )

        relative_vel_w = taxel_vel_w - object_point_vel_w
        normal_vel = torch.sum(normals_world * relative_vel_w, dim=-1, keepdim=True)
        tangential_vel_w = relative_vel_w - normals_world * normal_vel
        tangential_speed = torch.linalg.norm(tangential_vel_w, dim=-1)
        shear_static = float(args_cli.tangential_stiffness) * tangential_speed
        shear_dynamic = float(args_cli.friction_coefficient) * normal_force
        shear_mag = torch.minimum(shear_static, shear_dynamic)
        shear_world = -shear_mag.unsqueeze(-1) * tangential_vel_w / tangential_speed.unsqueeze(-1).clamp(min=1.0e-9)
        shear_finger = quat_apply_inverse(body_quat_taxel, shear_world)

        env_normal = torch.clamp(normal_force[env_id], min=0.0)
        env_shear = shear_finger[env_id]
        displacement_x = torch.clamp(
            env_shear[:, 0] * float(args_cli.openworldtactile_shear_px_per_n),
            -float(args_cli.openworldtactile_shear_clip_px),
            float(args_cli.openworldtactile_shear_clip_px),
        )
        displacement_y = torch.clamp(
            -env_shear[:, 2] * float(args_cli.openworldtactile_shear_px_per_n),
            -float(args_cli.openworldtactile_shear_clip_px),
            float(args_cli.openworldtactile_shear_clip_px),
        )

        force_map = env_normal.reshape(self.rows, self.cols).detach().cpu().numpy().astype(np.float32)
        displacement_x_map = displacement_x.reshape(self.rows, self.cols).detach().cpu().numpy().astype(np.float32)
        displacement_y_map = displacement_y.reshape(self.rows, self.cols).detach().cpu().numpy().astype(np.float32)
        contact_count = int(torch.count_nonzero(depth[env_id] > 0.0).item())
        total_force = float(torch.sum(env_normal).item())
        return ContactPadMaps(force_map, displacement_x_map, displacement_y_map, total_force, contact_count, None)


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
    """Render one virtual pad with OpenWorldTactile pressure color and SDF shear-driven texture displacement."""
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


def compute_force_map_pair(sdf_sampler: SdfTaxelPadSampler):
    """Compute left and right SDF-taxel virtual pad force maps for the selected environment."""
    return sdf_sampler.compute_pair(args_cli.record_env_id)


_PICK_SM_STATE_NAMES = {
    0: "REST",
    1: "APPROACH_ABOVE_OBJECT",
    2: "APPROACH_OBJECT",
    3: "GRASP_OBJECT",
    4: "LIFT_OBJECT",
}


def maybe_print_grasp_debug(
    step_count: int,
    pick_sm: PickAndLiftSm,
    tcp_position: torch.Tensor,
    object_position: torch.Tensor,
    object_grasp_position: torch.Tensor,
    desired_lift_position: torch.Tensor,
    initial_object_position: torch.Tensor,
    finger_joint_pos: torch.Tensor,
    gripper_action: torch.Tensor,
    left_result: ContactPadMaps,
    right_result: ContactPadMaps,
):
    """Print one-line diagnostics for tuning the nut grasp."""
    if not args_cli.debug_grasp or step_count % args_cli.debug_grasp_every != 0:
        return

    env_id = args_cli.record_env_id
    sm_state = int(pick_sm.sm_state[env_id].item())
    object_delta_z = float((object_position[env_id, 2] - initial_object_position[env_id, 2]).item())
    grasp_delta = tcp_position[env_id] - object_grasp_position[env_id]
    grasp_error = float(torch.linalg.norm(grasp_delta).item())
    grasp_xy_error = float(torch.linalg.norm(grasp_delta[:2]).item())
    grasp_z_error = float(grasp_delta[2].item())
    lift_error = float(torch.linalg.norm(tcp_position[env_id] - desired_lift_position[env_id]).item())
    env_finger_joint_pos = finger_joint_pos[env_id]
    finger_gap = float(torch.sum(env_finger_joint_pos).item()) if env_finger_joint_pos.numel() else 0.0
    finger_joint_text = ",".join(f"{float(value):.4f}" for value in env_finger_joint_pos.detach().cpu().tolist())
    lifted = object_delta_z >= args_cli.grasp_success_height
    print(
        f"[GRASP_DEBUG] step={step_count} "
        f"state={_PICK_SM_STATE_NAMES.get(sm_state, str(sm_state))} "
        f"gripper_cmd={float(gripper_action[env_id].item()):.1f} "
        f"object_z={float(object_position[env_id, 2].item()):.4f}m "
        f"object_dz={object_delta_z:.4f}m "
        f"tcp_z={float(tcp_position[env_id, 2].item()):.4f}m "
        f"grasp_err={grasp_error:.4f}m "
        f"grasp_xy_err={grasp_xy_error:.4f}m "
        f"grasp_z_err={grasp_z_error:.4f}m "
        f"lift_err={lift_error:.4f}m "
        f"finger_gap={finger_gap:.4f}m "
        f"finger_joints=[{finger_joint_text}] "
        f"left_force={left_result.total_force:.5f}N "
        f"right_force={right_result.total_force:.5f}N "
        f"left_taxels={left_result.contact_count} "
        f"right_taxels={right_result.contact_count} "
        f"lifted={lifted}"
    )


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
    sdf_sampler: SdfTaxelPadSampler,
    step_count: int,
    saved_count: int,
    pick_sm: PickAndLiftSm,
    output_dir: Path,
    left_openworldtactile_renderer: OpenWorldTactileRGBRenderer,
    right_openworldtactile_renderer: OpenWorldTactileRGBRenderer,
) -> int:
    """Save SDF-taxel force maps and textured OpenWorldTactile RGB images during grasp and lift states."""
    if args_cli.max_saved_frames == 0:
        return saved_count
    if args_cli.save_buffer_size <= 0 and args_cli.max_saved_frames > 0 and saved_count >= args_cli.max_saved_frames:
        return saved_count
    if step_count % args_cli.save_every != 0:
        return saved_count
    if pick_sm.sm_state[args_cli.record_env_id].item() < 3:
        return saved_count

    left_result, right_result = compute_force_map_pair(sdf_sampler)

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
        f"[SDF_TAXEL_RGB] frame={frame_id} "
        f"left_total={left_result.total_force:.6f}N left_sum={left_sum:.6f}N "
        f"left_rgb_scale={force_map_rgb_scale(left_result.force_map):.6f}N "
        f"left_active_taxels={left_result.contact_count} left_nonzero={int(np.count_nonzero(left_result.force_map))} "
        f"right_total={right_result.total_force:.6f}N right_sum={right_sum:.6f}N "
        f"right_rgb_scale={force_map_rgb_scale(right_result.force_map):.6f}N "
        f"right_active_taxels={right_result.contact_count} right_nonzero={int(np.count_nonzero(right_result.force_map))}"
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
    if args_cli.debug_grasp_every <= 0:
        raise ValueError("--debug_grasp_every must be positive.")
    if args_cli.grasp_success_height < 0.0:
        raise ValueError("--grasp_success_height must be non-negative.")
    if any(scale <= 0.0 for scale in args_cli.sdf_object_scale):
        raise ValueError("--sdf_object_scale values must be positive.")
    if args_cli.sdf_object_mass <= 0.0:
        raise ValueError("--sdf_object_mass must be positive.")
    if args_cli.sdf_object_static_friction < 0.0:
        raise ValueError("--sdf_object_static_friction must be non-negative.")
    if args_cli.sdf_object_dynamic_friction < 0.0:
        raise ValueError("--sdf_object_dynamic_friction must be non-negative.")
    if args_cli.sdf_object_solver_position_iterations <= 0:
        raise ValueError("--sdf_object_solver_position_iterations must be positive.")
    if args_cli.sdf_object_solver_velocity_iterations < 0:
        raise ValueError("--sdf_object_solver_velocity_iterations must be non-negative.")
    if args_cli.robot_solver_position_iterations <= 0:
        raise ValueError("--robot_solver_position_iterations must be positive.")
    if args_cli.robot_solver_velocity_iterations < 0:
        raise ValueError("--robot_solver_velocity_iterations must be non-negative.")
    if args_cli.lift_height <= 0.0:
        raise ValueError("--lift_height must be positive.")
    if args_cli.finger_surface_offset < 0.0:
        raise ValueError("--finger_surface_offset must be non-negative.")
    if args_cli.approach_height <= 0.0:
        raise ValueError("--approach_height must be positive.")
    if args_cli.position_threshold <= 0.0:
        raise ValueError("--position_threshold must be positive.")
    if args_cli.approach_above_timeout <= 0.0:
        raise ValueError("--approach_above_timeout must be positive.")
    if args_cli.approach_object_timeout <= 0.0:
        raise ValueError("--approach_object_timeout must be positive.")
    if args_cli.grasp_wait_time <= 0.0:
        raise ValueError("--grasp_wait_time must be positive.")
    if args_cli.normal_contact_stiffness <= 0.0:
        raise ValueError("--normal_contact_stiffness must be positive.")
    if args_cli.tangential_stiffness < 0.0:
        raise ValueError("--tangential_stiffness must be non-negative.")
    if args_cli.friction_coefficient < 0.0:
        raise ValueError("--friction_coefficient must be non-negative.")

    rows, cols = pad_image_shape()
    if rows <= 0 or cols <= 0:
        raise ValueError("--virtual_pad_size values must be positive.")
    object_usd = resolve_sdf_object_usd()

    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    configure_robot_contact_solver(env_cfg)
    configure_sdf_object(env_cfg, object_usd)

    env_step_dt = env_cfg.sim.dt * env_cfg.decimation
    if args_cli.max_steps > 0:
        # Keep the scripted run from timing out before the requested capture window.
        env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.max_steps * env_step_dt + 1.0)

    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    initial_object_position = env.unwrapped.scene["object"].data.root_pos_w.clone() - env.unwrapped.scene.env_origins
    left_body_id = robot.find_bodies("panda_leftfinger")[0][0]
    right_body_id = robot.find_bodies("panda_rightfinger")[0][0]
    finger_joint_ids = robot.find_joints("panda_finger_joint.*")[0]
    sdf_sampler = SdfTaxelPadSampler(env, left_body_id, right_body_id)

    saving_enabled = args_cli.max_saved_frames != 0
    output_dir = Path(args_cli.output_dir)
    if saving_enabled:
        output_dir.mkdir(parents=True, exist_ok=True)

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0

    grasp_position_offset = torch.tensor(
        args_cli.grasp_position_offset, dtype=torch.float32, device=env.unwrapped.device
    ).unsqueeze(0)
    pick_sm = PickAndLiftSm(
        env_step_dt,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        position_threshold=args_cli.position_threshold,
        approach_height=args_cli.approach_height,
    )
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
        print(f"[INFO]: Saving SDF taxel force maps, textured OpenWorldTactile RGB images, and grayscale images under: {output_dir.resolve()}")
    else:
        print("[INFO]: Local SDF taxel RGB saving disabled by default. Use --max_saved_frames N to save frames.")
    print(f"[INFO]: SDF object USD: {object_usd}")
    print(f"[INFO]: SDF mesh path pattern: {sdf_sampler.sdf_mesh_path}")
    print(
        "[INFO]: SDF object physics "
        f"mass={args_cli.sdf_object_mass}kg "
        f"static_friction={args_cli.sdf_object_static_friction} "
        f"dynamic_friction={args_cli.sdf_object_dynamic_friction} "
        f"object_solver_iters=({args_cli.sdf_object_solver_position_iterations}, "
        f"{args_cli.sdf_object_solver_velocity_iterations}) "
        f"robot_solver_iters=({args_cli.robot_solver_position_iterations}, "
        f"{args_cli.robot_solver_velocity_iterations})."
    )
    print(
        "[INFO]: Franka nut grasp tuning "
        f"grasp_position_offset={tuple(args_cli.grasp_position_offset)} "
        f"lift_with_grasp_offset={args_cli.lift_with_grasp_offset} "
        f"lift_target_mode={args_cli.lift_target_mode} "
        f"lift_height={args_cli.lift_height} "
        f"approach_height={args_cli.approach_height} "
        f"position_threshold={args_cli.position_threshold} "
        f"approach_timeouts=({args_cli.approach_above_timeout}, {args_cli.approach_object_timeout}) "
        f"grasp_wait_time={args_cli.grasp_wait_time}."
    )
    if args_cli.debug_grasp:
        print(
            "[INFO]: Grasp debug "
            f"every={args_cli.debug_grasp_every} step(s) "
            f"success_height={args_cli.grasp_success_height}m."
        )
    print(
        "[INFO]: Virtual pad grid "
        f"rows={rows} cols={cols} taxels={sdf_sampler.num_taxels} size={tuple(args_cli.virtual_pad_size)} "
        f"resolution={args_cli.virtual_pad_resolution} z_offset={args_cli.virtual_pad_z_offset}."
    )
    print(
        "[INFO]: Force map model: SDF taxel penalty force field. "
        f"normal_contact_stiffness={args_cli.normal_contact_stiffness} "
        f"tangential_stiffness={args_cli.tangential_stiffness} "
        f"friction_coefficient={args_cli.friction_coefficient} "
        f"rgb_force_full_scale={args_cli.rgb_force_full_scale}."
    )
    print(
        "[INFO]: OpenWorldTactile textured RGB renderer "
        f"texture_strength={args_cli.openworldtactile_texture_strength} "
        f"texture_blur={args_cli.openworldtactile_texture_blur} "
        f"pressure_blur={args_cli.openworldtactile_pressure_blur} "
        f"shear_px_per_n={args_cli.openworldtactile_shear_px_per_n} "
        f"shear_clip_px={args_cli.openworldtactile_shear_clip_px}."
    )
    if args_cli.show_rgb_maps:
        print(
            "[INFO]: Live SDF taxel OpenWorldTactile textured RGB + OpenWorldTactile SDK fxyz preview enabled in Isaac UI "
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
                object_grasp_position = object_position + grasp_position_offset

                if args_cli.lift_target_mode == "command":
                    desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]
                else:
                    desired_position = initial_object_position.clone()
                    desired_position[:, 2] = initial_object_position[:, 2] + args_cli.lift_height
                desired_lift_position = (
                    desired_position + grasp_position_offset if args_cli.lift_with_grasp_offset else desired_position
                )

                actions = pick_sm.compute(
                    torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                    torch.cat([object_grasp_position, desired_orientation], dim=-1),
                    torch.cat([desired_lift_position, desired_orientation], dim=-1),
                )

                left_result = None
                right_result = None
                if args_cli.show_rgb_maps and step_count % args_cli.force_preview_every == 0:
                    left_result, right_result = compute_force_map_pair(sdf_sampler)
                    show_rgb_pair(
                        left_result,
                        right_result,
                        left_openworldtactile_renderer,
                        right_openworldtactile_renderer,
                        left_openworldtactile_bridge,
                        right_openworldtactile_bridge,
                    )

                if args_cli.debug_grasp and step_count % args_cli.debug_grasp_every == 0:
                    if left_result is None or right_result is None:
                        left_result, right_result = compute_force_map_pair(sdf_sampler)
                    finger_joint_pos = robot.data.joint_pos[:, finger_joint_ids]
                    maybe_print_grasp_debug(
                        step_count,
                        pick_sm,
                        tcp_rest_position,
                        object_position,
                        object_grasp_position,
                        desired_lift_position,
                        initial_object_position,
                        finger_joint_pos,
                        actions[:, -1],
                        left_result,
                        right_result,
                    )

                saved_count = maybe_save_rgb_pair(
                    sdf_sampler,
                    step_count,
                    saved_count,
                    pick_sm,
                    output_dir,
                    left_openworldtactile_renderer,
                    right_openworldtactile_renderer,
                )
                if args_cli.save_buffer_size <= 0 and args_cli.max_saved_frames > 0 and saved_count >= args_cli.max_saved_frames:
                    break

                if dones.any():
                    reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    pick_sm.reset_idx(reset_ids)
                    object_data = env.unwrapped.scene["object"].data
                    initial_object_position[reset_ids] = (
                        object_data.root_pos_w[reset_ids] - env.unwrapped.scene.env_origins[reset_ids]
                    )

            step_count += 1

        if saving_enabled:
            print(f"[INFO]: Saved {saved_count} SDF-taxel textured OpenWorldTactile RGB + grayscale frame pairs.")
        else:
            print("[INFO]: Finished without saving local SDF-taxel RGB files.")
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
