# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
触觉器件是 VisuoTactileSensor，包含：elastomer 指尖
内部 tactile camera
OpenWorldTactile RGB renderer
可选 SDF force field

被接触物体
支持 cube / nut / none。
nut 用 Factory/factory_nut_m16.usd，有 SDF mesh，可以启用 SDF force field。
cube 是普通 Shape Prim cube，没有 SDF mesh，所以脚本里明确关闭 force field：cube: 只用 camera depth
nut: camera depth + SDF shear force

RGB 生成逻辑
先获取无接触 baseline depth。
每帧读取 tactile camera 的 depth：diff = nominal_depth - current_depth

diff 作为压力/高度图输入 renderer。
如果是 nut 且启用 force field：SDF shear_force -> displacement map

renderer 生成 RGB：diff -> 颜色变化/fz
displacement -> 纹理位移/fxy
"""
"""
.. code-block:: bash

    ./isaaclab.sh -p experiments/franka/current/tactile_camera_openworldtactile.py --headless

"""

"""Launch Omniverse Toolkit first."""

import argparse
import math

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick/lift DexCube with Franka and save tactile RGB images.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=800, help="Maximum environment steps before exiting.")
parser.add_argument("--output_dir", type=str, default="outputs/franka_lift_tactile_rgb", help="Directory for PNGs.")
parser.add_argument("--save_every", type=int, default=5, help="Save every N steps while grasping/lifting.")
parser.add_argument("--max_saved_frames", type=int, default=80, help="Stop saving after this many frame pairs.")
parser.add_argument("--record_env_id", type=int, default=0, help="Environment index to save.")
parser.add_argument(
    "--openworldtactile_max_pressure",
    type=float,
    default=6e-4,
    help="Depth-difference value that saturates tactile RGB mapping.",
)
parser.add_argument(
    "--rgb_mapping",
    type=str,
    default="wavelength",
    choices=["wavelength", "openworldtactile"],
    help="RGB image mapping used for saved left/right PNGs.",
)
parser.add_argument(
    "--wavelength_high_nm",
    type=float,
    default=600.0,
    help="No-contact/low-pressure wavelength for --rgb_mapping wavelength.",
)
parser.add_argument(
    "--wavelength_low_nm",
    type=float,
    default=400.0,
    help="Saturated/high-pressure wavelength for --rgb_mapping wavelength.",
)
parser.add_argument(
    "--rgb_pressure_scale",
    type=float,
    default=2.0,
    help="When auto-scaling wavelength RGB, use max_pressure = frame_diff_max * this value.",
)
parser.add_argument(
    "--disable_rgb_auto_scale",
    action="store_true",
    help="Disable per-frame wavelength RGB auto-scaling and use --openworldtactile_max_pressure directly.",
)
parser.add_argument(
    "--gelsight_contact_center",
    type=float,
    nargs=3,
    default=(0.0, -0.010, 0.06776),
    help="GelSight local coordinates of the sensitive contact-center used for mounting.",
)
parser.add_argument(
    "--pad_side_offset",
    type=float,
    default=0.015,
    help="Desired sensitive-pad center side offset in each Panda finger frame.",
)
parser.add_argument("--pad_z_offset", type=float, default=0.046, help="Desired sensitive-pad center Z offset.")
parser.add_argument("--mount_offset", type=float, nargs=3, default=None, help="Override left GelSight root offset.")
parser.add_argument("--right_mount_offset", type=float, nargs=3, default=None, help="Override right GelSight root offset.")
parser.add_argument("--mount_rot_rpy", type=float, nargs=3, default=None, help="Override left GelSight root RPY.")
parser.add_argument("--right_mount_rot_rpy", type=float, nargs=3, default=None, help="Override right GelSight root RPY.")
parser.add_argument(
    "--use_mounted_gelsight_usd",
    action="store_true",
    help="Experimental: add two full OpenWorldTactile GelSight USD fingers and pose-follow them to Franka fingers.",
)
parser.add_argument(
    "--enable_gelsight_collision",
    action="store_true",
    help="Let the mounted GelSight elastomers participate in physics contact. Off by default for arm stability.",
)
parser.add_argument("--disable_gelsight_collision", action="store_true", help="Explicitly disable mounted GelSight collisions.")
parser.add_argument(
    "--keep_panda_finger_collision",
    action="store_true",
    help="Compatibility flag; bare Panda finger collisions are kept by default.",
)
parser.add_argument(
    "--disable_panda_finger_collision",
    action="store_true",
    help="Disable bare Panda finger collisions when GelSight collision is enabled.",
)
parser.add_argument(
    "--gelsight_compliance_stiffness",
    type=float,
    default=None,
    help="Optional compliant-contact stiffness override. Default uses the GelSight USD values.",
)
parser.add_argument(
    "--gelsight_compliant_damping",
    type=float,
    default=None,
    help="Optional compliant-contact damping override. Default uses the GelSight USD values.",
)
parser.add_argument("--gelsight_contact_offset", type=float, default=0.0005, help="Mounted GelSight contact offset.")
parser.add_argument("--gelsight_rest_offset", type=float, default=0.0, help="Mounted GelSight rest offset.")
parser.add_argument(
    "--gelsight_max_depenetration_velocity",
    type=float,
    default=0.5,
    help="Mounted GelSight max depenetration velocity.",
)
parser.add_argument("--camera_near", type=float, default=0.002, help="Franka camera-only tactile near clipping distance.")
parser.add_argument("--camera_far", type=float, default=0.08, help="Franka camera-only tactile far clipping distance.")
parser.add_argument(
    "--finger_surface_offset",
    type=float,
    default=0.012,
    help="Franka camera-only side offset from each finger link center toward the gripper gap.",
)
parser.add_argument("--save_debug_depth", action="store_true", help="Save depth-difference debug images.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

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

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_from_euler_xyz

import isaaclab_tasks  # noqa: F401
from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.openworldtactile_sensor import OWT_ASSET_ROOT, VisuoTactileSensorCfg
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from pxr import Sdf, UsdPhysics

# initialize warp
wp.init()

GELSIGHT_USD = f"{OWT_ASSET_ROOT}/gelsight_r15_finger/gelsight_r15_finger.usd"
LEFT_MOUNT_RPY = (0.0, 0.0, 0.0)
RIGHT_MOUNT_RPY = (0.0, 0.0, math.pi)


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


def make_gelsight_cfg(prim_path: str) -> ArticulationCfg:
    """Create a OpenWorldTactile GelSight finger USD that is pose-followed to a Panda finger."""
    collision_enabled = args_cli.enable_gelsight_collision and not args_cli.disable_gelsight_collision
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path=GELSIGHT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=args_cli.gelsight_max_depenetration_velocity,
            ),
            compliant_contact_stiffness=args_cli.gelsight_compliance_stiffness if collision_enabled else None,
            compliant_contact_damping=args_cli.gelsight_compliant_damping if collision_enabled else None,
            physics_material_prim_path="elastomer",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=collision_enabled,
                contact_offset=args_cli.gelsight_contact_offset,
                rest_offset=args_cli.gelsight_rest_offset,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3), joint_pos={}, joint_vel={}),
        actuators={},
    )


def make_tactile_sensor_cfg(root_prim_path: str) -> VisuoTactileSensorCfg:
    """Create the same camera-based OpenWorldTactile RGB sensor used by the standalone demo."""
    return VisuoTactileSensorCfg(
        prim_path=f"{root_prim_path}/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG.replace(
            openworldtactile_max_pressure=args_cli.openworldtactile_max_pressure,
            openworldtactile_base_value=220,
            openworldtactile_pressure_blur=5,
            openworldtactile_displacement_scale=12000.0,
        ),
        enable_camera_tactile=True,
        enable_force_field=False,
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/Object",
        camera_cfg=TiledCameraCfg(
            prim_path=f"{root_prim_path}/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )


def make_franka_camera_tactile_sensor_cfg(
    name: str,
    finger_link: str,
    inward_rot: tuple[float, float, float, float],
    offset_pos: tuple[float, float, float],
) -> VisuoTactileSensorCfg:
    """Create a lightweight camera-only OpenWorldTactile sensor on a Panda finger link."""
    return VisuoTactileSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{finger_link}/{name}_tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG.replace(
            openworldtactile_max_pressure=args_cli.openworldtactile_max_pressure,
            openworldtactile_base_value=220,
            openworldtactile_pressure_blur=5,
            openworldtactile_displacement_scale=12000.0,
        ),
        enable_camera_tactile=True,
        enable_force_field=False,
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/Object",
        camera_cfg=TiledCameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{finger_link}/{name}_tactile_cam",
            update_period=0.0,
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            depth_clipping_behavior="max",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=0.05,
                horizontal_aperture=12.0,
                clipping_range=(args_cli.camera_near, args_cli.camera_far),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=offset_pos,
                rot=inward_rot,
                convention="ros",
            ),
        ),
    )


def add_tactile_sensors(env_cfg: LiftEnvCfg):
    """Add left and right tactile sensors without disturbing the stock Franka grasp by default."""
    if args_cli.use_mounted_gelsight_usd:
        env_cfg.scene.left_gelsight = make_gelsight_cfg("{ENV_REGEX_NS}/LeftGelSight")
        env_cfg.scene.right_gelsight = make_gelsight_cfg("{ENV_REGEX_NS}/RightGelSight")
        env_cfg.scene.left_tactile = make_tactile_sensor_cfg("{ENV_REGEX_NS}/LeftGelSight")
        env_cfg.scene.right_tactile = make_tactile_sensor_cfg("{ENV_REGEX_NS}/RightGelSight")
        return

    side_offset = args_cli.finger_surface_offset
    env_cfg.scene.left_tactile = make_franka_camera_tactile_sensor_cfg(
        name="left",
        finger_link="panda_leftfinger",
        inward_rot=(0.7071068, 0.7071068, 0.0, 0.0),
        offset_pos=(0.0, -side_offset, 0.046),
    )
    env_cfg.scene.right_tactile = make_franka_camera_tactile_sensor_cfg(
        name="right",
        finger_link="panda_rightfinger",
        inward_rot=(0.7071068, -0.7071068, 0.0, 0.0),
        offset_pos=(0.0, side_offset, 0.046),
    )


def add_filtered_pair(stage, prim_path_a: str, prim_path_b: str) -> bool:
    """Filter collisions between two prims if both exist."""
    prim_a = stage.GetPrimAtPath(prim_path_a)
    prim_b = stage.GetPrimAtPath(prim_path_b)
    if not prim_a.IsValid() or not prim_b.IsValid():
        return False
    filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(prim_a)
    filtered_pairs_api.CreateFilteredPairsRel().AddTarget(Sdf.Path(prim_path_b))
    return True


def find_collision_prim_paths(stage, root_path: str) -> list[str]:
    """Return collider prim paths under a subtree."""
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return []

    collision_paths: list[str] = []
    prim_queue = [root_prim]
    while prim_queue:
        prim = prim_queue.pop(0)
        if UsdPhysics.CollisionAPI(prim):
            collision_paths.append(prim.GetPath().pathString)
        prim_queue.extend(prim.GetChildren())
    return collision_paths


def add_filtered_collision_subtrees(stage, prim_path_a: str, prim_path_b: str) -> int:
    """Filter every collider under two subtrees."""
    collision_paths_a = find_collision_prim_paths(stage, prim_path_a)
    collision_paths_b = find_collision_prim_paths(stage, prim_path_b)
    filtered_pairs = 0

    for collider_path_a in collision_paths_a:
        for collider_path_b in collision_paths_b:
            if add_filtered_pair(stage, collider_path_a, collider_path_b):
                filtered_pairs += 1
    return filtered_pairs


def configure_gelsight_contact_physics(env):
    """Filter Robot/GelSight self-collisions and make the GelSight pads the contact surface."""
    scene = env.unwrapped.scene
    if not args_cli.use_mounted_gelsight_usd:
        print("[INFO]: Mounted GelSight USD disabled; using lightweight Franka finger cameras only.")
        return

    collision_enabled = args_cli.enable_gelsight_collision and not args_cli.disable_gelsight_collision
    if not collision_enabled:
        print(
            "[INFO]: Mounted GelSight collision disabled for arm stability; "
            "Panda finger collisions are kept for the stock grasp."
        )
        return

    filtered_pairs = 0
    disabled_fingers = 0
    for env_path in scene.env_prim_paths:
        robot_path = f"{env_path}/Robot"
        left_gelsight_path = f"{env_path}/LeftGelSight"
        right_gelsight_path = f"{env_path}/RightGelSight"
        filtered_pairs += add_filtered_collision_subtrees(scene.stage, left_gelsight_path, robot_path)
        filtered_pairs += add_filtered_collision_subtrees(scene.stage, right_gelsight_path, robot_path)

        if args_cli.disable_panda_finger_collision and not args_cli.keep_panda_finger_collision:
            for finger_path in (f"{robot_path}/panda_leftfinger", f"{robot_path}/panda_rightfinger"):
                sim_utils.modify_collision_properties(
                    finger_path, sim_utils.CollisionPropertiesCfg(collision_enabled=False)
                )
                disabled_fingers += 1

    print(
        f"[INFO]: Mounted GelSight collision enabled; Robot-GelSight filtered pairs={filtered_pairs}, "
        f"disabled bare Panda finger colliders={disabled_fingers}. "
        "If the object starts flying, rerun without --enable_gelsight_collision."
    )


def get_mount_quat(side: str, device: torch.device, num_envs: int) -> torch.Tensor:
    """Return the GelSight root orientation in the corresponding Panda finger frame."""
    if side == "left":
        rpy = tuple(args_cli.mount_rot_rpy) if args_cli.mount_rot_rpy is not None else LEFT_MOUNT_RPY
    elif side == "right":
        rpy = tuple(args_cli.right_mount_rot_rpy) if args_cli.right_mount_rot_rpy is not None else RIGHT_MOUNT_RPY
    else:
        raise ValueError(f"Unknown GelSight side: {side}")

    roll = torch.full((num_envs,), rpy[0], dtype=torch.float32, device=device)
    pitch = torch.full((num_envs,), rpy[1], dtype=torch.float32, device=device)
    yaw = torch.full((num_envs,), rpy[2], dtype=torch.float32, device=device)
    return quat_from_euler_xyz(roll, pitch, yaw)


def get_mount_offset(
    side: str,
    mount_quat_b: torch.Tensor,
    device: torch.device,
    num_envs: int,
) -> torch.Tensor:
    """Return the GelSight root offset that places the sensitive pad on the Panda finger pad."""
    override = args_cli.mount_offset if side == "left" else args_cli.right_mount_offset
    if override is not None:
        return torch.tensor(override, dtype=torch.float32, device=device).repeat(num_envs, 1)

    side_sign = -1.0 if side == "left" else 1.0
    desired_pad_center_b = torch.tensor(
        (0.0, side_sign * args_cli.pad_side_offset, args_cli.pad_z_offset),
        dtype=torch.float32,
        device=device,
    ).repeat(num_envs, 1)
    gelsight_contact_center = torch.tensor(
        args_cli.gelsight_contact_center,
        dtype=torch.float32,
        device=device,
    ).repeat(num_envs, 1)
    return desired_pad_center_b - quat_apply(mount_quat_b, gelsight_contact_center)


def create_mount_state(env) -> dict[str, int | torch.Tensor]:
    """Resolve Franka finger body ids and repeated left/right mount transforms."""
    scene = env.unwrapped.scene
    robot: Articulation = scene["robot"]
    num_envs = env.unwrapped.num_envs
    left_body_id = robot.find_bodies("panda_leftfinger")[0][0]
    right_body_id = robot.find_bodies("panda_rightfinger")[0][0]
    left_mount_quat_b = get_mount_quat("left", robot.device, num_envs)
    right_mount_quat_b = get_mount_quat("right", robot.device, num_envs)
    left_mount_offset_b = get_mount_offset("left", left_mount_quat_b, robot.device, num_envs)
    right_mount_offset_b = get_mount_offset("right", right_mount_quat_b, robot.device, num_envs)

    print(
        "[INFO]: Mounted GelSight pose defaults: "
        f"left_offset={tuple(left_mount_offset_b[0].detach().cpu().tolist())}, "
        f"right_offset={tuple(right_mount_offset_b[0].detach().cpu().tolist())}."
    )
    return {
        "left_body_id": left_body_id,
        "right_body_id": right_body_id,
        "left_mount_quat_b": left_mount_quat_b,
        "right_mount_quat_b": right_mount_quat_b,
        "left_mount_offset_b": left_mount_offset_b,
        "right_mount_offset_b": right_mount_offset_b,
    }


def update_gelsight_mount(
    robot: Articulation,
    gelsight: Articulation,
    finger_body_id: int,
    mount_offset_b: torch.Tensor,
    mount_quat_b: torch.Tensor,
) -> torch.Tensor:
    """Pose-follow one mounted GelSight root to one Panda finger body."""
    mount_pos_w, mount_quat_w = combine_frame_transforms(
        robot.data.body_pos_w[:, finger_body_id],
        robot.data.body_quat_w[:, finger_body_id],
        mount_offset_b,
        mount_quat_b,
    )
    mount_pose_w = torch.cat((mount_pos_w, mount_quat_w), dim=-1)
    gelsight.write_root_pose_to_sim(mount_pose_w)
    gelsight.write_root_velocity_to_sim(robot.data.body_vel_w[:, finger_body_id].clone())
    return mount_pose_w


def sync_gelsight_mounts(env, mount_state: dict[str, int | torch.Tensor]):
    """Pose-follow both mounted GelSight USDs to the current Franka finger poses."""
    scene = env.unwrapped.scene
    robot: Articulation = scene["robot"]
    update_gelsight_mount(
        robot,
        scene["left_gelsight"],
        mount_state["left_body_id"],
        mount_state["left_mount_offset_b"],
        mount_state["left_mount_quat_b"],
    )
    update_gelsight_mount(
        robot,
        scene["right_gelsight"],
        mount_state["right_body_id"],
        mount_state["right_mount_offset_b"],
        mount_state["right_mount_quat_b"],
    )


def capture_nominal_tactile(env):
    """Capture the no-contact baseline used by tactile RGB rendering."""
    env.unwrapped.scene["left_tactile"].get_initial_render()
    env.unwrapped.scene["right_tactile"].get_initial_render()


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

    # Dim spectrum edges gently so 400 nm reads as purple instead of fully saturated blue.
    factor = np.ones_like(wl, dtype=np.float32)
    factor = np.where((wl >= 380.0) & (wl < 420.0), 0.3 + 0.7 * (wl - 380.0) / 40.0, factor)
    factor = np.where((wl > 700.0) & (wl <= 780.0), 0.3 + 0.7 * (780.0 - wl) / 80.0, factor)
    rgb *= np.clip(factor, 0.0, 1.0)[..., None]

    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def rgb_max_pressure_from_diff(diff: np.ndarray) -> float:
    """Return the pressure/depth scale used by the saved RGB mapping."""
    if args_cli.disable_rgb_auto_scale:
        return args_cli.openworldtactile_max_pressure

    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    diff_max = float(np.max(np.clip(diff, 0.0, None))) if diff.size else 0.0
    if diff_max > 1.0e-9:
        return diff_max * args_cli.rgb_pressure_scale
    return args_cli.openworldtactile_max_pressure


def depth_diff_to_wavelength_rgb(diff: np.ndarray, max_pressure: float | None = None) -> np.ndarray:
    """Map tactile depth difference to wavelength RGB from high nm to low nm."""
    if max_pressure is None:
        max_pressure = rgb_max_pressure_from_diff(diff)
    diff = np.nan_to_num(diff, nan=0.0, posinf=max_pressure, neginf=0.0)
    pressure = np.clip(diff, 0.0, None) / max(max_pressure, 1.0e-9)
    pressure = np.clip(pressure, 0.0, 1.0)
    wavelength = args_cli.wavelength_high_nm - pressure * (args_cli.wavelength_high_nm - args_cli.wavelength_low_nm)
    return wavelength_to_rgb(wavelength)


def tactile_rgb_to_numpy(sensor, env_id: int) -> np.ndarray:
    """Return one saved tactile RGB image."""
    if args_cli.rgb_mapping == "wavelength":
        diff = tactile_depth_diff(sensor, env_id)
        image_np = depth_diff_to_wavelength_rgb(diff, rgb_max_pressure_from_diff(diff))
        return np.ascontiguousarray(np.transpose(image_np, axes=(1, 0, 2)))

    image = sensor.data.tactile_rgb_image[env_id].detach().cpu()
    if image.dtype != torch.uint8:
        image = (image * 255).to(torch.uint8) if image.max() <= 1.0 else image.to(torch.uint8)
    image_np = image.numpy()
    # Match the OpenWorldTactile demo save convention, which swaps the sensor H/W axes for PNG output.
    return np.ascontiguousarray(np.transpose(image_np, axes=(1, 0, 2)))


def save_rgb(path: Path, rgb: np.ndarray):
    """Save an RGB image through OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_depth_debug(path: Path, sensor, env_id: int):
    """Save a normalized depth-difference image for camera placement debugging."""
    diff = tactile_depth_diff(sensor, env_id)
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    diff = np.clip(diff, 0.0, None)
    diff = np.clip(diff / max(rgb_max_pressure_from_diff(diff), 1.0e-6), 0.0, 1.0)
    image = (diff * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def tactile_depth_diff(sensor, env_id: int) -> np.ndarray:
    """Return nominal-current tactile depth difference for one environment."""
    depth_key = "distance_to_image_plane"
    depth = sensor.data.tactile_depth_image[env_id].detach().cpu()
    nominal = sensor._nominal_tactile[depth_key][env_id].detach().cpu()
    return (nominal - depth).squeeze(-1).numpy()


def print_depth_stats(frame_id: str, left_sensor, right_sensor):
    """Print compact depth-difference stats for placement debugging."""
    left_diff = tactile_depth_diff(left_sensor, args_cli.record_env_id)
    right_diff = tactile_depth_diff(right_sensor, args_cli.record_env_id)
    left_max = float(np.nanmax(left_diff))
    right_max = float(np.nanmax(right_diff))
    left_rgb_pressure = rgb_max_pressure_from_diff(left_diff)
    right_rgb_pressure = rgb_max_pressure_from_diff(right_diff)
    if args_cli.rgb_mapping == "wavelength":
        left_rgb_mean = float(depth_diff_to_wavelength_rgb(left_diff, left_rgb_pressure).mean())
        right_rgb_mean = float(depth_diff_to_wavelength_rgb(right_diff, right_rgb_pressure).mean())
    else:
        left_rgb_mean = float(left_sensor.data.tactile_rgb_image[args_cli.record_env_id].float().mean().item())
        right_rgb_mean = float(right_sensor.data.tactile_rgb_image[args_cli.record_env_id].float().mean().item())
    print(
        f"[TACTILE] frame={frame_id} "
        f"left_diff_max={left_max:.6f} right_diff_max={right_max:.6f} "
        f"rgb_mapping={args_cli.rgb_mapping} "
        f"left_rgb_max_pressure={left_rgb_pressure:.6f} right_rgb_max_pressure={right_rgb_pressure:.6f} "
        f"left_diff_mean={np.nanmean(left_diff):.6f} right_diff_mean={np.nanmean(right_diff):.6f} "
        f"left_rgb_mean={left_rgb_mean:.2f} right_rgb_mean={right_rgb_mean:.2f}"
    )


def maybe_save_tactile_pair(env, step_count: int, saved_count: int, pick_sm: PickAndLiftSm, output_dir: Path) -> int:
    """Save tactile RGB frames during grasp and lift states."""
    if saved_count >= args_cli.max_saved_frames:
        return saved_count
    if step_count % args_cli.save_every != 0:
        return saved_count
    if pick_sm.sm_state[args_cli.record_env_id].item() < 3:
        return saved_count

    left_sensor = env.unwrapped.scene["left_tactile"]
    right_sensor = env.unwrapped.scene["right_tactile"]
    left_rgb = tactile_rgb_to_numpy(left_sensor, args_cli.record_env_id)
    right_rgb = tactile_rgb_to_numpy(right_sensor, args_cli.record_env_id)

    frame_id = f"{saved_count:05d}"
    save_rgb(output_dir / "left" / f"{frame_id}.png", left_rgb)
    save_rgb(output_dir / "right" / f"{frame_id}.png", right_rgb)
    if args_cli.save_debug_depth:
        save_depth_debug(output_dir / "debug_depth_left" / f"{frame_id}.png", left_sensor, args_cli.record_env_id)
        save_depth_debug(output_dir / "debug_depth_right" / f"{frame_id}.png", right_sensor, args_cli.record_env_id)
    print_depth_stats(frame_id, left_sensor, right_sensor)
    return saved_count + 1


def main():
    if args_cli.save_every <= 0:
        raise ValueError("--save_every must be positive.")
    if args_cli.record_env_id < 0 or args_cli.record_env_id >= args_cli.num_envs:
        raise ValueError("--record_env_id must be in [0, num_envs).")

    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    add_tactile_sensors(env_cfg)

    # Keep the scripted run from timing out before the requested capture window.
    env_step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.max_steps * env_step_dt + 1.0)

    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    configure_gelsight_contact_physics(env)
    env.reset()
    mount_state = create_mount_state(env) if args_cli.use_mounted_gelsight_usd else None
    if mount_state is not None:
        sync_gelsight_mounts(env, mount_state)
        env.unwrapped.scene.write_data_to_sim()
    capture_nominal_tactile(env)

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0

    pick_sm = PickAndLiftSm(env_step_dt, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.01)
    output_dir = Path(args_cli.output_dir)
    saved_count = 0

    print(f"[INFO]: Saving tactile RGB pairs under: {output_dir.resolve()}")
    if args_cli.use_mounted_gelsight_usd:
        print(
            "[INFO]: Experimental mounted GelSight USD mode is enabled; "
            f"saved RGB uses {args_cli.rgb_mapping} mapping."
        )
    else:
        print(
            "[INFO]: Stable mode: no GelSight USD articulation is added; "
            f"saved RGB uses lightweight Franka finger camera depth diff with {args_cli.rgb_mapping} mapping."
        )
    if args_cli.rgb_mapping == "wavelength" and not args_cli.disable_rgb_auto_scale:
        print(f"[INFO]: Wavelength RGB auto-scale enabled: max_pressure = frame_diff_max * {args_cli.rgb_pressure_scale}.")

    step_count = 0
    while simulation_app.is_running() and step_count < args_cli.max_steps:
        with torch.inference_mode():
            if mount_state is not None:
                sync_gelsight_mounts(env, mount_state)
            dones = env.step(actions)[-2]
            if mount_state is not None:
                sync_gelsight_mounts(env, mount_state)

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

            saved_count = maybe_save_tactile_pair(env, step_count, saved_count, pick_sm, output_dir)
            if saved_count >= args_cli.max_saved_frames:
                break

            if dones.any():
                reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                pick_sm.reset_idx(reset_ids)
                if mount_state is not None:
                    sync_gelsight_mounts(env, mount_state)
                    env.unwrapped.scene.write_data_to_sim()
                capture_nominal_tactile(env)

        step_count += 1

    print(f"[INFO]: Saved {saved_count} tactile RGB frame pairs.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
